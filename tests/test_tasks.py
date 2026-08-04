import io
import json
from types import SimpleNamespace

import pytest
import requests
from PIL import Image
from rq.exceptions import NoSuchJobError

import tasks
from images import InvalidShapeError
from provider import ProviderStatus


class FailedProvider:
    def get_status(self, task_id):
        return ProviderStatus(
            task_id=task_id,
            status="failed",
            progress=10,
            result_url=None,
            content=None,
            failure_reason="error",
            error="provider error",
        )


class SucceededProvider:
    def get_status(self, task_id):
        return ProviderStatus(
            task_id=task_id,
            status="succeeded",
            progress=100,
            result_url="https://example.invalid/result.png",
            content=None,
            failure_reason=None,
            error=None,
        )

    def download_result(self, result_url):
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 1024), "white").save(
            buffer,
            format="PNG",
        )
        return buffer.getvalue()


class RunningProvider:
    def get_status(self, task_id):
        return ProviderStatus(
            task_id=task_id,
            status="running",
            progress=80,
            result_url=None,
            content=None,
            failure_reason=None,
            error=None,
        )


class DownloadTimeoutProvider(SucceededProvider):
    def download_result(self, result_url):
        raise requests.ReadTimeout("result CDN stalled")


def poll_settings(**overrides):
    values = {
        "max_wait": 320,
        "hard_wait": 1800,
        "poll_interval": 10,
        "delayed_poll_interval": 60,
        "result_recovery_window": 5400,
        "recovery_retry_intervals": (2, 5, 15, 30, 60),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_timed_step_logs_duration_size_and_outcome(monkeypatch):
    messages = []
    clock = iter((10.0, 10.125))
    monkeypatch.setattr(tasks.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        tasks.logger,
        "info",
        lambda message: messages.append(message),
    )

    with tasks.timed_step("s3_download", "log-timing") as timing:
        timing["size_mb"] = tasks.size_mb(2 * 1024 * 1024)

    payload = json.loads(messages[0])
    assert payload == {
        "event": "real_to_render_step",
        "stage": "real_to_render",
        "step": "s3_download",
        "log_id": "log-timing",
        "duration_seconds": 0.125,
        "size_mb": 2.0,
        "outcome": "succeeded",
    }


def test_timed_step_logs_failed_operation(monkeypatch):
    messages = []
    clock = iter((20.0, 20.25))
    monkeypatch.setattr(tasks.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        tasks.logger,
        "info",
        lambda message: messages.append(message),
    )

    with pytest.raises(TimeoutError):
        with tasks.timed_step("provider_submit_api", "log-error"):
            raise TimeoutError("request timed out")

    payload = json.loads(messages[0])
    assert payload["duration_seconds"] == 0.25
    assert payload["outcome"] == "failed"
    assert payload["error_type"] == "TimeoutError"


def test_provider_failure_does_not_schedule_another_poll(monkeypatch):
    failures = []
    scheduled = []
    monkeypatch.setattr(
        tasks,
        "get_settings",
        poll_settings,
    )
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(tasks, "provider_client", lambda: FailedProvider())
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda *args, **kwargs: scheduled.append((args, kwargs)),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 100.0)

    tasks.poll_real_to_render(
        "log-1",
        True,
        "provider-1",
        submitted_at=0.0,
    )

    assert len(failures) == 1
    assert failures[0][1]["failure_reason"] == "error"
    assert scheduled == []


def test_hard_stage_timeout_fails_only_after_provider_still_reports_running(
    monkeypatch,
):
    failures = []
    monkeypatch.setattr(
        tasks,
        "get_settings",
        poll_settings,
    )
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: RunningProvider(),
    )
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("timeout must not schedule another poll")
        ),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 1801.0)

    tasks.poll_real_to_render(
        "log-2",
        False,
        "provider-2",
        submitted_at=0.0,
    )

    assert len(failures) == 1
    assert failures[0][1]["failure_reason"] == "provider_hard_timeout"


def test_invalid_shape_fails_before_upload_and_handoff(monkeypatch):
    failures = []
    monkeypatch.setattr(
        tasks,
        "get_settings",
        poll_settings,
    )
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(tasks, "provider_client", lambda: SucceededProvider())
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "validate_combined_render_shape",
        lambda content: (_ for _ in ()).throw(InvalidShapeError()),
    )
    monkeypatch.setattr(
        tasks,
        "object_storage",
        lambda: (_ for _ in ()).throw(
            AssertionError("invalid shape must not be uploaded")
        ),
    )
    monkeypatch.setattr(
        tasks,
        "enqueue_render_to_uv_once",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid shape must not be handed off")
        ),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 100.0)

    tasks.poll_real_to_render(
        "log-invalid-shape",
        True,
        "provider-invalid-shape",
        submitted_at=0.0,
    )

    assert len(failures) == 1
    assert isinstance(failures[0][0][1], InvalidShapeError)
    assert failures[0][1]["failure_reason"] == "invalid_shape"


def test_soft_timeout_keeps_polling_at_slower_interval(monkeypatch):
    scheduled = []
    failures = []
    monkeypatch.setattr(tasks, "get_settings", poll_settings)
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(tasks, "provider_client", lambda: RunningProvider())
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "report_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda **kwargs: scheduled.append(kwargs),
    )
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 321.0)

    tasks.poll_real_to_render(
        "log-delayed",
        True,
        "provider-delayed",
        submitted_at=0.0,
    )

    assert failures == []
    assert scheduled[0]["delay_seconds"] == 60


def test_result_download_timeout_is_retried_without_failing(monkeypatch):
    scheduled = []
    failures = []
    reports = []
    monkeypatch.setattr(tasks, "get_settings", poll_settings)
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: DownloadTimeoutProvider(),
    )
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "report_status",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda **kwargs: scheduled.append(kwargs),
    )
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 100.0)

    tasks.poll_real_to_render(
        "log-download-retry",
        True,
        "provider-download-retry",
        submitted_at=0.0,
    )

    assert failures == []
    assert scheduled[-1]["poll_number"] == 2
    assert scheduled[-1]["delay_seconds"] == 2
    assert reports[-1][1]["error_origin"] == "provider_result_download"
    assert reports[-1][1]["retrying"] is True


def test_result_url_refresh_preserves_original_recovery_deadline(monkeypatch):
    saved = []
    scheduled = []
    state = {
        "provider_status": "refresh_result_url",
        "result_url": "",
        "result_deadline": 200.0,
        "result_url_obtained_at": 50.0,
    }

    class ExpiredResultProvider(SucceededProvider):
        def download_result(self, result_url):
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError("expired result URL", response=response)

    monkeypatch.setattr(tasks, "get_settings", poll_settings)
    monkeypatch.setattr(tasks, "load_state", lambda log_id: state)
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: ExpiredResultProvider(),
    )
    monkeypatch.setattr(
        tasks,
        "save_state",
        lambda *args, **kwargs: saved.append(kwargs),
    )
    monkeypatch.setattr(tasks, "report_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda **kwargs: scheduled.append(kwargs),
    )
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an expired URL before the deadline must retry")
        ),
    )
    monkeypatch.setattr(tasks.time, "time", lambda: 100.0)

    tasks.poll_real_to_render(
        "log-result-refresh",
        True,
        "provider-result-refresh",
        submitted_at=0.0,
    )

    succeeded_update = next(
        update
        for update in saved
        if update.get("provider_status") == "succeeded"
        and "result_deadline" in update
    )
    assert succeeded_update["result_deadline"] == 200.0
    assert succeeded_update["result_url_obtained_at"] == 50.0
    assert any(
        update.get("provider_status") == "refresh_result_url"
        for update in saved
    )
    assert scheduled[-1]["delay_seconds"] == 2


def test_handoff_failure_keeps_pending_skin_and_retries(monkeypatch):
    scheduled = []
    reports = []
    state = {
        "intermediate_key": "real_to_render_intermediate/log-handoff.png",
        "recovery_phase": "handoff_pending",
    }
    monkeypatch.setattr(tasks, "get_settings", poll_settings)
    monkeypatch.setattr(tasks, "load_state", lambda log_id: state)
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "report_status",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "enqueue_render_to_uv_once",
        lambda **kwargs: (_ for _ in ()).throw(
            ConnectionError("Redis unavailable")
        ),
    )
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda **kwargs: scheduled.append(kwargs),
    )

    tasks.poll_real_to_render(
        "log-handoff",
        False,
        "provider-handoff",
        submitted_at=0.0,
    )

    assert reports[0][0][1] == "pending_skin"
    assert reports[-1][0][1] == "pending_skin"
    assert reports[-1][1]["error_origin"] == "enqueue_render_to_uv"
    assert scheduled[-1]["delay_seconds"] == 2


def test_poll_job_uses_configured_hard_timeout(monkeypatch):
    captured = {}

    class FakeQueue:
        def __init__(self, name, connection=None):
            captured["name"] = name

        def enqueue_in(self, delay, task, **kwargs):
            captured["delay"] = delay
            captured["task"] = task
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(
            poll_interval=10,
            job_timeout=320,
        ),
    )
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "redis_connection", lambda: object())
    monkeypatch.setattr(
        tasks.Job,
        "fetch",
        lambda *args, **kwargs: (_ for _ in ()).throw(NoSuchJobError),
    )

    tasks.schedule_poll(
        log_id="log-timeout",
        is_public=True,
        provider_task_id="provider-timeout",
        submitted_at=0.0,
        queue_prefix="",
        poll_number=1,
    )

    assert captured["name"] == "queue_real_to_render"
    assert captured["task"] == "tasks.poll_real_to_render"
    assert captured["job_timeout"] == 320
    assert captured["retry"].max == 5


def test_render_to_uv_handoff_has_bounded_rq_retry(monkeypatch):
    captured = {}

    class FakeQueue:
        def __init__(self, name, connection=None):
            captured["name"] = name

        def enqueue(self, task, **kwargs):
            captured["task"] = task
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(
            render_to_uv_task="worker_tasks.task_render_to_uv",
            pipeline_version="pipeline-v1",
            render_to_uv_job_timeout=120,
            render_to_uv_retry_max=5,
            render_to_uv_retry_intervals=(5, 15, 30, 60, 120),
        ),
    )
    monkeypatch.setattr(tasks, "Queue", FakeQueue)
    monkeypatch.setattr(tasks, "redis_connection", lambda: object())
    monkeypatch.setattr(
        tasks.Job,
        "fetch",
        lambda *args, **kwargs: (_ for _ in ()).throw(NoSuchJobError),
    )

    tasks.enqueue_render_to_uv_once(
        log_id="log-gpu-retry",
        is_public=True,
        intermediate_key="real_to_render_intermediate/log-gpu-retry.png",
        queue_prefix="",
    )

    assert captured["name"] == "queue_render_to_uv"
    assert captured["retry"].max == 5
    assert captured["retry"].intervals == [5, 15, 30, 60, 120]


def test_interrupted_submit_resumes_existing_provider_task(monkeypatch):
    scheduled = []
    reports = []
    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        tasks,
        "load_state",
        lambda log_id: {
            "provider_task_id": "provider-existing",
            "submitted_at": "100.0",
            "poll_number": "3",
        },
    )
    monkeypatch.setattr(
        tasks,
        "schedule_poll",
        lambda **kwargs: scheduled.append(kwargs),
    )
    monkeypatch.setattr(
        tasks,
        "report_status",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "object_storage",
        lambda: (_ for _ in ()).throw(
            AssertionError("resumed submit must not download from S3")
        ),
    )
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("resumed submit must not call provider submit")
        ),
    )

    tasks.submit_real_to_render(
        "log-resume",
        True,
        "uploads/log-resume.png",
        "image/png",
        "high_",
    )

    assert scheduled == [
        {
            "log_id": "log-resume",
            "is_public": True,
            "provider_task_id": "provider-existing",
            "submitted_at": 100.0,
            "queue_prefix": "high_",
            "poll_number": 4,
        }
    ]
    assert reports[-1][1]["provider_task_id"] == "provider-existing"


def test_submit_read_timeout_is_marked_unknown_without_resubmission(
    monkeypatch,
):
    saved = []
    reports = []
    failures = []

    class FakeStorage:
        def download(self, source, is_public):
            return b"source-image"

    class TimeoutProvider:
        def submit(self, **kwargs):
            raise requests.ReadTimeout("acceptance is uncertain")

    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(
            template_paths=(),
            prompt_template="{template_refs}",
            aspect_ratio="1:1",
            image_size="1K",
        ),
    )
    monkeypatch.setattr(tasks, "load_state", lambda log_id: {})
    monkeypatch.setattr(tasks, "object_storage", lambda: FakeStorage())
    monkeypatch.setattr(tasks, "image_data_url", lambda *args: "source")
    monkeypatch.setattr(tasks, "template_data_urls", lambda paths: [])
    monkeypatch.setattr(tasks, "provider_client", lambda: TimeoutProvider())
    monkeypatch.setattr(
        tasks,
        "save_state",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "report_status",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "fail_stage",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )

    tasks.submit_real_to_render(
        "log-submit-timeout",
        True,
        "uploads/log-submit-timeout.png",
    )

    assert failures == []
    assert any(
        kwargs.get("submission_state") == "in_flight"
        for _, kwargs in saved
    )
    assert saved[-1][1]["submission_state"] == "unknown"
    assert reports[-1][1]["provider_submission_state"] == "unknown"
    assert reports[-1][1]["requires_reconciliation"] is True


@pytest.mark.parametrize("submission_state", ["in_flight", "unknown"])
def test_uncertain_submit_is_never_resubmitted(
    monkeypatch,
    submission_state,
):
    reports = []
    monkeypatch.setattr(tasks, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        tasks,
        "load_state",
        lambda log_id: {"submission_state": submission_state},
    )
    monkeypatch.setattr(tasks, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tasks,
        "report_status",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("uncertain submission must not be repeated")
        ),
    )

    tasks.submit_real_to_render(
        "log-unknown",
        True,
        "uploads/log-unknown.png",
    )

    assert reports[-1][1]["provider_submission_state"] == "unknown"
    assert reports[-1][1]["requires_reconciliation"] is True
