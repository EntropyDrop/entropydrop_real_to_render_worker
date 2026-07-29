import json
from types import SimpleNamespace

import pytest
from rq.exceptions import NoSuchJobError

import tasks
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
        lambda: SimpleNamespace(max_wait=320),
    )
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


def test_stage_timeout_does_not_query_or_reschedule(monkeypatch):
    failures = []
    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(max_wait=320),
    )
    monkeypatch.setattr(
        tasks,
        "provider_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("provider must not be queried after timeout")
        ),
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
    monkeypatch.setattr(tasks.time, "time", lambda: 321.0)

    tasks.poll_real_to_render(
        "log-2",
        False,
        "provider-2",
        submitted_at=0.0,
    )

    assert len(failures) == 1
    assert failures[0][1]["failure_reason"] == "timeout"


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
