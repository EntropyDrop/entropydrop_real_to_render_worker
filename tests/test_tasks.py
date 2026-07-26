from types import SimpleNamespace

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
