from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from rq.job import Job, JobStatus
from rq.utils import utcformat

import recovery


class FakeConnection:
    def __init__(self, worker_data=None):
        self.worker_data = worker_data or {}
        self.pipeline_value = FakePipeline()

    def hgetall(self, key):
        return self.worker_data

    def pipeline(self):
        return self.pipeline_value


class FakePipeline:
    def __init__(self):
        self.executed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self):
        self.executed = True


def test_worker_owns_job_only_with_current_fresh_heartbeat():
    now = datetime.now(timezone.utc)
    connection = FakeConnection(
        {
            b"current_job": b"job-1",
            b"last_heartbeat": utcformat(now - timedelta(seconds=10)).encode(),
        }
    )
    job = SimpleNamespace(id="job-1", worker_name="worker-1")

    assert recovery.worker_owns_job(
        connection,
        job,
        orphan_grace_seconds=75,
        now=now,
    )

    connection.worker_data[b"last_heartbeat"] = utcformat(
        now - timedelta(seconds=76)
    ).encode()
    assert not recovery.worker_owns_job(
        connection,
        job,
        orphan_grace_seconds=75,
        now=now,
    )


def test_worker_death_marker_makes_started_job_recoverable():
    now = datetime.now(timezone.utc)
    connection = FakeConnection(
        {
            b"current_job": b"job-1",
            b"last_heartbeat": utcformat(now).encode(),
            b"death": utcformat(now).encode(),
        }
    )
    job = SimpleNamespace(id="job-1", worker_name="worker-1")

    assert not recovery.worker_owns_job(
        connection,
        job,
        orphan_grace_seconds=75,
        now=now,
    )


def test_requeue_interrupted_job_removes_execution_and_resets_job(monkeypatch):
    deleted = []
    enqueued = []

    class FakeExecution:
        def __init__(self, id, job_id, connection):
            self.id = id
            self.job_id = job_id

        def delete(self, job, pipeline):
            deleted.append((self.id, self.job_id, job.id))

    class FakeQueue:
        def enqueue_job(self, job, pipeline, at_front):
            enqueued.append((job.id, at_front))

    monkeypatch.setattr(recovery, "Execution", FakeExecution)
    connection = FakeConnection()
    now = datetime.now(timezone.utc)
    job = Job(id="job-1", connection=connection)
    job._status = JobStatus.STARTED
    job.worker_name = "dead-worker"
    job.started_at = now
    job.ended_at = now
    job.last_heartbeat = now
    job._exc_info = "old failure"
    job._cached_result = object()

    recovery.requeue_interrupted_job(
        connection,
        FakeQueue(),
        job,
        "execution-1",
    )

    assert deleted == [("execution-1", "job-1", "job-1")]
    assert enqueued == [("job-1", True)]
    assert job._status == JobStatus.QUEUED
    assert job.worker_name is None
    assert job.started_at is None
    assert job.ended_at is None
    assert job.last_heartbeat is None
    assert job._exc_info == ""
    assert job._cached_result is None
    assert connection.pipeline_value.executed is True
