import logging
from datetime import datetime, timezone

from redis import Redis
from rq import Queue, Worker
from rq.executions import Execution
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.utils import as_text, utcparse


RECOVERABLE_TASKS = {
    "tasks.submit_real_to_render",
    "tasks.poll_real_to_render",
}
logger = logging.getLogger("rq.worker")


def worker_owns_job(
    connection: Redis,
    job: Job,
    orphan_grace_seconds: int,
    now: datetime | None = None,
) -> bool:
    """Return whether the job's recorded worker is still actively heartbeating."""
    worker_name = getattr(job, "worker_name", None)
    if not worker_name:
        return False

    worker_key = f"{Worker.redis_worker_namespace_prefix}{worker_name}"
    raw = connection.hgetall(worker_key)
    if not raw:
        return False

    data = {
        as_text(key): as_text(value)
        for key, value in raw.items()
    }
    if data.get("death"):
        return False
    if data.get("current_job") != job.id:
        return False

    heartbeat = data.get("last_heartbeat")
    if not heartbeat:
        return False
    heartbeat_at = utcparse(heartbeat)
    current_time = now or datetime.now(timezone.utc)
    return (
        current_time - heartbeat_at
    ).total_seconds() <= orphan_grace_seconds


def requeue_interrupted_job(
    connection: Redis,
    queue: Queue,
    job: Job,
    execution_id: str,
) -> None:
    """Remove a dead execution and put its original job back at queue front."""
    execution = Execution(
        id=execution_id,
        job_id=job.id,
        connection=connection,
    )
    job._status = JobStatus.QUEUED
    job.worker_name = None
    job.started_at = None
    job.ended_at = None
    job.last_heartbeat = None
    job._exc_info = ""
    job._cached_result = None

    with connection.pipeline() as pipeline:
        # RQ initializes the supplied pipeline with MULTI from
        # Queue.enqueue_job(). Redis-py rejects MULTI if commands were queued
        # first, so the old execution must be deleted only after RQ has put
        # the pipeline into transaction mode. Both operations are still
        # committed atomically by the single execute() below.
        queue.enqueue_job(job, pipeline=pipeline, at_front=True)
        execution.delete(job, pipeline)
        pipeline.execute()


def recover_interrupted_jobs(
    connection: Redis,
    queues: list[Queue],
    orphan_grace_seconds: int,
) -> int:
    """Requeue real-to-render jobs whose owning RQ worker has disappeared."""
    recovered = 0
    for queue in queues:
        registry = queue.started_job_registry
        entries = registry.get_job_and_execution_ids(cleanup=False)
        for job_id, execution_id in entries:
            try:
                job = Job.fetch(job_id, connection=connection)
            except NoSuchJobError:
                continue
            if job.func_name not in RECOVERABLE_TASKS:
                continue

            lock = connection.lock(
                f"real_to_render:recovery:{job_id}",
                timeout=30,
                blocking_timeout=0,
            )
            if not lock.acquire(blocking=False):
                continue
            try:
                composite_key = f"{job_id}:{execution_id}"
                if connection.zscore(registry.key, composite_key) is None:
                    continue
                job.refresh()
                if job.get_status(refresh=False) != JobStatus.STARTED:
                    continue
                if worker_owns_job(
                    connection,
                    job,
                    orphan_grace_seconds,
                ):
                    continue
                requeue_interrupted_job(
                    connection,
                    queue,
                    job,
                    execution_id,
                )
                recovered += 1
                logger.warning(
                    "Recovered interrupted real-to-render job %s from %s",
                    job_id,
                    queue.name,
                )
            finally:
                lock.release()
    return recovered


class RecoveringWorker(Worker):
    """RQ worker that periodically recovers jobs orphaned by forced restarts."""

    def __init__(self, *args, orphan_grace_seconds: int, **kwargs):
        self.orphan_grace_seconds = orphan_grace_seconds
        super().__init__(*args, **kwargs)

    def recover_interrupted_jobs(self) -> int:
        return recover_interrupted_jobs(
            self.connection,
            self.queues,
            self.orphan_grace_seconds,
        )

    def run_maintenance_tasks(self):
        self.recover_interrupted_jobs()
        super().run_maintenance_tasks()
