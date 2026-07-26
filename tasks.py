import json
import time
import traceback
from datetime import timedelta
from functools import lru_cache

from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Callback, Job

from config import get_settings
from images import (
    image_data_url,
    normalize_combined_render,
    template_data_urls,
)
from provider import RealToRenderProvider
from storage import ObjectStorage


STAGE = "real_to_render"


@lru_cache(maxsize=1)
def redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(
        settings.redis_url,
        health_check_interval=20,
        socket_connect_timeout=5,
        socket_timeout=20,
    )


@lru_cache(maxsize=1)
def provider_client() -> RealToRenderProvider:
    settings = get_settings()
    return RealToRenderProvider(
        base_url=settings.provider_base_url,
        api_key=settings.provider_api_key,
        model=settings.provider_model,
        connect_timeout=settings.provider_connect_timeout,
        read_timeout=settings.provider_read_timeout,
        download_timeout=settings.image_download_timeout,
        proxy_url=settings.outbound_http_proxy,
    )


@lru_cache(maxsize=1)
def object_storage() -> ObjectStorage:
    settings = get_settings()
    return ObjectStorage(
        access_key_id=settings.aws_access_key_id,
        secret_access_key=settings.aws_secret_access_key,
        region=settings.aws_region,
        public_bucket=settings.public_bucket,
        private_bucket=settings.private_bucket,
        proxy_url=settings.outbound_http_proxy,
    )


def state_key(log_id: str) -> str:
    return f"real_to_render:state:{log_id}"


def report_status(log_id: str, status: str, **fields) -> None:
    settings = get_settings()
    payload = {
        "log_id": log_id,
        "status": status,
        "stage": STAGE,
        "pipeline_version": settings.pipeline_version,
    }
    payload.update(
        {key: value for key, value in fields.items() if value is not None}
    )
    redis_connection().lpush(
        settings.result_queue_key,
        json.dumps(payload, ensure_ascii=False),
    )


def save_state(log_id: str, **fields) -> None:
    settings = get_settings()
    key = state_key(log_id)
    mapping = {
        name: str(value)
        for name, value in fields.items()
        if value is not None
    }
    if mapping:
        redis_connection().hset(key, mapping=mapping)
    redis_connection().expire(key, settings.state_ttl)


def queue_name(prefix: str, base: str) -> str:
    if prefix not in ("", "high_"):
        raise ValueError(f"Unsupported queue prefix: {prefix!r}")
    return f"{prefix}{base}"


def schedule_poll(
    log_id: str,
    is_public: bool,
    provider_task_id: str,
    submitted_at: float,
    queue_prefix: str,
    poll_number: int,
) -> Job:
    settings = get_settings()
    queue = Queue(
        queue_name(queue_prefix, "queue_real_to_render"),
        connection=redis_connection(),
    )
    return queue.enqueue_in(
        timedelta(seconds=settings.poll_interval),
        "tasks.poll_real_to_render",
        args=(
            log_id,
            is_public,
            provider_task_id,
            submitted_at,
            queue_prefix,
            poll_number,
        ),
        job_id=f"real_to_render_poll_{log_id}_{poll_number}",
        job_timeout=settings.job_timeout,
        retry=None,
        on_failure=Callback(
            "tasks.real_to_render_job_failure",
            timeout=20,
        ),
        on_stopped=Callback(
            "tasks.real_to_render_job_stopped",
            timeout=20,
        ),
        result_ttl=60,
        failure_ttl=86400,
    )


def fail_stage(
    log_id: str,
    error: Exception | str,
    provider_task_id: str | None = None,
    failure_reason: str | None = None,
) -> None:
    message = str(error)
    save_state(
        log_id,
        terminal_status="failed",
        provider_task_id=provider_task_id,
        failure_reason=failure_reason,
        error=message,
        finished_at=time.time(),
    )
    report_status(
        log_id,
        "failed",
        provider_task_id=provider_task_id,
        failure_reason=failure_reason,
        error_msg=message,
    )


def real_to_render_job_failure(
    job: Job,
    connection: Redis,
    exception_type,
    exception_value,
    traceback_value,
) -> None:
    """RQ callback for hard timeout/stopped jobs that bypass task try/except."""
    del connection, exception_type, traceback_value
    log_id = job.args[0] if job.args else None
    if not log_id:
        return
    raw_task_id = redis_connection().hget(
        state_key(log_id),
        "provider_task_id",
    )
    provider_task_id = (
        raw_task_id.decode("utf-8")
        if isinstance(raw_task_id, bytes)
        else raw_task_id
    )
    fail_stage(
        log_id,
        exception_value or "Stage-1 RQ job stopped",
        provider_task_id=provider_task_id,
        failure_reason="worker_stopped",
    )


def real_to_render_job_stopped(job: Job, connection: Redis) -> None:
    real_to_render_job_failure(
        job,
        connection,
        None,
        "Stage-1 RQ job stopped",
        None,
    )


def submit_real_to_render(
    log_id: str,
    is_public: bool,
    source: str,
    content_type: str = "image/png",
    queue_prefix: str = "",
) -> None:
    """Submit exactly once. Any exception is terminal and is not retried."""
    provider_task_id = None
    try:
        settings = get_settings()
        report_status(log_id, "processing")

        source_content = object_storage().download(source, is_public)
        references = [
            image_data_url(source_content, content_type),
            *template_data_urls(settings.template_paths),
        ]
        template_refs = "".join(
            f"[图{index}]"
            for index in range(2, len(settings.template_paths) + 2)
        )
        prompt = settings.prompt_template.format(
            template_refs=template_refs
        )

        provider_task_id = provider_client().submit(
            images=references,
            prompt=prompt,
            aspect_ratio=settings.aspect_ratio,
            image_size=settings.image_size,
        )
        submitted_at = time.time()
        save_state(
            log_id,
            provider_task_id=provider_task_id,
            submitted_at=submitted_at,
            provider_status="running",
            progress=0,
        )
        schedule_poll(
            log_id=log_id,
            is_public=is_public,
            provider_task_id=provider_task_id,
            submitted_at=submitted_at,
            queue_prefix=queue_prefix,
            poll_number=1,
        )
        report_status(
            log_id,
            "processing",
            provider_task_id=provider_task_id,
        )
    except Exception as exc:
        traceback.print_exc()
        fail_stage(log_id, exc, provider_task_id=provider_task_id)


def enqueue_render_to_uv_once(
    log_id: str,
    is_public: bool,
    intermediate_key: str,
    queue_prefix: str,
) -> Job:
    settings = get_settings()
    connection = redis_connection()
    job_id = f"generation_{log_id}_render_to_uv"
    try:
        return Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        pass

    queue = Queue(
        queue_name(queue_prefix, "queue_render_to_uv"),
        connection=connection,
    )
    return queue.enqueue(
        settings.render_to_uv_task,
        args=(
            log_id,
            is_public,
            intermediate_key,
            "image/png",
            settings.pipeline_version,
        ),
        job_id=job_id,
        job_timeout=settings.render_to_uv_job_timeout,
        retry=None,
        result_ttl=60,
        failure_ttl=86400,
    )


def poll_real_to_render(
    log_id: str,
    is_public: bool,
    provider_task_id: str,
    submitted_at: float,
    queue_prefix: str = "",
    poll_number: int = 1,
) -> None:
    """Perform one short poll and schedule another only while still running."""
    try:
        settings = get_settings()
        elapsed = time.time() - float(submitted_at)
        if elapsed >= settings.max_wait:
            fail_stage(
                log_id,
                f"Stage 1 timed out after {settings.max_wait} seconds",
                provider_task_id=provider_task_id,
                failure_reason="timeout",
            )
            return

        status = provider_client().get_status(provider_task_id)
        save_state(
            log_id,
            provider_task_id=provider_task_id,
            provider_status=status.status,
            progress=status.progress,
            last_polled_at=time.time(),
        )

        if status.status == "running":
            schedule_poll(
                log_id=log_id,
                is_public=is_public,
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=poll_number + 1,
            )
            return

        if status.status == "failed":
            detail = status.error or status.failure_reason or "Provider failed"
            fail_stage(
                log_id,
                detail,
                provider_task_id=provider_task_id,
                failure_reason=status.failure_reason or "error",
            )
            return

        raw_result = provider_client().download_result(status.result_url)
        normalized_png, dimensions = normalize_combined_render(raw_result)
        intermediate_key = (
            f"real_to_render_intermediate/{log_id}.png"
        )
        object_storage().upload_png(
            intermediate_key,
            normalized_png,
            is_public,
        )
        enqueue_render_to_uv_once(
            log_id=log_id,
            is_public=is_public,
            intermediate_key=intermediate_key,
            queue_prefix=queue_prefix,
        )
        save_state(
            log_id,
            terminal_status="handed_off",
            provider_status="succeeded",
            progress=100,
            intermediate_key=intermediate_key,
            dimensions=f"{dimensions[0]}x{dimensions[1]}",
            finished_at=time.time(),
        )
        report_status(
            log_id,
            "pending_skin",
            provider_task_id=provider_task_id,
            edited_result=intermediate_key,
        )
    except Exception as exc:
        traceback.print_exc()
        fail_stage(
            log_id,
            exc,
            provider_task_id=provider_task_id,
            failure_reason="error",
        )
