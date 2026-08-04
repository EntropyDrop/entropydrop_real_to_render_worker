import json
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import timedelta
from functools import lru_cache
from typing import Iterator

import requests
from redis import Redis
from rq import Queue, Retry
from rq.exceptions import NoSuchJobError
from rq.job import Callback, Job

from config import get_settings
from images import (
    InvalidShapeError,
    image_data_url,
    normalize_combined_render,
    template_data_urls,
    validate_combined_render_shape,
)
from provider import (
    RealToRenderProvider,
    TERMINAL_FAILURE_STATUSES,
    VIOLATION_ERROR_MESSAGE,
)
from storage import ObjectStorage


STAGE = "real_to_render"
BYTES_PER_MB = 1024 * 1024
logger = logging.getLogger("rq.job")


def size_mb(byte_count: int) -> float:
    return round(byte_count / BYTES_PER_MB, 3)


@contextmanager
def timed_step(
    step: str,
    log_id: str,
    **initial_fields,
) -> Iterator[dict]:
    """Log one JSON timing event on both success and failure."""
    fields = dict(initial_fields)
    started_at = time.perf_counter()
    try:
        yield fields
    except Exception as exc:
        fields["outcome"] = "failed"
        fields["error_type"] = type(exc).__name__
        raise
    else:
        fields["outcome"] = "succeeded"
    finally:
        payload = {
            "event": "real_to_render_step",
            "stage": STAGE,
            "step": step,
            "log_id": log_id,
            "duration_seconds": round(
                time.perf_counter() - started_at,
                3,
            ),
            **fields,
        }
        logger.info(json.dumps(payload, ensure_ascii=False))


@lru_cache(maxsize=1)
def redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(
        settings.redis_url,
        health_check_interval=20,
        socket_connect_timeout=5,
        socket_timeout=20,
        retry_on_timeout=True,
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
        api_use_proxy=settings.provider_use_proxy,
        download_direct_fallback=(
            settings.image_download_direct_fallback
        ),
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


def load_state(log_id: str) -> dict[str, str]:
    raw = redis_connection().hgetall(state_key(log_id))
    return {
        (
            key.decode("utf-8")
            if isinstance(key, bytes)
            else str(key)
        ): (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
        )
        for key, value in raw.items()
    }


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
    delay_seconds: int | None = None,
) -> Job:
    settings = get_settings()
    job_id = f"real_to_render_poll_{log_id}_{poll_number}"
    try:
        return Job.fetch(job_id, connection=redis_connection())
    except NoSuchJobError:
        pass

    queue = Queue(
        queue_name(queue_prefix, "queue_real_to_render"),
        connection=redis_connection(),
    )
    retry_intervals = list(
        getattr(
            settings,
            "recovery_retry_intervals",
            (2, 5, 15, 30, 60),
        )
    )
    return queue.enqueue_in(
        timedelta(
            seconds=(
                delay_seconds
                if delay_seconds is not None
                else settings.poll_interval
            )
        ),
        "tasks.poll_real_to_render",
        args=(
            log_id,
            is_public,
            provider_task_id,
            submitted_at,
            queue_prefix,
            poll_number,
        ),
        job_id=job_id,
        job_timeout=settings.job_timeout,
        retry=Retry(
            max=getattr(settings, "poll_job_retry_max", 5),
            interval=retry_intervals,
        ),
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


def _state_int(state: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(state.get(key) or default)
    except (TypeError, ValueError):
        return default


def schedule_stage_recovery(
    *,
    log_id: str,
    is_public: bool,
    provider_task_id: str,
    submitted_at: float,
    queue_prefix: str,
    poll_number: int,
    phase: str,
    error: Exception | str,
    state: dict[str, str] | None = None,
    backend_status: str = "processing",
    edited_result: str | None = None,
) -> Job:
    """Persist a transient failure and schedule a safe, idempotent retry."""
    settings = get_settings()
    state = state or load_state(log_id)
    attempt_key = f"{phase}_attempts"
    attempt = _state_int(state, attempt_key) + 1
    intervals = getattr(
        settings,
        "recovery_retry_intervals",
        (2, 5, 15, 30, 60),
    )
    delay = intervals[min(attempt - 1, len(intervals) - 1)]
    message = str(error)
    save_state(
        log_id,
        recovery_phase=phase,
        **{
            attempt_key: attempt,
            "last_error": message,
            "last_error_type": type(error).__name__,
            "next_retry_at": time.time() + delay,
        },
    )
    report_status(
        log_id,
        backend_status,
        provider_task_id=provider_task_id,
        edited_result=edited_result,
        retrying=True,
        error_origin=phase,
        retry_attempt=attempt,
        next_retry_seconds=delay,
        error_msg=message,
    )
    return schedule_poll(
        log_id=log_id,
        is_public=is_public,
        provider_task_id=provider_task_id,
        submitted_at=submitted_at,
        queue_prefix=queue_prefix,
        poll_number=poll_number + 1,
        delay_seconds=delay,
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
    state = load_state(log_id)
    if job.func_name == "tasks.submit_real_to_render":
        queue_prefix = job.args[4] if len(job.args) > 4 else ""
        if provider_task_id:
            submitted_at = float(
                state.get("submitted_at") or time.time()
            )
            schedule_poll(
                log_id=log_id,
                is_public=bool(job.args[1]),
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=max(
                    _state_int(state, "poll_number") + 1,
                    int(time.time()),
                ),
            )
            report_status(
                log_id,
                "processing",
                provider_task_id=provider_task_id,
                provider_submission_state="accepted",
                retrying=True,
                error_origin="submit_job_interrupted",
            )
            return
        if state.get("submission_state") in {"in_flight", "unknown"}:
            save_state(
                log_id,
                submission_state="unknown",
                recovery_phase="submit_unknown",
                last_error=str(exception_value or "Submission interrupted"),
            )
            report_status(
                log_id,
                "processing",
                provider_submission_state="unknown",
                retrying=False,
                requires_reconciliation=True,
                error_origin="provider_submit_api",
                error_msg=str(exception_value or "Submission interrupted"),
            )
            return
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
    """Submit once, or resume polling when a prior submission was persisted."""
    provider_task_id = None
    try:
        settings = get_settings()

        existing_state = load_state(log_id)
        terminal_status = existing_state.get("terminal_status")
        if terminal_status:
            logger.info(
                "Skipping resumed submission for %s: terminal_status=%s",
                log_id,
                terminal_status,
            )
            return

        provider_task_id = existing_state.get("provider_task_id")
        if provider_task_id:
            submitted_at = float(
                existing_state.get("submitted_at") or time.time()
            )
            next_poll_number = (
                int(existing_state.get("poll_number") or 0) + 1
            )
            report_status(
                log_id,
                "processing",
                provider_task_id=provider_task_id,
                provider_submission_state="accepted",
            )
            schedule_poll(
                log_id=log_id,
                is_public=is_public,
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=next_poll_number,
            )
            logger.warning(
                "Resumed provider polling for interrupted submission %s",
                log_id,
            )
            return

        if existing_state.get("submission_state") in {
            "in_flight",
            "unknown",
        }:
            save_state(
                log_id,
                submission_state="unknown",
                recovery_phase="submit_unknown",
            )
            report_status(
                log_id,
                "processing",
                provider_submission_state="unknown",
                retrying=False,
                requires_reconciliation=True,
                error_origin="provider_submit_api",
                error_msg=(
                    existing_state.get("last_error")
                    or "Provider submission acceptance is uncertain"
                ),
            )
            logger.error(
                "Refusing to resubmit uncertain provider request for %s",
                log_id,
            )
            return

        report_status(log_id, "processing")
        with timed_step(
            "s3_download",
            log_id,
            storage_scope="public" if is_public else "private",
        ) as timing:
            source_content = object_storage().download(source, is_public)
            timing["size_mb"] = size_mb(len(source_content))

        with timed_step(
            "prepare_api_request",
            log_id,
            template_count=len(settings.template_paths),
        ) as timing:
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
            timing["image_count"] = len(references)

        save_state(
            log_id,
            submission_state="in_flight",
            submission_started_at=time.time(),
        )
        report_status(
            log_id,
            "processing",
            provider_submission_state="in_flight",
        )
        try:
            with timed_step("provider_submit_api", log_id) as timing:
                provider_task_id = provider_client().submit(
                    images=references,
                    prompt=prompt,
                    aspect_ratio=settings.aspect_ratio,
                    image_size=settings.image_size,
                )
                timing["provider_task_id"] = provider_task_id
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", 0)
            if 400 <= status_code < 500:
                fail_stage(
                    log_id,
                    exc,
                    failure_reason="provider_rejected",
                )
                return
            save_state(
                log_id,
                submission_state="unknown",
                recovery_phase="submit_unknown",
                last_error=str(exc),
                last_error_type=type(exc).__name__,
            )
            report_status(
                log_id,
                "processing",
                provider_submission_state="unknown",
                retrying=False,
                requires_reconciliation=True,
                error_origin="provider_submit_api",
                error_msg=str(exc),
            )
            return
        except Exception as exc:
            save_state(
                log_id,
                submission_state="unknown",
                recovery_phase="submit_unknown",
                last_error=str(exc),
                last_error_type=type(exc).__name__,
            )
            report_status(
                log_id,
                "processing",
                provider_submission_state="unknown",
                retrying=False,
                requires_reconciliation=True,
                error_origin="provider_submit_api",
                error_msg=str(exc),
            )
            return
        submitted_at = time.time()
        save_state(
            log_id,
            submission_state="accepted",
            provider_task_id=provider_task_id,
            submitted_at=submitted_at,
            provider_status="running",
            progress=0,
            poll_number=0,
        )
        report_status(
            log_id,
            "processing",
            provider_task_id=provider_task_id,
            provider_submission_state="accepted",
        )
        schedule_poll(
            log_id=log_id,
            is_public=is_public,
            provider_task_id=provider_task_id,
            submitted_at=submitted_at,
            queue_prefix=queue_prefix,
            poll_number=1,
        )
    except Exception as exc:
        traceback.print_exc()
        if provider_task_id:
            raise
        fail_stage(log_id, exc, provider_task_id=provider_task_id)


def resume_real_to_render(
    log_id: str,
    is_public: bool,
    source: str,
    content_type: str,
    queue_prefix: str,
    provider_task_id: str,
) -> None:
    """Recover an accepted provider task without ever resubmitting it."""
    del source, content_type
    state = load_state(log_id)
    if state.get("terminal_status"):
        return
    submitted_at = float(state.get("submitted_at") or time.time())
    poll_number = max(
        _state_int(state, "poll_number") + 1,
        int(time.time()),
    )
    save_state(
        log_id,
        submission_state="accepted",
        provider_task_id=provider_task_id,
        submitted_at=submitted_at,
    )
    report_status(
        log_id,
        "processing",
        provider_task_id=provider_task_id,
        provider_submission_state="accepted",
        retrying=True,
        error_origin="backend_recovery",
    )
    schedule_poll(
        log_id=log_id,
        is_public=is_public,
        provider_task_id=provider_task_id,
        submitted_at=submitted_at,
        queue_prefix=queue_prefix,
        poll_number=poll_number,
    )


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
        retry=Retry(
            max=settings.render_to_uv_retry_max,
            interval=list(settings.render_to_uv_retry_intervals),
        ),
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
    """Advance one recoverable stage without resubmitting the provider job."""
    settings = get_settings()
    state = load_state(log_id)
    if state.get("terminal_status"):
        return

    elapsed = time.time() - float(submitted_at)
    intermediate_key = state.get("intermediate_key")

    if not intermediate_key:
        result_url = (
            state.get("result_url")
            if state.get("provider_status") == "succeeded"
            else None
        )
        if not result_url:
            try:
                with timed_step(
                    "provider_status_api",
                    log_id,
                    provider_task_id=provider_task_id,
                    poll_number=poll_number,
                ) as timing:
                    status = provider_client().get_status(provider_task_id)
                    timing["provider_status"] = status.status
                    timing["provider_progress"] = status.progress
            except Exception as exc:
                traceback.print_exc()
                schedule_stage_recovery(
                    log_id=log_id,
                    is_public=is_public,
                    provider_task_id=provider_task_id,
                    submitted_at=submitted_at,
                    queue_prefix=queue_prefix,
                    poll_number=poll_number,
                    phase="provider_status_api",
                    error=exc,
                    state=state,
                )
                return

            save_state(
                log_id,
                provider_task_id=provider_task_id,
                provider_status=status.status,
                progress=status.progress,
                last_polled_at=time.time(),
                poll_number=poll_number,
            )

            if status.status == "running":
                if elapsed >= settings.hard_wait:
                    fail_stage(
                        log_id,
                        (
                            "Provider still reported running after "
                            f"{settings.hard_wait} seconds"
                        ),
                        provider_task_id=provider_task_id,
                        failure_reason="provider_hard_timeout",
                    )
                    return
                delayed = elapsed >= settings.max_wait
                if state.get("recovery_phase") == "provider_status_api":
                    save_state(
                        log_id,
                        recovery_phase="provider_running",
                        last_error="",
                    )
                    report_status(
                        log_id,
                        "processing",
                        provider_task_id=provider_task_id,
                        provider_submission_state="accepted",
                    )
                if delayed and not state.get("soft_timeout_at"):
                    save_state(log_id, soft_timeout_at=time.time())
                    report_status(
                        log_id,
                        "processing",
                        provider_task_id=provider_task_id,
                        provider_submission_state="accepted",
                        provider_delayed=True,
                    )
                schedule_poll(
                    log_id=log_id,
                    is_public=is_public,
                    provider_task_id=provider_task_id,
                    submitted_at=submitted_at,
                    queue_prefix=queue_prefix,
                    poll_number=poll_number + 1,
                    delay_seconds=(
                        settings.delayed_poll_interval
                        if delayed
                        else settings.poll_interval
                    ),
                )
                return

            if status.status in TERMINAL_FAILURE_STATUSES:
                if status.status == "violation":
                    detail = VIOLATION_ERROR_MESSAGE
                    failure_reason = "violation"
                else:
                    detail = (
                        status.error
                        or status.failure_reason
                        or "Provider failed"
                    )
                    failure_reason = status.failure_reason or "failed"
                fail_stage(
                    log_id,
                    detail,
                    provider_task_id=provider_task_id,
                    failure_reason=failure_reason,
                )
                return

            result_url = status.result_url
            result_deadline = float(
                state.get("result_deadline")
                or (time.time() + settings.result_recovery_window)
            )
            save_state(
                log_id,
                provider_status="succeeded",
                progress=100,
                result_url=result_url,
                result_url_obtained_at=(
                    state.get("result_url_obtained_at") or time.time()
                ),
                result_deadline=result_deadline,
                recovery_phase="result_fetch_pending",
            )
            state = load_state(log_id)

        result_deadline = float(
            state.get("result_deadline")
            or (time.time() + settings.result_recovery_window)
        )
        try:
            with timed_step(
                "provider_result_download",
                log_id,
                provider_task_id=provider_task_id,
            ) as timing:
                raw_result = provider_client().download_result(result_url)
                timing["size_mb"] = size_mb(len(raw_result))
        except Exception as exc:
            traceback.print_exc()
            if time.time() >= result_deadline:
                fail_stage(
                    log_id,
                    (
                        "Provider result could not be recovered before "
                        "its recovery deadline"
                    ),
                    provider_task_id=provider_task_id,
                    failure_reason="result_recovery_timeout",
                )
                return
            if isinstance(exc, requests.HTTPError):
                status_code = getattr(exc.response, "status_code", 0)
                if status_code in {403, 404}:
                    save_state(
                        log_id,
                        provider_status="refresh_result_url",
                        result_url="",
                    )
            schedule_stage_recovery(
                log_id=log_id,
                is_public=is_public,
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=poll_number,
                phase="provider_result_download",
                error=exc,
                state=state,
            )
            return

        try:
            with timed_step("normalize_render", log_id) as timing:
                normalized_png, dimensions = normalize_combined_render(
                    raw_result
                )
                timing["input_size_mb"] = size_mb(len(raw_result))
                timing["output_size_mb"] = size_mb(len(normalized_png))
                timing["width"] = dimensions[0]
                timing["height"] = dimensions[1]
            with timed_step("validate_shape", log_id) as timing:
                try:
                    overlap = validate_combined_render_shape(normalized_png)
                except InvalidShapeError as exc:
                    if exc.overlap_ratio is not None:
                        timing["overlap_ratio"] = round(
                            exc.overlap_ratio,
                            6,
                        )
                    raise
                timing["overlap_ratio"] = round(overlap, 6)
        except InvalidShapeError as exc:
            traceback.print_exc()
            fail_stage(
                log_id,
                exc,
                provider_task_id=provider_task_id,
                failure_reason="invalid_shape",
            )
            return
        except Exception as exc:
            traceback.print_exc()
            schedule_stage_recovery(
                log_id=log_id,
                is_public=is_public,
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=poll_number,
                phase="normalize_render",
                error=exc,
                state=state,
            )
            return

        intermediate_key = f"real_to_render_intermediate/{log_id}.png"
        try:
            with timed_step(
                "s3_upload",
                log_id,
                storage_scope="public" if is_public else "private",
                size_mb=size_mb(len(normalized_png)),
            ):
                object_storage().upload_png(
                    intermediate_key,
                    normalized_png,
                    is_public,
                )
        except Exception as exc:
            traceback.print_exc()
            schedule_stage_recovery(
                log_id=log_id,
                is_public=is_public,
                provider_task_id=provider_task_id,
                submitted_at=submitted_at,
                queue_prefix=queue_prefix,
                poll_number=poll_number,
                phase="s3_upload",
                error=exc,
                state=state,
            )
            return

        save_state(
            log_id,
            provider_status="succeeded",
            progress=100,
            intermediate_key=intermediate_key,
            dimensions=f"{dimensions[0]}x{dimensions[1]}",
            recovery_phase="handoff_pending",
        )

    report_status(
        log_id,
        "pending_skin",
        provider_task_id=provider_task_id,
        provider_submission_state="accepted",
        edited_result=intermediate_key,
    )
    try:
        with timed_step("enqueue_render_to_uv", log_id):
            enqueue_render_to_uv_once(
                log_id=log_id,
                is_public=is_public,
                intermediate_key=intermediate_key,
                queue_prefix=queue_prefix,
            )
    except Exception as exc:
        traceback.print_exc()
        schedule_stage_recovery(
            log_id=log_id,
            is_public=is_public,
            provider_task_id=provider_task_id,
            submitted_at=submitted_at,
            queue_prefix=queue_prefix,
            poll_number=poll_number,
            phase="enqueue_render_to_uv",
            error=exc,
            state=state,
            backend_status="pending_skin",
            edited_result=intermediate_key,
        )
        return

    save_state(
        log_id,
        terminal_status="handed_off",
        recovery_phase="complete",
        finished_at=time.time(),
    )
