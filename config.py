import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int_tuple(
    name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw = os.getenv(name, ",".join(str(value) for value in default))
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off"
    )


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_http_url(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"{name} must be an http:// or https:// proxy URL"
        )
    return value


@dataclass(frozen=True)
class Settings:
    redis_url: str
    outbound_http_proxy: str | None
    result_queue_key: str
    provider_base_url: str
    provider_use_proxy: bool
    provider_api_key: str
    provider_connect_timeout: int
    provider_read_timeout: int
    image_download_timeout: int
    image_download_direct_fallback: bool
    poll_interval: int
    delayed_poll_interval: int
    max_wait: int
    hard_wait: int
    result_recovery_window: int
    recovery_retry_intervals: tuple[int, ...]
    poll_job_retry_max: int
    job_timeout: int
    state_ttl: int
    recovery_interval: int
    orphan_grace: int
    render_to_uv_task: str
    render_to_uv_job_timeout: int
    render_to_uv_retry_max: int
    render_to_uv_retry_intervals: tuple[int, ...]
    prompts_root_dir: str
    templates_root_dir: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    public_bucket: str
    private_bucket: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        redis_url=_required("REDIS_URL"),
        outbound_http_proxy=_optional_http_url("OUTBOUND_HTTP_PROXY"),
        result_queue_key=os.getenv(
            "GENERATE_RESULT_QUEUE_KEY", "generate_results"
        ),
        provider_base_url=_required("IMAGE_API_BASE_URL").rstrip("/"),
        provider_use_proxy=_bool("IMAGE_API_USE_PROXY", False),
        provider_api_key=_required("IMAGE_API_KEY"),
        provider_connect_timeout=_positive_int(
            "IMAGE_API_CONNECT_TIMEOUT_SECONDS", 5
        ),
        provider_read_timeout=_positive_int(
            "IMAGE_API_READ_TIMEOUT_SECONDS", 20
        ),
        image_download_timeout=_positive_int(
            "IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 30
        ),
        image_download_direct_fallback=_bool(
            "IMAGE_DOWNLOAD_DIRECT_FALLBACK", True
        ),
        poll_interval=_positive_int(
            "REAL_TO_RENDER_POLL_INTERVAL_SECONDS", 10
        ),
        delayed_poll_interval=_positive_int(
            "REAL_TO_RENDER_DELAYED_POLL_INTERVAL_SECONDS", 60
        ),
        max_wait=_positive_int("REAL_TO_RENDER_MAX_WAIT_SECONDS", 320),
        hard_wait=_positive_int(
            "REAL_TO_RENDER_HARD_WAIT_SECONDS", 1800
        ),
        result_recovery_window=_positive_int(
            "REAL_TO_RENDER_RESULT_RECOVERY_SECONDS", 5400
        ),
        recovery_retry_intervals=_positive_int_tuple(
            "REAL_TO_RENDER_RETRY_INTERVALS_SECONDS",
            (2, 5, 15, 30, 60),
        ),
        poll_job_retry_max=_positive_int(
            "REAL_TO_RENDER_POLL_JOB_RETRY_MAX", 5
        ),
        job_timeout=_positive_int(
            "REAL_TO_RENDER_JOB_TIMEOUT_SECONDS",
            320,
        ),
        state_ttl=_positive_int("REAL_TO_RENDER_STATE_TTL_SECONDS", 86400),
        recovery_interval=_positive_int(
            "REAL_TO_RENDER_RECOVERY_INTERVAL_SECONDS",
            15,
        ),
        orphan_grace=_positive_int(
            "REAL_TO_RENDER_ORPHAN_GRACE_SECONDS",
            75,
        ),
        render_to_uv_task=os.getenv(
            "RENDER_TO_UV_TASK", "worker_tasks.task_render_to_uv"
        ),
        render_to_uv_job_timeout=_positive_int(
            "RENDER_TO_UV_JOB_TIMEOUT_SECONDS", 120
        ),
        render_to_uv_retry_max=_positive_int(
            "RENDER_TO_UV_RETRY_MAX", 5
        ),
        render_to_uv_retry_intervals=_positive_int_tuple(
            "RENDER_TO_UV_RETRY_INTERVALS_SECONDS",
            (5, 15, 30, 60, 120),
        ),
        prompts_root_dir=os.getenv(
            "PROMPTS_ROOT_DIR", str(BASE_DIR / "prompts")
        ),
        templates_root_dir=os.getenv(
            "TEMPLATES_ROOT_DIR", str(BASE_DIR / "templates")
        ),
        aws_access_key_id=_required("AWS_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=_required("AWS_S3_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_REGION", "us-east-2"),
        public_bucket=_required("AWS_BUCKET_NAME"),
        private_bucket=_required("AWS_PRIVATE_BUCKET_NAME"),
    )
