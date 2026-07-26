import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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
    provider_model: str
    provider_connect_timeout: int
    provider_read_timeout: int
    image_download_timeout: int
    poll_interval: int
    max_wait: int
    job_timeout: int
    state_ttl: int
    pipeline_version: str
    template_paths: tuple[Path, ...]
    prompt_file: Path | None
    prompt_template: str
    image_size: str
    aspect_ratio: str
    render_to_uv_task: str
    render_to_uv_job_timeout: int
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    public_bucket: str
    private_bucket: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    template_paths = tuple(
        Path(item.strip())
        for item in os.getenv("REAL_TO_RENDER_TEMPLATE_PATHS", "").split(",")
        if item.strip()
    )
    if not template_paths:
        raise RuntimeError(
            "REAL_TO_RENDER_TEMPLATE_PATHS must contain at least one template"
        )

    prompt_file_value = os.getenv("REAL_TO_RENDER_PROMPT_FILE", "").strip()
    prompt_file = Path(prompt_file_value) if prompt_file_value else None
    if prompt_file:
        if not prompt_file.is_file():
            raise FileNotFoundError(
                f"REAL_TO_RENDER_PROMPT_FILE does not exist: {prompt_file}"
            )
        prompt_template = prompt_file.read_text(encoding="utf-8").strip()
    else:
        prompt_template = _required("REAL_TO_RENDER_PROMPT")
    if not prompt_template:
        raise RuntimeError("The real-to-render prompt must not be empty")
    if "{template_refs}" not in prompt_template and len(template_paths) != 3:
        raise RuntimeError(
            "A fixed prompt without {template_refs} requires exactly three templates"
        )

    return Settings(
        redis_url=_required("REDIS_URL"),
        outbound_http_proxy=_optional_http_url("OUTBOUND_HTTP_PROXY"),
        result_queue_key=os.getenv(
            "GENERATE_RESULT_QUEUE_KEY", "generate_results"
        ),
        provider_base_url=_required("IMAGE_API_BASE_URL").rstrip("/"),
        provider_use_proxy=_bool("IMAGE_API_USE_PROXY", False),
        provider_api_key=_required("IMAGE_API_KEY"),
        provider_model=os.getenv("IMAGE_API_MODEL", "nano-banana-pro"),
        provider_connect_timeout=_positive_int(
            "IMAGE_API_CONNECT_TIMEOUT_SECONDS", 5
        ),
        provider_read_timeout=_positive_int(
            "IMAGE_API_READ_TIMEOUT_SECONDS", 20
        ),
        image_download_timeout=_positive_int(
            "IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 30
        ),
        poll_interval=_positive_int(
            "REAL_TO_RENDER_POLL_INTERVAL_SECONDS", 10
        ),
        max_wait=_positive_int("REAL_TO_RENDER_MAX_WAIT_SECONDS", 320),
        job_timeout=_positive_int(
            "REAL_TO_RENDER_JOB_TIMEOUT_SECONDS",
            320,
        ),
        state_ttl=_positive_int("REAL_TO_RENDER_STATE_TTL_SECONDS", 86400),
        pipeline_version=os.getenv(
            "REAL_TO_RENDER_PIPELINE_VERSION",
            "real2render-t41-t51-t52-sking-ddj-v54-v1",
        ),
        template_paths=template_paths,
        prompt_file=prompt_file,
        prompt_template=prompt_template,
        image_size=os.getenv("REAL_TO_RENDER_IMAGE_SIZE", "1K"),
        aspect_ratio=os.getenv("REAL_TO_RENDER_ASPECT_RATIO", "1:1"),
        render_to_uv_task=os.getenv(
            "RENDER_TO_UV_TASK", "worker_tasks.task_render_to_uv"
        ),
        render_to_uv_job_timeout=_positive_int(
            "RENDER_TO_UV_JOB_TIMEOUT_SECONDS", 120
        ),
        aws_access_key_id=_required("AWS_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=_required("AWS_S3_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_REGION", "us-east-2"),
        public_bucket=_required("AWS_BUCKET_NAME"),
        private_bucket=_required("AWS_PRIVATE_BUCKET_NAME"),
    )
