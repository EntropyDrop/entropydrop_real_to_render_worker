# EntropyDrop Real-to-Render Worker (sample)

This is an isolated worker for stage 1 of the image-to-skin pipeline. It does
not import or copy `SkingDataset/DDJ_real2render`.

## Ownership

```text
entropydrop_backend
  -> queue_real_to_render
  -> this CPU/I/O worker
       submit provider task once
       persist provider_task_id in Redis and report it to the backend
       schedule one short poll every 10 seconds
       upload real_to_render_intermediate/<log_id>.png
  -> queue_render_to_uv
  -> entropydrop_gpu_worker
```

The backend remains the only service allowed to update the SQL user balance.
This worker reports `failed`; the backend performs one idempotent refund and
adds a positive `CreditLog(action="refund")`.

## Failure policy

- Provider `failed`, including `failure_reason="error"`: fail once, no retry.
- Provider moderation failures: fail once, no retry.
- Provider or download HTTP timeout: fail once, no retry.
- Overall stage-1 timeout (default 320 seconds): fail once, no retry.
- Each scheduled poll RQ execution has a 320-second hard timeout, configured
  by `REAL_TO_RENDER_JOB_TIMEOUT_SECONDS`.
- S3 or handoff error: fail once, no retry.
- Every sample RQ job is enqueued with `retry=None`.

The provider documentation says `failure_reason="error"` may be resubmitted.
This sample intentionally does not do so because a resubmission can create a
second billable generation.

## Polling

The submit job returns after obtaining the provider task ID. It does not sleep.
Each `poll_real_to_render` execution performs one GET and either:

1. schedules another poll after `REAL_TO_RENDER_POLL_INTERVAL_SECONDS`;
2. reports terminal failure; or
3. uploads the result and hands off to `queue_render_to_uv`.

The provider result URL is downloaded immediately because it is valid for only
two hours.

## Templates, prompt, and pipeline version

The versioned stage-1 inputs are:

1. `templates/template41.png` as 图2;
2. `templates/template51.png` as 图3;
3. `templates/template52.png` as 图4;
4. `prompts/real_to_render.zh-hans.txt`.

The default bundle identifier is
`real2render-t41-t51-t52-sking-ddj-v54-v1`. The corresponding stage-2
checkpoint is `/root/Sking/SKING_DDJ_v54.pt` in the GPU container. A production
pipeline version should continue to identify the provider model, ordered
templates, prompt, Dense UV checkpoint, and renderer mappings together.

## Local setup

```bash
cp .env.example .env
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/python run_worker.py
```

Scale concurrency by running 2-4 worker processes/containers. Do not add a
thread pool to the FastAPI process or GPU process.

## HTTP proxy and Redis

By default, requests to `IMAGE_API_BASE_URL` are sent directly:

- `IMAGE_API_USE_PROXY=false`: provider submit and poll requests bypass the
  proxy;
- `OUTBOUND_HTTP_PROXY`: provider result-image downloads and S3 traffic use
  this proxy;
- set `IMAGE_API_USE_PROXY=true` only when provider submit and poll requests
  should also use `OUTBOUND_HTTP_PROXY`.

For a proxy listening on the Docker host:

```dotenv
OUTBOUND_HTTP_PROXY=http://host.docker.internal:9100
IMAGE_API_USE_PROXY=false
REDIS_URL=redis://:replace-me@host.docker.internal:6380/0
```

The second line is not an HTTP proxy setting. Redis uses its own TCP protocol,
so `REDIS_URL` must point to Redis directly or to a local TCP forward like the
GPU worker's `autossh -L` tunnel. On Linux Docker, add
`host.docker.internal:host-gateway` to the container if that hostname is not
already available.

When the HTTP proxy and Redis tunnel run as sidecars in the same Compose
network, use service names instead:

```dotenv
OUTBOUND_HTTP_PROXY=http://http-proxy:9100
IMAGE_API_USE_PROXY=false
REDIS_URL=redis://:replace-me@redis-tunnel:6380/0
```

Do not put credentials directly in the committed `.env.example`; inject the
real proxy URL, Redis URL, and AWS credentials through the deployment secret
store.

## Step timing logs

Each stage-1 job emits one JSON log record per significant step. The record
contains `log_id`, `step`, `duration_seconds`, and `outcome`. File-transfer
records also contain `size_mb` (1 MB = 1024 x 1024 bytes); provider poll
records contain `poll_number`,
`provider_status`, and `provider_progress`.

Logged steps are `s3_download`, `prepare_api_request`,
`provider_submit_api`, `provider_status_api`, `provider_result_download`,
`normalize_render`, `s3_upload`, and `enqueue_render_to_uv`. Signed URLs,
credentials, prompts, and image contents are not logged.

## Backend enqueue contract

The eventual backend branch for the new model should enqueue:

```python
from rq.job import Callback

q_real_to_render.enqueue(
    "tasks.submit_real_to_render",
    args=(log.id, log.is_public, log.source, content_type, prefix),
    job_id=f"generation_{log.id}_real_to_render",
    job_timeout=60,
    retry=None,
    on_failure=Callback("tasks.real_to_render_job_failure", timeout=20),
    on_stopped=Callback("tasks.real_to_render_job_stopped", timeout=20),
)
```

The new model must use `recoverable=False`. The backend should retain the
original upload in `source`; this worker reports the normalized render in
`edited_result`.

The GPU worker sample contract is:

```python
def task_render_to_uv(
    log_id: str,
    is_public: bool,
    source: str,
    content_type: str,
    pipeline_version: str,
): ...
```

That function should preload `/root/Sking/SKING_DDJ_v54.pt`, SigLIP2, and
mappings once, report stage `render_to_uv`, and also use no RQ retry.

## License

GNU Affero General Public License v3.0. See `LICENSE`.
