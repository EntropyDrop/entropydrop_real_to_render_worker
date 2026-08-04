from dataclasses import dataclass
from typing import Any

import requests


TERMINAL_FAILURE_STATUSES = {"failed", "violation"}
TERMINAL_STATUSES = {"succeeded"} | TERMINAL_FAILURE_STATUSES
KNOWN_STATUSES = TERMINAL_STATUSES | {"running"}


class ProviderProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderStatus:
    task_id: str
    status: str
    progress: float
    result_url: str | None
    content: str | None
    failure_reason: str
    error: str

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], fallback_task_id: str = ""
    ) -> "ProviderStatus":
        task_id = str(payload.get("id") or fallback_task_id).strip()
        status = str(payload.get("status") or "").strip().lower()
        if not task_id:
            raise ProviderProtocolError("Provider response has no task id")
        if status not in KNOWN_STATUSES:
            raise ProviderProtocolError(
                f"Provider returned unknown status {status!r}"
            )

        try:
            progress = float(payload.get("progress", 0))
        except (TypeError, ValueError) as exc:
            raise ProviderProtocolError(
                f"Provider returned invalid progress: {payload.get('progress')!r}"
            ) from exc

        result_url = None
        content = None
        results = payload.get("results") or []
        if results:
            first = results[0] or {}
            result_url = first.get("url")
            content = first.get("content")
        if status == "succeeded" and not result_url:
            raise ProviderProtocolError(
                "Provider reported succeeded without a result URL"
            )

        return cls(
            task_id=task_id,
            status=status,
            progress=max(0.0, min(100.0, progress)),
            result_url=result_url,
            content=content,
            failure_reason=str(payload.get("failure_reason") or ""),
            error=str(payload.get("error") or ""),
        )


class RealToRenderProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        connect_timeout: int,
        read_timeout: int,
        download_timeout: int,
        proxy_url: str | None = None,
        api_use_proxy: bool = False,
        download_direct_fallback: bool = True,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = (connect_timeout, read_timeout)
        self.download_timeout = (connect_timeout, download_timeout)
        self.session = session or requests.Session()
        # Do not implicitly inherit HTTP_PROXY/HTTPS_PROXY from the process.
        # Every proxied request in this worker must opt in explicitly below.
        self.session.trust_env = False
        self.proxies = (
            {"http": proxy_url, "https": proxy_url}
            if proxy_url
            else None
        )
        self.api_proxies = self.proxies if api_use_proxy else None
        self.download_direct_fallback = download_direct_fallback
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def submit(
        self,
        images: list[str],
        prompt: str,
        aspect_ratio: str,
        image_size: str,
    ) -> str:
        response = self.session.post(
            f"{self.base_url}/v1/api/generate",
            headers=self.headers,
            json={
                "model": self.model,
                "prompt": prompt,
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
                "replyType": "async",
                "images": images,
            },
            timeout=self.timeout,
            proxies=self.api_proxies,
        )
        response.raise_for_status()
        task_id = str(response.json().get("id") or "").strip()
        if not task_id:
            raise ProviderProtocolError(
                "Provider submit response has no task id"
            )
        return task_id

    def get_status(self, task_id: str) -> ProviderStatus:
        response = self.session.get(
            f"{self.base_url}/v1/api/result",
            headers={"Authorization": self.headers["Authorization"]},
            params={"id": task_id},
            timeout=self.timeout,
            proxies=self.api_proxies,
        )
        response.raise_for_status()
        return ProviderStatus.from_payload(
            response.json(), fallback_task_id=task_id
        )

    def download_result(self, url: str) -> bytes:
        routes = [self.proxies]
        if self.proxies and self.download_direct_fallback:
            routes.append(None)

        last_error: requests.RequestException | None = None
        for proxies in routes:
            try:
                response = self.session.get(
                    url,
                    timeout=self.download_timeout,
                    proxies=proxies,
                )
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc

        assert last_error is not None
        raise last_error
