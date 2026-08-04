import pytest
import requests

from provider import (
    ProviderProtocolError,
    ProviderStatus,
    RealToRenderProvider,
)


class FakeResponse:
    def __init__(self, payload=None, content=b"image"):
        self.payload = payload or {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self):
        self.trust_env = True
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return FakeResponse({"id": "provider-task"})

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/v1/api/result"):
            return FakeResponse(
                {
                    "id": "provider-task",
                    "status": "running",
                    "progress": 50,
                }
            )
        return FakeResponse(content=b"result-image")


class ProxyFailingSession(RecordingSession):
    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if kwargs.get("proxies") is not None:
            raise requests.ReadTimeout("proxy stalled")
        return FakeResponse(content=b"direct-result")


def test_parses_documented_success_response():
    status = ProviderStatus.from_payload(
        {
            "id": "task-1",
            "results": [
                {
                    "url": "https://example.com/result.png",
                    "content": "description",
                }
            ],
            "progress": 100,
            "status": "succeeded",
            "failure_reason": "",
            "error": "",
        }
    )

    assert status.task_id == "task-1"
    assert status.status == "succeeded"
    assert status.progress == 100
    assert status.result_url == "https://example.com/result.png"


def test_parses_documented_failure_response_without_retry_signal():
    status = ProviderStatus.from_payload(
        {
            "id": "task-2",
            "results": [],
            "progress": 20,
            "status": "failed",
            "failure_reason": "error",
            "error": "Invalid input parameters",
        }
    )

    assert status.status == "failed"
    assert status.failure_reason == "error"
    assert status.error == "Invalid input parameters"


def test_success_requires_result_url():
    with pytest.raises(ProviderProtocolError):
        ProviderStatus.from_payload(
            {
                "id": "task-3",
                "results": [],
                "progress": 100,
                "status": "succeeded",
            }
        )


def test_unknown_status_is_terminal_protocol_error():
    with pytest.raises(ProviderProtocolError):
        ProviderStatus.from_payload(
            {
                "id": "task-4",
                "progress": 50,
                "status": "queued-forever",
            }
        )


def test_provider_api_bypasses_proxy_but_result_download_uses_it():
    session = RecordingSession()
    provider = RealToRenderProvider(
        base_url="https://provider.example",
        api_key="secret",
        model="model",
        connect_timeout=5,
        read_timeout=20,
        download_timeout=30,
        proxy_url="http://proxy:9100",
        session=session,
    )

    provider.submit(["source"], "prompt", "1:1", "1K")
    provider.get_status("provider-task")
    assert provider.download_result("https://cdn.example/result.png") == (
        b"result-image"
    )

    assert session.trust_env is False
    assert provider.proxies == {
        "http": "http://proxy:9100",
        "https": "http://proxy:9100",
    }
    assert session.calls[0][2]["proxies"] is None
    assert session.calls[1][2]["proxies"] is None
    assert session.calls[2][2]["proxies"] == provider.proxies


def test_provider_api_can_explicitly_opt_in_to_proxy():
    provider = RealToRenderProvider(
        base_url="https://provider.example",
        api_key="secret",
        model="model",
        connect_timeout=5,
        read_timeout=20,
        download_timeout=30,
        proxy_url="http://proxy:9100",
        api_use_proxy=True,
    )

    assert provider.api_proxies == provider.proxies


def test_result_download_falls_back_from_proxy_to_direct():
    session = ProxyFailingSession()
    provider = RealToRenderProvider(
        base_url="https://provider.example",
        api_key="secret",
        model="model",
        connect_timeout=5,
        read_timeout=20,
        download_timeout=30,
        proxy_url="http://proxy:9100",
        download_direct_fallback=True,
        session=session,
    )

    assert provider.download_result("https://cdn.example/result.png") == (
        b"direct-result"
    )
    assert session.calls[0][2]["proxies"] == provider.proxies
    assert session.calls[1][2]["proxies"] is None
