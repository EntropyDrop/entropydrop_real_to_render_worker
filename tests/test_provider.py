import pytest
import requests

from provider import (
    ProviderProtocolError,
    ProviderStatus,
    RealToRenderProvider,
)


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


def test_configures_explicit_proxy_for_provider_and_downloads():
    session = requests.Session()
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

    assert provider.proxies == {
        "http": "http://proxy:9100",
        "https": "http://proxy:9100",
    }
