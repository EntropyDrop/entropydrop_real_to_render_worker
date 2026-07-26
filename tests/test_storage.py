import storage


def test_configures_explicit_proxy_for_s3(monkeypatch):
    captured = {}

    def fake_client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(storage.boto3, "client", fake_client)

    storage.ObjectStorage(
        access_key_id="access",
        secret_access_key="secret",
        region="us-east-2",
        public_bucket="public",
        private_bucket="private",
        proxy_url="http://proxy:9100",
    )

    assert captured["service_name"] == "s3"
    assert captured["config"].proxies == {
        "http": "http://proxy:9100",
        "https": "http://proxy:9100",
    }
