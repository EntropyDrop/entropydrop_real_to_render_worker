import boto3
from botocore.config import Config


class ObjectStorage:
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        public_bucket: str,
        private_bucket: str,
        proxy_url: str | None = None,
    ):
        self.public_bucket = public_bucket
        self.private_bucket = private_bucket
        config_options = {
            "connect_timeout": 5,
            "read_timeout": 20,
            "retries": {
                "total_max_attempts": 3,
                "mode": "standard",
            },
        }
        if proxy_url:
            config_options["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(**config_options),
        )

    def _bucket(self, is_public: bool) -> str:
        return self.public_bucket if is_public else self.private_bucket

    def download(self, key: str, is_public: bool) -> bytes:
        response = self.client.get_object(
            Bucket=self._bucket(is_public),
            Key=key,
        )
        return response["Body"].read()

    def upload_png(self, key: str, content: bytes, is_public: bool) -> str:
        request = {
            "Bucket": self._bucket(is_public),
            "Key": key,
            "Body": content,
            "ContentType": "image/png",
        }
        if is_public:
            request["ACL"] = "public-read"
        self.client.put_object(**request)
        return key
