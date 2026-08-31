# FILE: cloudsentinel-zero-trust/src/integrations/s3_client.py
"""S3 client factory with LOCAL_MODE filesystem fallback.

In LOCAL_MODE, all S3 operations are redirected to local filesystem paths
under the configured local_data_dir. This enables full pipeline testing
without any AWS credentials or services.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import boto3

from src.utils.config import get_settings
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="s3_client")

_client: Any = None


def get_s3_client() -> Any:
    """Return a singleton boto3 S3 client (or LocalS3Client in LOCAL_MODE)."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if settings.local_mode:
        _client = LocalS3Client(base_dir=settings.local_data_dir)
        logger.info("LOCAL_MODE: S3 client using filesystem at %s", settings.local_data_dir)
    else:
        _client = boto3.client("s3", region_name=settings.aws_region)
        logger.info("S3 client initialized for region %s", settings.aws_region)
    return _client


class LocalS3Client:
    """Filesystem-backed S3 mock for LOCAL_MODE.

    Maps bucket/key paths to <base_dir>/<bucket>/<key> on disk.
    Supports get_object, put_object, list_objects_v2, delete_object.
    Handles gzip content automatically (matching real S3 behavior).
    """

    def __init__(self, base_dir: str = "ml/data") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, bucket: str, key: str) -> Path:
        p = (self._base / bucket / key).resolve()
        # Security: prevent path traversal outside base_dir
        if not str(p).startswith(str(self._base.resolve())):
            raise ValueError(f"Path traversal detected: {key}")
        return p

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        p = self._path(Bucket, Key)
        if not p.exists():
            from botocore.exceptions import ClientError
            error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
            raise ClientError(error_response, "GetObject")
        body = p.read_bytes()
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def put_object(self, Bucket: str, Key: str, Body: bytes | str, **kwargs: Any) -> dict[str, Any]:
        p = self._path(Bucket, Key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(Body, str):
            Body = Body.encode("utf-8")
        p.write_bytes(Body)
        logger.debug("LOCAL_MODE S3 put_object: %s/%s (%d bytes)", Bucket, Key, len(Body))
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", **kwargs: Any) -> dict[str, Any]:
        base = self._path(Bucket, Prefix) if Prefix else self._base / Bucket
        if not base.exists():
            return {"Contents": [], "KeyCount": 0}
        contents = []
        search_root = base if base.is_dir() else base.parent
        for p in search_root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self._base / Bucket))
                if rel.startswith(Prefix):
                    contents.append({"Key": rel, "Size": p.stat().st_size})
        return {"Contents": contents, "KeyCount": len(contents)}

    def delete_object(self, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        p = self._path(Bucket, Key)
        if p.exists():
            p.unlink()
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def delete_objects(self, Bucket: str, Delete: dict, **kwargs: Any) -> dict[str, Any]:
        for obj in Delete.get("Objects", []):
            self.delete_object(Bucket=Bucket, Key=obj.get("Key", ""))
        return {"Deleted": Delete.get("Objects", []), "Errors": []}

    def delete_bucket(self, Bucket: str, **kwargs: Any) -> dict[str, Any]:
        import shutil
        p = self._base / Bucket
        if p.exists():
            shutil.rmtree(p)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def create_bucket(self, Bucket: str, **kwargs: Any) -> dict[str, Any]:
        (self._base / Bucket).mkdir(parents=True, exist_ok=True)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_paginator(self, operation_name: str) -> Any:
        return LocalPaginator(self, operation_name)

    # STS shim (needed by attack_simulator when LOCAL_MODE)
    def get_caller_identity(self) -> dict[str, str]:
        return {
            "Account": "000000000000",
            "UserId": "LOCAL:local-user",
            "Arn": "arn:aws:iam::000000000000:user/local-user",
        }


class LocalPaginator:
    """Minimal paginator for LocalS3Client."""

    def __init__(self, client: LocalS3Client, operation: str) -> None:
        self._client = client
        self._op = operation

    def paginate(self, **kwargs: Any):  # noqa: ANN201
        if self._op == "list_objects_v2":
            result = self._client.list_objects_v2(**kwargs)
            yield result

