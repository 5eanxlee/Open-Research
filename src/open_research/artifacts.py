from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .config import Settings


@dataclass(slots=True)
class ArtifactPayload:
    kind: str
    extension: str
    content_type: str
    data: bytes


@dataclass(slots=True)
class ArtifactReference:
    kind: str
    uri: str
    content_type: str
    size_bytes: int
    sha256: str

    def as_metadata(self) -> dict[str, str | int]:
        return asdict(self)


class ArtifactStore(ABC):
    @abstractmethod
    async def save_artifact(
        self,
        *,
        run_id: str,
        source_id: str,
        payload: ArtifactPayload,
    ) -> ArtifactReference:
        raise NotImplementedError


class DisabledArtifactStore(ArtifactStore):
    async def save_artifact(
        self,
        *,
        run_id: str,
        source_id: str,
        payload: ArtifactPayload,
    ) -> ArtifactReference:
        raise RuntimeError("Artifact storage is disabled.")


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()

    async def save_artifact(
        self,
        *,
        run_id: str,
        source_id: str,
        payload: ArtifactPayload,
    ) -> ArtifactReference:
        target = (
            self.root
            / run_id
            / source_id
            / f"{payload.kind}.{payload.extension.lstrip('.')}"
        )
        digest = sha256(payload.data).hexdigest()

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload.data)

        await asyncio.to_thread(_write)
        return ArtifactReference(
            kind=payload.kind,
            uri=str(target),
            content_type=payload.content_type,
            size_bytes=len(payload.data),
            sha256=digest,
        )


class S3ArtifactStore(ArtifactStore):
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "open-research",
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url

    async def save_artifact(
        self,
        *,
        run_id: str,
        source_id: str,
        payload: ArtifactPayload,
    ) -> ArtifactReference:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "boto3 is not installed. Install the storage extra to enable S3 artifacts."
            ) from exc

        key = (
            f"{self.prefix}/{run_id}/{source_id}/"
            f"{payload.kind}.{payload.extension.lstrip('.')}"
        )
        digest = sha256(payload.data).hexdigest()

        def _upload() -> None:
            client = boto3.client("s3", endpoint_url=self.endpoint_url)
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload.data,
                ContentType=payload.content_type,
            )

        await asyncio.to_thread(_upload)
        return ArtifactReference(
            kind=payload.kind,
            uri=f"s3://{self.bucket}/{key}",
            content_type=payload.content_type,
            size_bytes=len(payload.data),
            sha256=digest,
        )


def build_artifact_store(settings: Settings) -> ArtifactStore | None:
    backend = settings.resolved_artifact_store_backend
    if backend == "disabled":
        return None
    if backend == "local":
        return LocalArtifactStore(settings.artifact_store_path)
    if settings.artifact_store_s3_bucket is None:
        raise ValueError("ARTIFACT_STORE_S3_BUCKET must be configured for S3 artifact storage.")
    return S3ArtifactStore(
        bucket=settings.artifact_store_s3_bucket,
        prefix=settings.artifact_store_s3_prefix,
        endpoint_url=settings.artifact_store_s3_endpoint_url,
    )
