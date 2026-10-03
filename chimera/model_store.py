"""Identity-addressed acquisition and integrity checks for upstream models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ArtifactError, IntegrityError


class ModelStoreError(ArtifactError):
    """Base exception for model asset acquisition and integrity errors."""


class ModelAcquisitionError(ModelStoreError):
    """Raised when a declared upstream model cannot be downloaded safely."""

    retryable = True


class ModelIntegrityError(IntegrityError, ModelStoreError):
    """Raised when a cached artifact or its manifest is missing or invalid."""


def _load_dependency_manifest() -> dict[str, Any]:
    path = Path(__file__).with_name("model_dependencies.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelStoreError(f"Unable to read model dependency manifest: {path}") from exc
    if data.get("schema_version") != 1 or not isinstance(data.get("dependencies"), list):
        raise ModelStoreError(f"Unsupported model dependency manifest schema: {path}")
    return data


DEPENDENCY_MANIFEST = _load_dependency_manifest()
DEPENDENCIES = {
    entry["artifact_id"]: entry for entry in DEPENDENCY_MANIFEST["dependencies"]
}
ASSET_ALIASES = {
    "esm2_t30_150m": "esm2-t30-150m-ur50d",
    "esm2_t30_150m_contact_regression": "esm2-t30-150m-contact-regression",
    "esm2_t30_150m_contact_regression": "esm2-t30-150m-contact-regression",
    "esmfold_v1": "esmfold-v1-3b",
    "proteinmpnn_v48_020": "proteinmpnn-v-48-020",
    "rfdiffusion_base": "rfdiffusion-base",
    "alphafold2_params_v23": "alphafold2-params-2022-12-06",
}
MODEL_MANIFEST_VERSION = 1
NATIVE_STATUSES = {
    "NATIVE_VERIFIED",
    "NATIVE_UNVERIFIED",
    "APPROXIMATION",
    "MISSING",
    "INCOMPATIBLE",
    "UNAVAILABLE",
}


@dataclass(frozen=True)
class Asset:
    """An immutable upstream asset declaration loaded from the dependency lock."""

    name: str
    artifact_id: str
    filename: str
    url: str
    metadata: dict[str, Any]


PUBLIC_ASSETS = {
    alias: Asset(
        name=DEPENDENCIES[artifact_id]["model_variant"],
        artifact_id=artifact_id,
        filename=DEPENDENCIES[artifact_id]["filename"],
        url=DEPENDENCIES[artifact_id]["source_url"],
        metadata=DEPENDENCIES[artifact_id],
    )
    for alias, artifact_id in ASSET_ALIASES.items()
}


def cache_root(root: os.PathLike[str] | str | None = None) -> Path:
    """Return the configured model cache (new name first, legacy name supported)."""
    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get("CHIMERA_MODEL_CACHE")
    if configured is None:
        configured = os.environ.get("PSC_CHIMERA_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "psc-chimera" / "models").resolve()


def _asset(key: str) -> Asset:
    if key in PUBLIC_ASSETS:
        return PUBLIC_ASSETS[key]
    for asset in PUBLIC_ASSETS.values():
        if key == asset.artifact_id:
            return asset
    raise KeyError(f"Unknown declared model asset: {key}")


def asset_directory(key: str, root: os.PathLike[str] | str | None = None) -> Path:
    asset = _asset(key)
    return cache_root(root) / asset.artifact_id


def asset_path(key: str, root: os.PathLike[str] | str | None = None) -> Path:
    asset = _asset(key)
    return asset_directory(key, root) / asset.filename


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(key: str, root: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    asset = _asset(key)
    directory = asset_directory(key, root)
    checkpoint = directory / asset.filename
    manifest_path = directory / "manifest.json"
    return {
        "artifact_id": asset.artifact_id,
        "metadata": asset.metadata,
        "checkpoint_path": str(checkpoint),
        "manifest_path": str(manifest_path),
        "present": checkpoint.is_file(),
        "manifest_present": manifest_path.is_file(),
        "verification": verify_asset(key, root),
    }


def list_assets(root: os.PathLike[str] | str | None = None) -> list[dict[str, Any]]:
    entries = []
    for key in ASSET_ALIASES:
        asset = _asset(key)
        checkpoint = asset_path(key, root)
        manifest_path = checkpoint.parent / "manifest.json"
        entries.append(
            {
                "artifact_id": asset.artifact_id,
                "metadata": asset.metadata,
                "checkpoint_path": str(checkpoint),
                "manifest_path": str(manifest_path),
                "present": checkpoint.is_file(),
                "manifest_present": manifest_path.is_file(),
                "verification": {
                    "state": "NOT_VERIFIED" if checkpoint.is_file() else "MISSING",
                    "integrity": "UNKNOWN",
                    "compatibility": "UNKNOWN",
                    "native_status": "NATIVE_UNVERIFIED"
                    if checkpoint.is_file()
                    else "MISSING",
                },
            }
        )
    return entries


def verify_asset(
    key: str, root: os.PathLike[str] | str | None = None
) -> dict[str, Any]:
    """Verify local file and manifest hashes without network access.

    A locally recorded digest establishes cache integrity, not upstream origin.
    Native status changes only when explicit runtime evidence is recorded.
    """
    asset = _asset(key)
    directory = asset_directory(key, root)
    checkpoint = directory / asset.filename
    manifest_path = directory / "manifest.json"
    if not checkpoint.is_file():
        return {
            "artifact_id": asset.artifact_id,
            "state": "MISSING",
            "integrity": "UNAVAILABLE",
            "compatibility": "UNKNOWN",
            "native_status": "MISSING",
            "native_validation": None,
            "message": "required artifact unavailable locally" if asset.metadata["required"] else "artifact not cached",
        }
    if not manifest_path.is_file():
        return {
            "artifact_id": asset.artifact_id,
            "state": "UNKNOWN",
            "integrity": "UNKNOWN",
            "compatibility": "UNKNOWN",
            "native_status": "NATIVE_UNVERIFIED",
            "native_validation": None,
            "message": "checkpoint exists without a CHIMERA integrity manifest",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "artifact_id": asset.artifact_id,
            "state": "CORRUPT",
            "integrity": "INVALID",
            "compatibility": "UNKNOWN",
            "native_status": "NATIVE_UNVERIFIED",
            "native_validation": None,
            "message": f"invalid artifact manifest: {exc}",
        }

    expected_hash = manifest.get("sha256")
    actual_hash = _hash_file(checkpoint)
    native_status = manifest.get("native_status", "NATIVE_UNVERIFIED")
    native_validation = manifest.get("native_validation")
    native_status_valid = (
        isinstance(native_status, str)
        and native_status in NATIVE_STATUSES
        and (
            (
                native_status == "NATIVE_UNVERIFIED"
                and native_validation is None
            )
            or (
                isinstance(native_validation, dict)
                and native_validation.get("status") == native_status
                and isinstance(native_validation.get("detail"), str)
                and bool(native_validation["detail"].strip())
                and isinstance(native_validation.get("evidence"), dict)
                and (
                    native_status != "NATIVE_VERIFIED"
                    or native_validation["evidence"].get("result") == "PASS"
                )
            )
        )
    )
    metadata_fields = (
        "model_family",
        "model_variant",
        "upstream_project",
        "upstream_repository",
        "upstream_revision",
        "release",
        "filename",
        "source_url",
        "format",
        "architecture",
        "license",
        "license_url",
    )
    metadata_matches = (
        manifest.get("manifest_version") == MODEL_MANIFEST_VERSION
        and manifest.get("artifact_id") == asset.artifact_id
        and all(
            manifest.get(field) == asset.metadata.get(field)
            for field in metadata_fields
        )
        and manifest.get("upstream_sha256") == asset.metadata.get("upstream_sha256")
        and manifest.get("sha256_source")
        == (
            "upstream_published"
            if asset.metadata.get("upstream_sha256") is not None
            else "locally_recorded"
        )
        and manifest.get("size_bytes") == checkpoint.stat().st_size
        and native_status_valid
    )
    manifest_hash_valid = (
        isinstance(manifest.get("manifest_sha256"), str)
        and manifest["manifest_sha256"] == _manifest_hash(manifest)
    )
    file_hash_valid = isinstance(expected_hash, str) and expected_hash == actual_hash
    published_hash = asset.metadata.get("upstream_sha256")
    upstream_hash_valid = published_hash is None or actual_hash == published_hash
    if not (metadata_matches and manifest_hash_valid and file_hash_valid and upstream_hash_valid):
        return {
            "artifact_id": asset.artifact_id,
            "state": "CORRUPT",
            "integrity": "INVALID",
            "compatibility": "UNKNOWN",
            "native_status": "NATIVE_UNVERIFIED",
            "native_validation": None,
            "sha256": actual_hash,
            "message": "artifact, manifest, or published upstream checksum mismatch",
        }
    if native_status == "NATIVE_VERIFIED":
        message = (
            "artifact integrity and local native-smoke evidence verified; "
            "upstream origin remains checksum-unconfirmed"
        )
    elif native_status == "UNAVAILABLE":
        message = (
            "artifact integrity verified; native runtime unavailable: "
            f"{native_validation['detail']}"
        )
    elif native_status == "INCOMPATIBLE":
        message = (
            "artifact integrity verified; native integration is incompatible: "
            f"{native_validation['detail']}"
        )
    else:
        message = (
            "integrity verified; model architecture/adapter compatibility "
            "is not established"
        )
    return {
        "artifact_id": asset.artifact_id,
        "state": "INTEGRITY_VERIFIED",
        "integrity": "VALID",
        "integrity_basis": "UPSTREAM_PUBLISHED_SHA256"
        if published_hash is not None
        else "LOCALLY_RECORDED_SHA256",
        "upstream_origin": "CHECKSUM_UNCONFIRMED"
        if published_hash is None
        else "UPSTREAM_CHECKSUM_MATCHED",
        "compatibility": manifest.get("compatibility_status", "UNKNOWN"),
        "native_status": native_status,
        "native_validation": native_validation,
        "sha256": actual_hash,
        "size_bytes": checkpoint.stat().st_size,
        "manifest_sha256": manifest["manifest_sha256"],
        "message": message,
    }


def _write_manifest(asset: Asset, checkpoint: Path, stage: Path) -> dict[str, Any]:
    metadata = asset.metadata
    manifest: dict[str, Any] = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "artifact_id": asset.artifact_id,
        "model_family": metadata["model_family"],
        "model_variant": metadata["model_variant"],
        "upstream_project": metadata["upstream_project"],
        "upstream_repository": metadata["upstream_repository"],
        "upstream_revision": metadata["upstream_revision"],
        "release": metadata["release"],
        "filename": asset.filename,
        "source_url": asset.url,
        "sha256": _hash_file(checkpoint),
        "sha256_source": "upstream_published"
        if metadata.get("upstream_sha256") is not None
        else "locally_recorded",
        "upstream_sha256": metadata.get("upstream_sha256"),
        "size_bytes": checkpoint.stat().st_size,
        "format": metadata["format"],
        "architecture": metadata["architecture"],
        "license": metadata["license"],
        "license_url": metadata["license_url"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "integrity_status": "UPSTREAM_SHA256_VERIFIED"
        if metadata.get("upstream_sha256") is not None
        else "LOCAL_SHA256_RECORDED",
        "compatibility_status": "UNKNOWN",
        "native_status": "NATIVE_UNVERIFIED",
        "native_validation": None,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def ensure_asset(
    key: str,
    root: os.PathLike[str] | str | None = None,
    *,
    force: bool = False,
    timeout_seconds: int = 600,
) -> Path:
    """Explicitly fetch a declared asset, atomically publishing it after hashing."""
    asset = _asset(key)
    directory = asset_directory(key, root)
    destination = directory / asset.filename
    if directory.exists() and any(directory.iterdir()):
        result = verify_asset(key, root)
        if result["state"] == "INTEGRITY_VERIFIED":
            if force:
                raise ModelIntegrityError(
                    "Immutable artifact identities cannot be overwritten; select a new cache or move the existing artifact aside."
                )
            return destination
        raise ModelIntegrityError(
            f"Refusing to overwrite or reuse {asset.artifact_id}: {result['message']}"
        )
    if directory.exists():
        directory.rmdir()

    root_path = cache_root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{asset.artifact_id}.", dir=root_path))
    staged_checkpoint = staging / asset.filename
    try:
        request = urllib.request.Request(
            asset.url, headers={"User-Agent": "psc-chimera-model-store/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                content_length = response.headers.get("Content-Length")
                with staged_checkpoint.open("xb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                if content_length is not None and staged_checkpoint.stat().st_size != int(content_length):
                    raise ModelAcquisitionError(
                        f"Incomplete download for {asset.artifact_id}: expected {content_length} bytes, received {staged_checkpoint.stat().st_size}"
                    )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelAcquisitionError(
                f"Unable to download {asset.artifact_id} from its declared upstream source: {exc}"
            ) from exc

        size = staged_checkpoint.stat().st_size
        if size == 0:
            raise ModelAcquisitionError(f"Upstream returned an empty file for {asset.artifact_id}")
        actual_hash = _hash_file(staged_checkpoint)
        expected_hash = asset.metadata.get("upstream_sha256")
        if expected_hash is not None and actual_hash != expected_hash:
            raise ModelIntegrityError(
                f"Upstream SHA-256 mismatch for {asset.artifact_id}: expected {expected_hash}, received {actual_hash}"
            )
        _write_manifest(asset, staged_checkpoint, staging)
        try:
            os.rename(staging, directory)
        except FileExistsError:
            result = verify_asset(key, root)
            if result["state"] == "INTEGRITY_VERIFIED":
                return destination
            raise ModelIntegrityError(
                f"Another process created an invalid cache entry for {asset.artifact_id}"
            )
        result = verify_asset(key, root)
        if result["state"] != "INTEGRITY_VERIFIED":
            raise ModelIntegrityError(
                f"Newly acquired {asset.artifact_id} failed local verification: {result['message']}"
            )
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def record_native_status(
    key: str,
    *,
    status: str,
    detail: str,
    evidence: dict[str, Any] | None = None,
    root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Persist local runtime evidence without changing artifact bytes or identity."""
    if not isinstance(status, str) or status not in NATIVE_STATUSES:
        raise ValueError(f"Unsupported native model status: {status}")
    if status == "MISSING":
        raise ValueError("MISSING is derived from cache absence and cannot be recorded")
    if not isinstance(detail, str) or not detail.strip():
        raise ValueError("Native model status detail cannot be empty")
    if evidence is not None and not isinstance(evidence, dict):
        raise TypeError("Native model status evidence must be a JSON object")
    if status == "NATIVE_VERIFIED" and (
        evidence is None or evidence.get("result") != "PASS"
    ):
        raise ValueError("NATIVE_VERIFIED requires evidence with result='PASS'")

    verification = verify_asset(key, root)
    if verification["state"] != "INTEGRITY_VERIFIED":
        raise ModelIntegrityError(
            f"Cannot record native status for an unverified artifact: {verification['message']}"
        )

    manifest_path = asset_directory(key, root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_status"] = status
    manifest["native_validation"] = {
        "status": status,
        "detail": detail,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence or {},
    }
    manifest["compatibility_status"] = {
        "NATIVE_VERIFIED": "VALID",
        "INCOMPATIBLE": "INVALID",
    }.get(status, "UNKNOWN")
    manifest["manifest_sha256"] = _manifest_hash(manifest)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return verify_asset(key, root)


def load_esm2(
    root: os.PathLike[str] | str | None = None,
    checkpoint: os.PathLike[str] | str | None = None,
):
    """Load a cached ESM-2 checkpoint through fair-esm after integrity checks."""
    path = asset_path("esm2_t30_150m", root)
    if checkpoint is not None and Path(checkpoint).expanduser().resolve() != path.resolve():
        raise ModelIntegrityError(
            "ESM-2 checkpoints must be resolved by immutable artifact identity; fetch it with `chimera models fetch esm2_t30_150m`."
        )
    verification = verify_asset("esm2_t30_150m", root)
    if verification["state"] != "INTEGRITY_VERIFIED":
        raise ModelIntegrityError(
            f"ESM-2 checkpoint is not integrity-verified: {verification['message']}"
        )
    try:
        import esm
    except ImportError as exc:
        raise ModelStoreError(
            "fair-esm is required to load ESM-2; install the package's optional `esm` dependency."
        ) from exc
    import argparse
    import torch

    torch.serialization.add_safe_globals([argparse.Namespace])
    model_data = torch.load(path, map_location="cpu", weights_only=True)
    regression_path = asset_path("esm2_t30_150m_contact_regression", root)
    regression_verification = verify_asset(
        "esm2_t30_150m_contact_regression", root
    )
    if regression_verification["state"] == "INTEGRITY_VERIFIED":
        regression_data = torch.load(
            regression_path, map_location="cpu", weights_only=True
        )
    elif regression_verification["state"] == "MISSING":
        regression_data = None
    else:
        raise ModelIntegrityError(
            "ESM-2 contact-regression auxiliary is not integrity-verified: "
            f"{regression_verification['message']}"
        )
    if not isinstance(model_data, dict) or (
        regression_data is not None and not isinstance(regression_data, dict)
    ):
        raise ModelIntegrityError("ESM-2 checkpoint payload has an invalid structure")
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            path.stem,
            model_data,
            regression_data,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ModelIntegrityError(
            f"Canonical fair-esm loader rejected the ESM-2 checkpoint: {exc}"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, alphabet, path
