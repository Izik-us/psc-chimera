"""Versioned checkpoint manifests for merge-safe CHIMERA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import hashlib
import json
import platform
from pathlib import Path
import subprocess

from ._version import __version__


@dataclass(frozen=True)
class CheckpointManifest:
    """Machine-readable description of the architecture and provenance a checkpoint belongs to."""

    format_version: int = 4
    serialization_revision: int = 1
    chimera_version: str = __version__
    transport: str = "se3_schrodinger_bridge"
    sequence_policy: str = "autoregressive"
    geometry_edge_dim: int = 28
    objective_count: int = 5
    objective_schema_hash: str | None = None
    uncertainty: str = "mc_dropout_epistemic"
    acquisition: str = "gaussian_expected_improvement"
    config_hash: str | None = None
    state_schema_hash: str | None = None
    git_commit: str | None = None
    git_worktree_clean: bool | None = None
    source_tree_hash: str | None = None
    dataset_manifest: str | None = None
    dataset_version: str | None = None
    dataset_hash: str | None = None
    preprocessing_hash: str | None = None
    environment_hash: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    platform: str | None = None
    numpy_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    cuda_device_names: str | None = None
    deterministic_algorithms_enabled: bool | None = None
    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    matmul_allow_tf32: bool | None = None
    cudnn_allow_tf32: bool | None = None
    training_regime: str | None = None
    random_seed: int | None = None
    min_validation_examples: int | None = None
    optimizer_class: str | None = None
    optimizer_hash: str | None = None
    scheduler_class: str | None = None
    scheduler_hash: str | None = None

    def has_complete_provenance(self) -> bool:
        required = (
            "config_hash",
            "state_schema_hash",
            "objective_schema_hash",
            "git_commit",
            "source_tree_hash",
            "dataset_manifest",
            "dataset_version",
            "dataset_hash",
            "preprocessing_hash",
            "environment_hash",
            "python_version",
            "torch_version",
            "platform",
            "numpy_version",
            "training_regime",
            "optimizer_class",
            "optimizer_hash",
            "scheduler_class",
            "scheduler_hash",
        )
        runtime_flags = (
            self.cuda_available,
            self.deterministic_algorithms_enabled,
            self.cudnn_deterministic,
            self.cudnn_benchmark,
            self.matmul_allow_tf32,
            self.cudnn_allow_tf32,
        )
        return (
            all(getattr(self, name) for name in required)
            and self.git_worktree_clean is not None
            and self.min_validation_examples is not None
            and all(value is not None for value in runtime_flags)
        )

    def validate(self, expected: "CheckpointManifest" | None = None) -> None:
        if self.format_version not in (1, 2, 3, 4):
            raise ValueError(f"unsupported checkpoint format_version={self.format_version}")
        if self.serialization_revision < 0:
            raise ValueError("serialization_revision must be non-negative")
        if self.transport != "se3_schrodinger_bridge":
            raise ValueError("checkpoint does not declare the canonical SE(3) SB transport")
        if self.sequence_policy != "autoregressive":
            raise ValueError("CHIMERA sequence policy must be autoregressive")
        if self.geometry_edge_dim != 28:
            raise ValueError("checkpoint geometry contract must use 28-D ProteinMPNN-inspired edges")
        if self.objective_count != 5:
            raise ValueError("CHIMERA expects five objective channels")
        if self.uncertainty != "mc_dropout_epistemic":
            raise ValueError("checkpoint uncertainty contract must be MC-dropout epistemic")
        if self.acquisition != "gaussian_expected_improvement":
            raise ValueError("checkpoint acquisition contract must be Gaussian expected improvement")
        for name in (
            "config_hash", "state_schema_hash", "objective_schema_hash", "git_commit", "source_tree_hash", "dataset_manifest",
            "dataset_version", "dataset_hash", "preprocessing_hash", "environment_hash",
            "python_version", "torch_version", "platform", "training_regime",
            "numpy_version", "cuda_version", "cuda_device_names", "optimizer_class",
            "optimizer_hash", "scheduler_class", "scheduler_hash",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if self.random_seed is not None and (not isinstance(self.random_seed, int) or self.random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer or None")
        if self.git_worktree_clean is not None and not isinstance(self.git_worktree_clean, bool):
            raise ValueError("git_worktree_clean must be a boolean or None")
        for name in (
            "cuda_available", "deterministic_algorithms_enabled", "cudnn_deterministic",
            "cudnn_benchmark", "matmul_allow_tf32", "cudnn_allow_tf32",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None")
        if self.cudnn_version is not None and not isinstance(self.cudnn_version, int):
            raise ValueError("cudnn_version must be an integer or None")
        if self.min_validation_examples is not None and (
            not isinstance(self.min_validation_examples, int) or self.min_validation_examples < 1
        ):
            raise ValueError("min_validation_examples must be a positive integer or None")


def state_schema_hash(state_dict: Mapping[str, Any]) -> str:
    """Hash parameter names, tensor shapes and dtypes without hashing tensor data."""
    entries = []
    for name in sorted(state_dict):
        value = state_dict[name]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            entries.append((name, tuple(value.shape), str(value.dtype)))
        else:
            entries.append((name, type(value).__name__))
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 fingerprint for a JSON-serializable model configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_provenance() -> dict[str, Any]:
    try:
        import numpy as np

        numpy_version = str(np.__version__)
    except Exception:
        numpy_version = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
        cudnn_version = torch.backends.cudnn.version()
        cuda_device_names = (
            json.dumps(
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                separators=(",", ":"),
            )
            if cuda_available else None
        )
        deterministic_algorithms_enabled = bool(torch.are_deterministic_algorithms_enabled())
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
        matmul_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        torch_version = None
        cuda_available = None
        cuda_version = None
        cudnn_version = None
        cuda_device_names = None
        deterministic_algorithms_enabled = None
        cudnn_deterministic = None
        cudnn_benchmark = None
        matmul_allow_tf32 = None
        cudnn_allow_tf32 = None
    result = {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "platform": platform.platform(),
        "numpy_version": numpy_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "cuda_device_names": cuda_device_names,
        "deterministic_algorithms_enabled": deterministic_algorithms_enabled,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "matmul_allow_tf32": matmul_allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
    }
    required_runtime = (
        result["python_version"],
        result["torch_version"],
        result["numpy_version"],
        result["platform"],
    )
    result["environment_hash"] = config_hash(result) if all(required_runtime) else None
    return result


def git_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Hash commit, tracked source diffs, and only untracked source/config files."""
    root = Path(repository_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            [
                "git", "diff", "--binary", "HEAD", "--",
                "*.py", "*.pyi", "*.toml", "*.yaml", "*.yml", "*.ini",
            ],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        untracked_paths = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_clean": None, "source_tree_hash": None}

    source_extensions = {".ini", ".py", ".pyi", ".toml", ".yaml", ".yml"}
    untracked_files = {}
    try:
        for relative in sorted(set(untracked_paths) - {""}):
            path = root / relative
            if path.suffix.lower() in source_extensions and path.is_file() and "data" not in path.parts:
                untracked_files[relative.replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree_hash = config_hash({
            "commit": commit,
            "tracked_source_diff": hashlib.sha256(diff).hexdigest(),
            "untracked_source_files": untracked_files,
        })
    except OSError:
        tree_hash = None
    return {
        "git_commit": commit or None,
        "git_worktree_clean": not bool(status.strip()),
        "source_tree_hash": tree_hash,
    }


def save_manifest(path: str | Path, manifest: CheckpointManifest | None = None) -> None:
    manifest = manifest or CheckpointManifest()
    manifest.validate()
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> CheckpointManifest:
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = CheckpointManifest(**data)
    manifest.validate()
    return manifest


def validate_checkpoint_compatibility(
    actual: CheckpointManifest,
    expected: CheckpointManifest,
    *,
    migration_id: str | None = None,
) -> str:
    """Return a compatibility verdict for checkpoint provenance validation."""
    try:
        actual.validate()
        expected.validate()
    except (TypeError, ValueError):
        return "incompatible"
    if actual == expected:
        return "exact-compatible" if actual.has_complete_provenance() else "unverified-provenance"

    if migration_id == "v4-serialization-metadata-v0-to-v1":
        if (
            actual.format_version == expected.format_version == 4
            and actual.serialization_revision == 0
            and expected.serialization_revision == 1
        ):
            actual_dict = asdict(actual)
            expected_dict = asdict(expected)
            actual_dict.pop("serialization_revision")
            expected_dict.pop("serialization_revision")
            if actual_dict == expected_dict and actual.has_complete_provenance() and expected.has_complete_provenance():
                return "expected-compatible"
        return "incompatible"
    return "incompatible"
