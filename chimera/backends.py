"""Strict adapters for optional upstream foundation-model backends.

The adapters intentionally fail closed: an upstream checkpoint is never loaded
into a structurally different local approximation.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import torch


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendSpec:
    name: str
    module: str
    checkpoint: Path


def require_backend(spec: BackendSpec):
    """Import an upstream backend and verify its checkpoint exists."""
    if not spec.checkpoint.is_file():
        raise BackendUnavailable(f"{spec.name} checkpoint not found: {spec.checkpoint}")
    try:
        return import_module(spec.module)
    except ImportError as exc:
        raise BackendUnavailable(
            f"{spec.name} is not installed; install the upstream project before use"
        ) from exc


def load_strict_state(module, checkpoint: str | Path, name: str) -> None:
    """Load a native state dict with no missing or unexpected parameters."""
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise BackendUnavailable(
            f"{name} checkpoint does not match the adapter architecture"
        ) from exc
