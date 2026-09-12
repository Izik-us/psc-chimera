"""Strict adapters for optional upstream foundation-model backends.

This module owns the boundary between CHIMERA's local approximations and
native upstream implementations.  A checkpoint is never loaded merely because
its filename looks plausible: the backend must be installed and the checkpoint
must be structurally compatible with the target module.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import torch


class BackendUnavailable(RuntimeError):
    """Raised when an optional native backend cannot be used safely."""


@dataclass(frozen=True)
class BackendSpec:
    """Declarative description of one optional native backend."""

    name: str
    module: str
    checkpoint: Path

    def normalized_checkpoint(self) -> Path:
        return Path(self.checkpoint).expanduser().resolve()


def require_backend(spec: BackendSpec):
    """Import an upstream backend and verify that its checkpoint exists.

    Imports are intentionally lazy so the core CPU/local implementation does
    not acquire heavyweight upstream dependencies merely by importing chimera.
    """
    checkpoint = spec.normalized_checkpoint()
    if not checkpoint.is_file():
        raise BackendUnavailable(
            f"{spec.name} checkpoint not found: {checkpoint}"
        )
    try:
        return import_module(spec.module)
    except ImportError as exc:
        raise BackendUnavailable(
            f"{spec.name} is not installed; install the upstream project before use"
        ) from exc


def _unwrap_state_dict(payload: Any, name: str) -> Mapping[str, torch.Tensor]:
    """Extract a state dict from common checkpoint container formats."""
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not payload:
        raise BackendUnavailable(f"{name} checkpoint does not contain a state dict")
    if not all(isinstance(k, str) for k in payload):
        raise BackendUnavailable(f"{name} checkpoint has invalid parameter names")
    if not all(torch.is_tensor(v) for v in payload.values()):
        raise BackendUnavailable(f"{name} checkpoint contains non-tensor parameters")
    return payload


def load_strict_state(module: torch.nn.Module, checkpoint: str | Path, name: str) -> None:
    """Load a native state dict with an explicit, fail-closed compatibility check."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise BackendUnavailable(f"{name} checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions without weights_only
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise BackendUnavailable(f"Unable to read {name} checkpoint: {path}") from exc

    state = _unwrap_state_dict(payload, name)
    try:
        incompatible = module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise BackendUnavailable(
            f"{name} checkpoint does not match the adapter architecture"
        ) from exc

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BackendUnavailable(
            f"{name} checkpoint is not an exact state-dict match: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def freeze_eval(module: torch.nn.Module) -> torch.nn.Module:
    """Freeze a native pretrained component and force deterministic inference mode."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def load_frozen_native(
    module: torch.nn.Module,
    checkpoint: str | Path,
    name: str,
) -> torch.nn.Module:
    """Strictly load, freeze, and switch a native pretrained component to eval mode."""
    load_strict_state(module, checkpoint, name)
    return freeze_eval(module)
