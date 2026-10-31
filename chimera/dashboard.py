"""Read-only terminal dashboard for the CHIMERA library."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .model_store import cache_root, list_assets


def readme_text() -> str:
    source = Path(__file__).resolve().parent.parent / "README.md"
    if source.is_file():
        return source.read_text(encoding="utf-8")
    try:
        return importlib.metadata.metadata("psc-chimera").get_payload() or (
            "PSC-CHIMERA\nComputational protein-engineering research library.\n"
        )
    except importlib.metadata.PackageNotFoundError:
        return "PSC-CHIMERA\nFull README unavailable in this installation.\n"


def _checkpoints() -> list[dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for folder in ("Models", "models", "checkpoints", "runs"):
        root = Path.cwd() / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*.pt"):
            try:
                stat = path.stat()
            except OSError:
                continue
            discovered[str(path.resolve())] = {
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "inspection": "NOT_INSPECTED",
            }
    return sorted(discovered.values(), key=lambda item: item["path"])


def collect_status(cache_dir: str | Path | None = None) -> dict[str, Any]:
    assets = list_assets(cache_dir)
    try:
        import torch

        runtime = {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except ImportError:
        runtime = {
            "torch_available": False, "torch_version": None,
            "cuda_available": False, "cuda_version": None,
            "gpu_count": 0, "devices": [],
        }

    from . import __all__ as public_api

    checkpoints = _checkpoints()
    return {
        "name": "PSC-CHIMERA",
        "version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime": runtime,
        "package_api": {
            "state": "AVAILABLE",
            "public_exports": len(public_api),
            "console_entrypoint": "chimera.cli:main",
        },
        "model_assets": {
            "declared": len(assets),
            "cached": sum(bool(x["present"]) for x in assets),
            "integrity_verified": sum(
                x["verification"]["state"] == "INTEGRITY_VERIFIED" for x in assets
            ),
            "items": [
                {
                    "artifact_id": x["artifact_id"],
                    "present": x["present"],
                    "integrity": x["verification"]["state"],
                    "native_status": x["verification"]["native_status"],
                }
                for x in assets
            ],
        },
        "canonical_checkpoints": {
            "discovered": len(checkpoints),
            "items": checkpoints,
            "note": "Inventoried by path and size only; tensor contents are not loaded.",
        },
        "model_cache": str(cache_root(cache_dir)),
    }


def render_dashboard(*, include_readme: bool = True,
                     cache_dir: str | Path | None = None) -> str:
    status = collect_status(cache_dir)
    lines: list[str] = []
    if include_readme:
        lines.extend([readme_text().rstrip(), ""])
    runtime = status["runtime"]
    assets = status["model_assets"]
    lines.extend([
        "=" * 72,
        "CHIMERA SYSTEM STATUS",
        "=" * 72,
        f"Version: {status['version']}",
        f"Python: {status['python_version']}",
        f"Platform: {status['platform']}",
        f"PyTorch: {runtime['torch_version'] or 'NOT INSTALLED'}",
        f"CUDA: {'AVAILABLE' if runtime['cuda_available'] else 'UNAVAILABLE'}",
        f"GPUs: {runtime['gpu_count']}",
        f"Public package exports: {status['package_api']['public_exports']}",
        f"Model assets: {assets['cached']}/{assets['declared']} cached; "
        f"{assets['integrity_verified']} integrity verified",
        f"Canonical checkpoints discovered: {status['canonical_checkpoints']['discovered']}",
        f"Model cache: {status['model_cache']}",
        "",
        "Integrity verification does not establish native compatibility or biological validation.",
        "Checkpoint files are inventoried without loading tensor contents.",
        "",
        "Commands: chimera --help | chimera status | chimera models list",
        "          chimera train sequence resume --checkpoint Models/CHIMERAv2.pt --dataset data/train.jsonl",
    ])
    return "\n".join(lines)
