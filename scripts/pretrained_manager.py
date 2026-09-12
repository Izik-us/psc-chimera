"""Unified pretrained-model asset manager for PSC-CHIMERA."""
from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Asset:
    name: str
    filename: str
    url: Optional[str]
    license_note: str = ""


PUBLIC_ASSETS = {
    "esm2_t30_150m": Asset(
        "esm2_t30_150M_UR50D",
        "esm2_t30_150M_UR50D.pt",
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "Meta FAIR ESM checkpoint; verify upstream terms before redistribution.",
    ),
    "esmfold_v1": Asset(
        "esmfold_v1",
        "esmfold_3B_v1.pt",
        "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt",
        "ESMFold public checkpoint; verify upstream terms.",
    ),
    "proteinmpnn_v48_020": Asset(
        "ProteinMPNN v_48_020",
        "proteinmpnn_v48_020.pt",
        "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt",
        "Native ProteinMPNN checkpoint; requires the compatible upstream implementation.",
    ),
    "rfdiffusion_base": Asset(
        "RFdiffusion Base_ckpt",
        "rfdiffusion_base.pt",
        "https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt",
        "Native RFdiffusion checkpoint; requires the compatible upstream implementation.",
    ),
}


def cache_root(root: Optional[os.PathLike[str] | str] = None) -> Path:
    default = Path.home() / ".cache" / "psc-chimera" / "models"
    return Path(root or os.environ.get("PSC_CHIMERA_MODEL_CACHE", default))


def ensure_asset(key: str, root=None, *, force: bool = False) -> Path:
    if key not in PUBLIC_ASSETS:
        raise KeyError(f"Unknown public asset: {key}")
    asset = PUBLIC_ASSETS[key]
    destination = cache_root(root) / asset.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        return destination

    fd, temp_name = tempfile.mkstemp(prefix=asset.filename + ".", suffix=".part", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        urllib.request.urlretrieve(asset.url, temporary)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty download for {asset.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ensure_public_assets(root=None) -> dict[str, Path]:
    return {key: ensure_asset(key, root) for key in PUBLIC_ASSETS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PSC-CHIMERA pretrained asset manager")
    parser.add_argument("--root", default=None)
    parser.add_argument("--asset", choices=[*PUBLIC_ASSETS, "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.asset == "all":
        for key, path in ensure_public_assets(args.root).items():
            print(f"{key}: {path}")
    else:
        print(ensure_asset(args.asset, args.root, force=args.force))
