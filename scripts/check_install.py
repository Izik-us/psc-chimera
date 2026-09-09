"""Fast PSC-CHIMERA environment diagnostic.

Run with the project's Python interpreter after installation:
    python scripts/check_install.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REQUIRED = {
    "torch": "PyTorch",
    "transformers": "Transformers / ESM-2 support",
    "esm": "fair-esm",
    "Bio": "Biopython",
    "faiss": "FAISS structural retrieval",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "pandas": "pandas",
    "h5py": "h5py",
    "rich": "rich",
    "typer": "typer",
}


def main() -> int:
    print("PSC-CHIMERA environment diagnostic")
    print("=" * 40)
    print(f"Python: {torch.__config__.show().splitlines()[0] if torch else 'unknown'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    missing = []
    for module, label in REQUIRED.items():
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK  ' if present else 'MISS'} {label}: {module}")
        if not present:
            missing.append(module)

    root = Path(__file__).resolve().parents[1]
    weights = root / "weights"
    for name in ("proteinmpnn_v48_020.pt", "rfdiffusion_base.pt"):
        path = weights / name
        print(f"{'OK  ' if path.is_file() and path.stat().st_size else 'MISS'} checkpoint: {path}")

    if missing:
        print("\nMissing runtime packages:", ", ".join(missing))
        print("Install the project with: python -m pip install -e .")
        return 1

    print("\nEnvironment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
