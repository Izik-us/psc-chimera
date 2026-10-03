"""Fast PSC-CHIMERA environment diagnostic.

Run with the project's Python interpreter after installation:
    python scripts/check_install.py
"""

from __future__ import annotations

import importlib.util
import sys

REQUIRED = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "einops": "einops",
}

OPTIONAL = {
    "esm": "fair-esm ESM/codon workflows",
    "Bio": "Biopython data acquisition",
    "faiss": "FAISS retrieval",
    "scipy": "SciPy research utilities",
    "matplotlib": "matplotlib visualization",
}


def main() -> int:
    print("PSC-CHIMERA environment diagnostic")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")

    if sys.version_info < (3, 10):
        print("ERROR: PSC-CHIMERA requires Python >= 3.10")
        return 1

    torch = None
    if importlib.util.find_spec("torch") is not None:
        import torch as _torch

        torch = _torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch: MISSING")

    missing = []
    for module, label in REQUIRED.items():
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK  ' if present else 'MISS'} {label}: {module}")
        if not present:
            missing.append(module)

    for module, label in OPTIONAL.items():
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK  ' if present else 'SKIP'} optional {label}: {module}")

    if missing:
        print("\nMissing runtime packages:", ", ".join(missing))
        print("Install the project with: python -m pip install -e .")
        return 1

    try:
        import chimera

        print(f"\nCHIMERA import: OK (version {chimera.__version__})")
    except ImportError as exc:
        print(f"\nCHIMERA import: FAILED: {exc}")
        return 1

    print("\nCore environment looks ready.")
    print("Verify optional model assets with: chimera models verify --offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
