#!/usr/bin/env python3
"""Validate the repository's documented CHIMERA architecture contract."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    required_terms = (
        "Schrödinger-bridge",
        "PCGrad",
        "DPO",
        "Bayesian uncertainty",
        "Expected Improvement",
        "autoregressive",
        "28 edge features",
    )
    missing = [term for term in required_terms if term.lower() not in readme.lower()]
    if missing:
        raise SystemExit("README is missing architecture-contract terms: " + ", ".join(missing))

    required_files = (
        "chimera/schrodinger_bridge.py",
        "chimera/pcgrad.py",
        "chimera/dpo.py",
        "chimera/bayesian.py",
        "chimera/proteinmpnn.py",
        "chimera/autoregressive_policy.py",
        "chimera/architecture.py",
    )
    for path in required_files:
        if not (root / path).is_file():
            raise SystemExit(f"Missing canonical implementation: {path}")

    forbidden_paths = (
        "chimera/runtime_compat.py",
    )
    for path in forbidden_paths:
        if (root / path).exists():
            raise SystemExit(f"Legacy runtime compatibility shim must not exist: {path}")

    init_text = (root / "chimera/__init__.py").read_text(encoding="utf-8")
    forbidden_markers = ("runtime_compat", "monkeypatch", "monkey-patch")
    for marker in forbidden_markers:
        if marker in init_text.lower():
            raise SystemExit(f"Package initializer contains forbidden runtime patching marker: {marker}")

    print("CHIMERA architecture contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
