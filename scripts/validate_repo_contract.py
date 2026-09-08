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

    for path in (
        "chimera/schrodinger_bridge.py",
        "chimera/pcgrad.py",
        "chimera/dpo.py",
        "chimera/bayesian.py",
        "chimera/proteinmpnn.py",
    ):
        if not (root / path).is_file():
            raise SystemExit(f"Missing canonical implementation: {path}")

    print("CHIMERA architecture contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
