"""Static P0-P2 architecture checks for local/CI execution."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    compat = ROOT / "chimera" / "runtime_compat.py"
    assert not compat.exists(), "runtime_compat.py must not return"

    init = (ROOT / "chimera" / "__init__.py").read_text(encoding="utf-8")
    assert "monkey" not in init.lower()
    assert "runtime_compat" not in init

    flow = (ROOT / "chimera" / "flow_matching.py").read_text(encoding="utf-8")
    assert "from .lie import" in flow or "import lie" in flow

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
