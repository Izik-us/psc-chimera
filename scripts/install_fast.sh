#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
CPU_ONLY=0
WITH_DEV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --with-dev) WITH_DEV=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
PYTHON=".venv/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel setuptools

if [[ "$CPU_ONLY" -eq 1 ]]; then
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# Install the project from its declared dependency contract.  When --cpu-only
# was requested, the already-installed CPU torch satisfies setup.py without
# pulling a second PyTorch build.  This keeps the fast path complete rather
# than maintaining a hand-copied dependency list that can drift from setup.py.
"$PYTHON" -m pip install -e .

if [[ "$WITH_DEV" -eq 1 ]]; then
  "$PYTHON" -m pip install -r requirements-dev.txt
fi

"$PYTHON" scripts/check_install.py

echo
echo "PSC-CHIMERA fast environment ready."
echo "Run: $PYTHON scripts/run_design.py --help"
echo "Check: $PYTHON scripts/check_install.py"
echo "Weights: ./scripts/download_weights.sh ./weights"
