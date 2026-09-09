#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

PY="${ROOT}/.venv/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
"$PY" -m pip install -e .

cat <<EOF

PSC-CHIMERA environment ready.
Python: $PY

Run:
  $PY scripts/run_design.py --help

Development/test dependencies:
  $PY -m pip install -e '.[dev]'

Optional molecular-dynamics validation:
  $PY -m pip install -e '.[md]'

Supported native checkpoints:
  bash scripts/download_weights.sh ./weights
EOF
