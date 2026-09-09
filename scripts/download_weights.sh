#!/usr/bin/env bash
# PSC-CHIMERA Weight Downloader
#
# Usage:
#   bash scripts/download_weights.sh [WEIGHTS_DIR]
#   bash scripts/download_weights.sh ./weights
#
# Idempotent and resumable. Downloads are written to .part files first so an
# interrupted transfer can never be mistaken for a complete checkpoint.

set -euo pipefail

WEIGHTS_DIR="${1:-./weights}"
mkdir -p "$WEIGHTS_DIR"

CURL_ARGS=(
  --fail
  --location
  --retry 5
  --retry-delay 2
  --retry-all-errors
  --continue-at -
  --progress-bar
)

download() {
  local url="$1"
  local out="$2"
  local label="$3"
  local part="${out}.part"

  if [[ -s "$out" ]]; then
    printf '  ✓ %s already present: %s\n' "$label" "$out"
    return 0
  fi

  printf '  ↓ %s\n' "$label"
  printf '    %s\n' "$out"
  curl "${CURL_ARGS[@]}" "$url" -o "$part"

  if [[ ! -s "$part" ]]; then
    echo "ERROR: download completed but produced an empty file: $part" >&2
    rm -f "$part"
    return 1
  fi

  mv -f "$part" "$out"
  printf '  ✓ %s ready\n' "$label"
}

cat <<EOF
========================================================
 PSC-CHIMERA Weight Downloader
 Destination: $WEIGHTS_DIR
========================================================
EOF

download \
  "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt" \
  "$WEIGHTS_DIR/proteinmpnn_v48_020.pt" \
  "ProteinMPNN v_48_020"

download \
  "https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt" \
  "$WEIGHTS_DIR/rfdiffusion_base.pt" \
  "RFdiffusion Base_ckpt"

cat <<EOF

OpenFold / AlphaFold and PoET remain optional native-backend artifacts.
The local approximation classes intentionally do not load incompatible
upstream checkpoints.

Next:
  pip install -e .
  python scripts/run_design.py --help
========================================================
EOF
