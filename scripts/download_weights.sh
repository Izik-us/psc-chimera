#!/usr/bin/env bash
# PSC-CHIMERA Weight Downloader
#
# Usage:
#   bash scripts/download_weights.sh [WEIGHTS_DIR]
#   bash scripts/download_weights.sh ./weights
#
# The script is deliberately idempotent and resumable. It downloads only
# missing artifacts, verifies that each transfer produced a non-empty file,
# and uses curl retries so interrupted/slow connections are less painful.

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

  if [[ -s "$out" ]]; then
    printf '  ✓ %s already present: %s\n' "$label" "$out"
    return 0
  fi

  printf '  ↓ %s\n' "$label"
  printf '    %s\n' "$out"
  curl "${CURL_ARGS[@]}" "$url" -o "$out"

  if [[ ! -s "$out" ]]; then
    echo "ERROR: download completed but produced an empty file: $out" >&2
    rm -f "$out"
    return 1
  fi
  printf '  ✓ %s ready\n' "$label"
}

cat <<EOF
========================================================
 PSC-CHIMERA Weight Downloader
 Destination: $WEIGHTS_DIR
========================================================
EOF

# ProteinMPNN is small and directly usable by the native adapter.
download \
  "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt" \
  "$WEIGHTS_DIR/proteinmpnn_v48_020.pt" \
  "ProteinMPNN v_48_020"

# RFdiffusion checkpoint. The native adapter verifies compatibility before use.
download \
  "https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt" \
  "$WEIGHTS_DIR/rfdiffusion_base.pt" \
  "RFdiffusion Base_ckpt"

cat <<EOF

[3/4] OpenFold / AlphaFold parameters
  CHIMERA's local EvoFormer is an approximation and is NOT checkpoint-compatible
  with OpenFold/AlphaFold parameters. Do not load those weights into the local
  approximation. Use an official OpenFold installation/checkpoint when the
  native adapter is enabled.

[4/4] PoET
  PoET is optional for the current local pipeline. Its checkpoint is not
  downloaded automatically because licensing and artifact hosting can change.
  See the README for the supported native-backend workflow.

========================================================
 Downloaded native checkpoint artifacts:
   $WEIGHTS_DIR/proteinmpnn_v48_020.pt
   $WEIGHTS_DIR/rfdiffusion_base.pt

Next:
  pip install -e .
  python scripts/run_design.py --help
========================================================
EOF
