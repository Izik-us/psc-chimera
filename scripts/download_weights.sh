#!/usr/bin/env bash
# PSC-CHIMERA pretrained asset bootstrap
# Usage: bash scripts/download_weights.sh [WEIGHTS_DIR] [--core|--full]
# core: ESM-2 150M + ESMFold + native ProteinMPNN + native RFdiffusion
# full: core + AlphaFold2/OpenFold v2.3 parameters for the native EvoFormer path
set -euo pipefail
WEIGHTS_DIR="./weights"
PROFILE="core"
for arg in "$@"; do
  case "$arg" in
    --full) PROFILE="full" ;;
    --core) PROFILE="core" ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/download_weights.sh [WEIGHTS_DIR] [--core|--full]' '' 'core: public ESM-2/ESMFold plus native ProteinMPNN/RFdiffusion checkpoints' 'full: core plus AlphaFold2/OpenFold v2.3 parameters for the native EvoFormer adapter' '' 'Native upstream checkpoints are never loaded into CHIMERA approximation classes.'
      exit 0 ;;
    -*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) WEIGHTS_DIR="$arg" ;;
  esac
done
mkdir -p "$WEIGHTS_DIR/upstream" "$WEIGHTS_DIR/alphafold2/params"
CURL_ARGS=(--fail --location --retry 5 --retry-delay 2 --retry-all-errors --continue-at - --progress-bar)
download() {
  local url="$1" out="$2" label="$3" part="${out}.part"
  if [[ -s "$out" ]]; then printf '  ✓ %s already present\n' "$label"; return; fi
  printf '  ↓ %s\n' "$label"
  curl "${CURL_ARGS[@]}" "$url" -o "$part"
  [[ -s "$part" ]] || { echo "ERROR: empty download: $part" >&2; rm -f "$part"; return 1; }
  mv -f "$part" "$out"
}
cat <<EOF
========================================================
 PSC-CHIMERA Pretrained Asset Bootstrap
 Destination: $WEIGHTS_DIR
 Profile:     $PROFILE
========================================================
EOF

download "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt" "$WEIGHTS_DIR/upstream/esm2_t30_150M_UR50D.pt" "ESM-2 t30 150M UR50D"
download "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt" "$WEIGHTS_DIR/upstream/esmfold_3B_v1.pt" "ESMFold v1"
download "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt" "$WEIGHTS_DIR/upstream/proteinmpnn_v48_020.pt" "ProteinMPNN v_48_020"
download "https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt" "$WEIGHTS_DIR/upstream/rfdiffusion_base.pt" "RFdiffusion Base_ckpt"
if [[ "$PROFILE" == "full" ]]; then
  archive="$WEIGHTS_DIR/alphafold2/alphafold_params_2022-12-06.tar"
  download "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" "$archive" "AlphaFold2 v2.3 model parameters"
  if [[ ! -f "$WEIGHTS_DIR/alphafold2/params/.extracted" ]]; then
    printf '  ↓ Extracting AlphaFold2 parameters\n'
    tar -xf "$archive" -C "$WEIGHTS_DIR/alphafold2/params"
    touch "$WEIGHTS_DIR/alphafold2/params/.extracted"
  else
    printf '  ✓ AlphaFold2 parameters already extracted\n'
  fi
fi
cat <<EOF

✓ Bootstrap complete.

ESM-2 for CodonOptimizer:
  $WEIGHTS_DIR/upstream/esm2_t30_150M_UR50D.pt

Native ProteinMPNN/RFdiffusion assets:
  $WEIGHTS_DIR/upstream/

AlphaFold2/OpenFold parameters (full profile):
  $WEIGHTS_DIR/alphafold2/params/

AlphaFold2 has the Evoformer. AlphaFold3 uses a Pairformer instead, so AF3
parameters are not substituted for this Evoformer dependency.

The local CHIMERA approximation classes deliberately do not consume native
upstream checkpoints without compatible adapters.
========================================================
EOF
