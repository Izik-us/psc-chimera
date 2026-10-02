#!/usr/bin/env bash
# Compatibility wrapper for identity-addressed `chimera models fetch`.
set -euo pipefail
WEIGHTS_DIR="./weights"
PROFILE="core"
for arg in "$@"; do
  case "$arg" in
    --full) PROFILE="full" ;;
    --core) PROFILE="core" ;;
    -h|--help)
      printf '%s\n' \
        'Usage: bash scripts/download_weights.sh [CACHE_DIR] [--core|--full]' \
        '' \
        'Downloads are registered in identity-addressed directories with local integrity manifests.' \
        'The full profile also includes the optional large AlphaFold2 parameter archive.'
      exit 0
      ;;
    -*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) WEIGHTS_DIR="$arg" ;;
  esac
done

if ! command -v chimera >/dev/null 2>&1; then
  echo 'Install PSC-CHIMERA first so the chimera models command is available.' >&2
  exit 2
fi

ASSETS=(
  esm2_t30_150m
  esmfold_v1
  proteinmpnn_v48_020
  rfdiffusion_base
)
if [[ "$PROFILE" == "full" ]]; then
  ASSETS+=(alphafold2_params_v23)
fi
for asset in "${ASSETS[@]}"; do
  chimera models fetch "$asset" --cache-dir "$WEIGHTS_DIR"
done
