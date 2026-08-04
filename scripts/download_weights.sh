#!/usr/bin/env bash
# PSC-CHIMERA Weight Downloader
# Run this script once to download all pretrained model weights.
# Usage: bash scripts/download_weights.sh [--weights-dir /path/to/weights]

set -e

WEIGHTS_DIR=${1:-"./weights"}
mkdir -p "$WEIGHTS_DIR"

echo "========================================================"
echo " PSC-CHIMERA Pretrained Weight Downloader"
echo " Downloading to: $WEIGHTS_DIR"
echo "========================================================"

# ── 1. ProteinMPNN (smallest, fastest) ──────────────────────────────────────
echo ""
echo "[1/4] ProteinMPNN weights..."
if [ ! -f "$WEIGHTS_DIR/proteinmpnn_v48_020.pt" ]; then
    wget -q --show-progress \
        "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt" \
        -O "$WEIGHTS_DIR/proteinmpnn_v48_020.pt"
    echo "  ✓ ProteinMPNN downloaded (~3MB)"
else
    echo "  ✓ ProteinMPNN already present"
fi

# ── 2. RFdiffusion ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] RFdiffusion Base model..."
if [ ! -f "$WEIGHTS_DIR/rfdiffusion_base.pt" ]; then
    wget -q --show-progress \
        "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt" \
        -O "$WEIGHTS_DIR/rfdiffusion_base.pt"
    echo "  ✓ RFdiffusion downloaded (~440MB)"
else
    echo "  ✓ RFdiffusion already present"
fi

# ── 3. OpenFold / AlphaFold2 weights ────────────────────────────────────────
echo ""
echo "[3/4] AlphaFold2 parameters (for EvoFormer)..."
echo "  NOTE: AlphaFold2 weights require manual download."
echo "  1. Go to: https://github.com/google-deepmind/alphafold"
echo "  2. Run: bash scripts/download_alphafold_params.sh $WEIGHTS_DIR/alphafold/"
echo "  3. OR use ESMFold instead (automatic, see README):"
echo "     python -c \"import esm; esm.pretrained.esmfold_v1()\""
echo "  Skipping — see instructions above."

# ── 4. PoET weights ─────────────────────────────────────────────────────────
echo ""
echo "[4/4] PoET weights..."
if [ ! -f "$WEIGHTS_DIR/poet_weights.pt" ]; then
    echo "  Downloading PoET from Zenodo (CC BY-NC-SA 4.0)..."
    wget -q --show-progress \
        "https://zenodo.org/record/10061322/files/poet.ckpt" \
        -O "$WEIGHTS_DIR/poet_weights.pt" 2>/dev/null || \
    echo "  PoET weights available at: https://zenodo.org/record/10061322"
else
    echo "  ✓ PoET already present"
fi

echo ""
echo "========================================================"
echo " Download complete. Update paths in chimera_v2.py:"
echo ""
echo "   CHIMERAv2.from_pretrained("
echo "       flow_ckpt = '$WEIGHTS_DIR/rfdiffusion_base.pt',"
echo "       mpnn_ckpt = '$WEIGHTS_DIR/proteinmpnn_v48_020.pt',"
echo "   )"
echo "========================================================"
