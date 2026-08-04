#!/usr/bin/env bash
# ============================================================
# PSC-CHIMERA → GitHub Push Script
# Izik-us/psc-chimera
# ============================================================
# Run this once from inside the psc-chimera/ folder:
#   bash push_to_github.sh
# ============================================================

set -e

GITHUB_USER="Izik-us"
REPO_NAME="psc-chimera"
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "========================================================"
echo " PSC-CHIMERA → GitHub"
echo " Repo: ${REPO_URL}"
echo "========================================================"
echo ""

# ── Step 1: Check dependencies ───────────────────────────────
echo "[1/5] Checking dependencies..."

if ! command -v git &>/dev/null; then
    echo "  ✗ git not found. Install from https://git-scm.com/"
    exit 1
fi
echo "  ✓ git found: $(git --version)"

# Check for GitHub CLI (preferred) or fall back to git + token
if command -v gh &>/dev/null; then
    USE_GH=true
    echo "  ✓ GitHub CLI found: $(gh --version | head -1)"
else
    USE_GH=false
    echo "  ⚠ GitHub CLI not found — will use git + token auth"
    echo "    (Recommend installing gh: https://cli.github.com/)"
fi

# ── Step 2: Initialize git repo ──────────────────────────────
echo ""
echo "[2/5] Initializing git repository..."

if [ ! -d ".git" ]; then
    git init
    echo "  ✓ Git initialized"
else
    echo "  ✓ Git already initialized"
fi

git add -A
git commit -m "Initial commit: CHIMERA v2 PSC Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture

- chimera_v2.py: Full model with flow matching, multi-scale designer,
  structural RAG, DPO, Pareto optimization, Bayesian uncertainty
- flow_matching.py: SE(3) OT-Flow Matching (20 NFE vs 200 for DDPM)
- multi_objective.py: RAG + DPO + Pareto + uncertainty + multi-scale GNN
- codon_optimizer.py: ESM-2 encoder + autoregressive decoder + expression critic
- chimera_v1.py: Reference implementation
- evoformer.py / proteinmpnn.py / se3_diffusion.py: Backbone wrappers
- data/training_data.py: Complete data sourcing guide
- tests/: Full test suite (shape + integration + math tests)
- scripts/: Weight downloader + CLI design runner
- .github/workflows/: CI/CD with GitHub Actions

Part of the Pharmacosynthetic Constructor (PSC) engineering pipeline.
Stage 1 computational design engine for NRPS A-domain engineering." 2>/dev/null || \
git commit --allow-empty -m "Update: CHIMERA v2 PSC Engineering Pipeline"

echo "  ✓ Changes committed"

# ── Step 3: Create GitHub repo and push ──────────────────────
echo ""
echo "[3/5] Creating GitHub repository..."

if [ "$USE_GH" = true ]; then
    # GitHub CLI path — handles auth interactively
    if ! gh auth status &>/dev/null; then
        echo "  Logging into GitHub..."
        gh auth login
    fi
    echo "  ✓ Authenticated"

    # Create repo if it doesn't exist
    if gh repo view "${GITHUB_USER}/${REPO_NAME}" &>/dev/null; then
        echo "  ✓ Repo already exists on GitHub"
    else
        gh repo create "${REPO_NAME}" \
            --public \
            --description "CHIMERA v2: Computational design engine for PSC NRPS engineering. Stage 1 of the Pharmacosynthetic Constructor pipeline." \
            --homepage "https://github.com/${GITHUB_USER}/${REPO_NAME}"
        echo "  ✓ Repository created: ${REPO_URL}"
    fi

    # Set remote and push
    git remote remove origin 2>/dev/null || true
    git remote add origin "${REPO_URL}"
    git branch -M main
    git push -u origin main --force
    echo "  ✓ Code pushed to GitHub"

else
    # Manual git path — ask for token
    echo ""
    echo "  GitHub CLI not available. Using Personal Access Token."
    echo ""
    echo "  To create a token:"
    echo "  1. Go to: https://github.com/settings/tokens/new"
    echo "  2. Note: 'psc-chimera push'"
    echo "  3. Expiration: 90 days"
    echo "  4. Scopes: check 'repo' (full control)"
    echo "  5. Click 'Generate token' — copy it"
    echo ""
    read -s -p "  Paste your GitHub token here (hidden): " GH_TOKEN
    echo ""

    if [ -z "$GH_TOKEN" ]; then
        echo "  ✗ No token provided. Exiting."
        exit 1
    fi

    AUTH_URL="https://${GITHUB_USER}:${GH_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

    # Create repo via API
    echo "  Creating repo via GitHub API..."
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Authorization: token ${GH_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        https://api.github.com/user/repos \
        -d "{
            \"name\": \"${REPO_NAME}\",
            \"description\": \"CHIMERA v2: Computational design engine for PSC NRPS engineering\",
            \"private\": false,
            \"auto_init\": false
        }")

    if [ "$HTTP_STATUS" = "201" ]; then
        echo "  ✓ Repository created: ${REPO_URL}"
    elif [ "$HTTP_STATUS" = "422" ]; then
        echo "  ✓ Repository already exists"
    else
        echo "  ✗ API error (HTTP $HTTP_STATUS). Check your token has 'repo' scope."
        exit 1
    fi

    # Push
    git remote remove origin 2>/dev/null || true
    git remote add origin "$AUTH_URL"
    git branch -M main
    git push -u origin main --force
    echo "  ✓ Code pushed"
fi

# ── Step 4: Add topics/tags ───────────────────────────────────
echo ""
echo "[4/5] Adding repo topics..."
if [ "$USE_GH" = true ]; then
    gh repo edit "${GITHUB_USER}/${REPO_NAME}" \
        --add-topic "protein-design" \
        --add-topic "nrps" \
        --add-topic "drug-delivery" \
        --add-topic "bioinformatics" \
        --add-topic "pytorch" \
        --add-topic "flow-matching" \
        --add-topic "alphafold" \
        2>/dev/null && echo "  ✓ Topics added" || echo "  ⚠ Topics skipped (gh version may not support)"
else
    echo "  ⚠ Skipping topics (requires GitHub CLI)"
fi

# ── Step 5: Done ─────────────────────────────────────────────
echo ""
echo "[5/5] Done!"
echo ""
echo "========================================================"
echo ""
echo "  ✓ Repo live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo "  Next steps:"
echo "  1. Download weights:"
echo "     bash scripts/download_weights.sh"
echo ""
echo "  2. Run tests (no GPU needed):"
echo "     pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "     pip install einops numpy pytest"
echo "     pytest tests/ -v"
echo ""
echo "  3. When you have a machine with GPU:"
echo "     pip install -e ."
echo "     python scripts/run_design.py --substrate PHE --n-designs 100"
echo ""
echo "========================================================"
