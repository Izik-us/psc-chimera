<div align="center">

# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

[![Tests](https://github.com/Izik-us/psc-chimera/actions/workflows/tests.yml/badge.svg)](https://github.com/Izik-us/psc-chimera/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

*Stage 1 computational design engine of the Pharmacosynthetic Constructor (PSC) Engineering Pipeline*

[Overview](#overview) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Installation](#installation) • [Pipeline](#psc-pipeline) • [Roadmap](#roadmap)

</div>

---

## Overview

CHIMERA designs **NRPS A-domain sequences** that fold and function inside mammalian cells — solving the 30-year unsolved problem of NRPS mammalian expression.

The Pharmacosynthetic Constructor (PSC) is a theoretical biomedical engineering framework for **in situ therapeutic synthesis**: a reprogrammable molecular machine that uses the body's own biochemistry as feedstock to manufacture therapeutic molecules directly inside target cells. CHIMERA is the computational design engine for its catalytic core (Layer 1).

**Key capabilities:**
- Bridges bacterial NRPS chemistry → mammalian-functional design via SE(3) OT-Flow Matching
- Evolutionary context from animal NRPS MSAs (C. elegans, rotifers, springtails) via frozen EvoFormer
- 4-scale hierarchical sequence design: residue → domain → module → icosahedral assembly
- Structural retrieval-augmented generation (FAISS index of PDB A-domain structures)
- True Pareto-front multi-objective optimization across 5 biological objectives
- Direct Preference Optimization (DPO) learning from PROTEUS experimental results
- Bayesian uncertainty + Expected Improvement for active learning batch selection

---

## Architecture

```
Animal NRPS MSA (Stage 0: NCBI, antiSMASH, Suring et al. 2023)
       │
 EvoFormer [FROZEN ~700M params]
       │
 TriangularPairUpdateConnector    ← TRAINABLE ~12M params total
 SubstratePocketConditioner       ←
 EvolCrossAttentionConnector      ←
       │
 SE(3) OT-Flow Matching [FROZEN — RFdiffusion base]
 Bridge: bacterial NRPS → mammalian design
 20 NFE with RK4 (10x faster than DDPM)
       │
 Multi-Scale Hierarchical Sequence Designer
 Scale 1: Residue (ProteinMPNN GNN)
 Scale 2: Domain (A/T/C/TE attention)
 Scale 3: Module-module interface
 Scale 4: Icosahedral face (PSC assembly)
       │
 Pareto Multi-Objective Head (5 objectives)
 + Bayesian Uncertainty (MC Dropout)
 → Ranked Pareto frontier for PROTEUS
```

| Component | Params | Status |
|-----------|--------|--------|
| EvoFormer (48 blocks) | ~700M | Frozen |
| Flow matching backbone | ~50M | Frozen |
| ProteinMPNN base | ~10M | Frozen |
| **Connectors + heads** | **~12M** | **Trainable** |

---

## Quick Start

```python
from chimera import CHIMERAv2, NRPSConstraints
import torch

# Load model (stubs work without checkpoints for testing)
model = CHIMERAv2.from_pretrained(
    flow_ckpt = "weights/rfdiffusion_base.pt",    # download: scripts/download_weights.sh
    mpnn_ckpt = "weights/proteinmpnn_v48_020.pt",
)

# Define NRPS A-domain design constraints
constraints = NRPSConstraints(
    stachelhaus_positions = torch.tensor([235,236,239,278,299,301,322,330,517,518]),
    domain_boundaries     = torch.tensor([[[0,300],[300,400],[400,500],[500,580],[580,600]]]),
    module_boundaries     = torch.tensor([[[0,600],[0,0],[0,0],[0,0],[0,0]]]),
    icosahedral_face      = torch.tensor([7]),
    ppt_serine_position   = 519,
    fixed_mask            = None,
    hotspot_coords        = None,
    hotspot_indices       = None,
    target_substrate      = "PHE",
)

# Design sequences: bridge from bacterial NRPS → mammalian-functional
results = model.design(
    nrps_msa        = msa_tokens,          # from Stage 0 databases
    source_backbone = (bacterial_R, bacterial_t),  # PDB 1AMU
    initial_pair_features = pair_features,
    target_substrate = "PHE",
    n_designs        = 500,
    n_pareto_samples = 50,
)

print(f"Pareto frontier: {results['pareto_count']} sequences")
# → Send top 50 to PROTEUS for mammalian cell screening

# Update model from PROTEUS experimental results (DPO)
model.update_from_proteus(
    survivors     = sequences_that_passed,
    failures      = sequences_that_failed,
    msa           = msa_tokens,
    pair_features = pair_features,
)
```

---

## Installation

### 1. Clone and install

```bash
git clone https://github.com/Izik-us/psc-chimera.git
cd psc-chimera
pip install -e .
```

### 2. Download pretrained weights

```bash
bash scripts/download_weights.sh
```

This downloads ProteinMPNN (~3MB) and RFdiffusion (~440MB) automatically.
For EvoFormer, follow the OpenFold or ESMFold instructions printed by the script.

### 3. Install external dependencies (optional, for production)

```bash
# OpenFold (EvoFormer backbone)
git clone https://github.com/aqlaboratory/openfold.git
pip install -e openfold/

# RFdiffusion
git clone https://github.com/RosettaCommons/RFdiffusion.git
pip install -e RFdiffusion/

# ProteinMPNN
git clone https://github.com/dauparas/ProteinMPNN.git
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Companion Model: CodonOptimizer

The `CodonOptimizer` prepares PROTEUS-validated NRPS sequences for mRNA delivery.

```python
from chimera import optimize_nrps_for_mammalian_expression

result = optimize_nrps_for_mammalian_expression(
    aa_sequence = your_nrps_sequence,
    beam_search = True,
    verbose     = True,
)

print(f"CAI: {result['cai']:.3f}")        # target ≥ 0.96
print(f"GC:  {result['gc_content']:.3f}") # target 0.58-0.65
# result['dna_sequence'] → send to Trilink/Aldevron for mRNA synthesis
```

Implements Fath et al. 2011 nine-parameter optimization with a neural
autoregressive decoder (ESM-2 → TransformerDecoder → synonymous mask).

---

## PSC Pipeline

```
Stage 0:  Sequence retrieval
          NCBI (XP_018648700, NIUQ01002120.1), antiSMASH DB
          Suring et al. 2023 — 199 confirmed animal NRPS clusters

Stage 1:  CHIMERA v2 computational design        ← THIS REPO
          500 candidates → Pareto frontier → 50 for PROTEUS

Stage 1.5: PoET evolutionary plausibility scoring
           Truong Jr & Bepler 2023 (NeurIPS)

Stage 2:  CodonOptimizer mRNA preparation        ← THIS REPO
          CAI ≥ 0.96, GC 58-65%, all bad motifs removed

Stage 3:  PROTEUS directed evolution
          HEK293 mammalian cells, 4-6 rounds per module
          CHIMERA updated via DPO after each round

Stage 4:  Polycistronic mRNA delivery
          Single LNP → intracellular icosahedral self-assembly
          240nm PSC forms inside target cell

Stage 5:  PSC functional validation
          LC-MS product quantification
          Sensor ring activation assay
```

---

## Production Setup (Items 1-9)

The codebase contains stub implementations for the three pretrained backbones.
See the production setup guide for step-by-step instructions for each item.

Items that can be done immediately (no GPU needed):
- Items 7, 8, 9: code fixes — random sampling, Hamming diversity, block counts

Items requiring GPU (do with Colab or local machine):
- Item 4: ESM-2 150M swap into CodonOptimizer (~2 hours)
- Item 3: ProteinMPNN integration (~3 hours)
- Item 1: EvoFormer/OpenFold (~1 day)
- Item 2: RFdiffusion weight transfer (~1 day)

Hardware recommendations:
- **MacBook Pro M4 Max 48GB** — best overall (48GB unified memory ≈ 48GB VRAM)
- RTX 4090 Laptop 16GB — best Windows/Linux option
- Google Colab Pro ($10/mo) or RunPod (~$0.74/hr RTX 4090) while waiting

---

## Roadmap

- [x] CHIMERA v1 — basic EvoFormer + SE3Denoiser + ProteinMPNN connectors
- [x] CHIMERA v2 — flow matching, multi-scale designer, RAG, DPO, Pareto
- [x] CodonOptimizer — autoregressive + expression critic
- [x] Test suite — shape/integration tests, CI/CD
- [ ] Production weight loading — Items 1-9 (in progress)
- [ ] Stage 0 data pipeline — NRPS sequence retrieval scripts
- [ ] First training run on Fath et al. codon optimization data
- [ ] PROTEUS integration loop
- [ ] First NRPS design → wet lab validation

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Suring et al. 2023 *Genes* | Animal NRPS sequences, Stage 0 templates |
| Fath et al. 2011 *PLoS ONE* | 9-parameter codon optimization, training data |
| Lipman et al. 2022 *ICLR* | OT-Flow Matching (replaces DDPM) |
| Yim et al. 2023 | SE(3) flow matching for proteins |
| Rafailov et al. 2023 *NeurIPS* | Direct Preference Optimization (DPO) |
| Truong Jr & Bepler 2023 *NeurIPS* | PoET — evolutionary fitness scoring |
| Watson et al. 2023 *Nature* | RFdiffusion — backbone generation |
| Jumper et al. 2021 *Nature* | AlphaFold2 EvoFormer |
| Dauparas et al. 2022 *Science* | ProteinMPNN — sequence design |

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
<sub>PSC Engineering Pipeline · CHIMERA v2 · Theoretical Biomedical Engineering Framework</sub>
</div>
