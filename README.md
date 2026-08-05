<div align="center">

# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

[![Tests](https://github.com/Izik-us/psc-chimera/actions/workflows/tests.yml/badge.svg)](https://github.com/Izik-us/psc-chimera/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

*Stage 1 computational design engine of the Pharmacosynthetic Constructor (PSC) Engineering Pipeline*

[Overview](#overview) • [What CHIMERA Engineers](#what-chimera-engineers) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Installation](#installation) • [Pipeline](#psc-pipeline) • [Roadmap](#roadmap)

</div>

---

## Overview

The **Pharmacosynthetic Constructor (PSC)** is a theoretical biomedical engineering framework for *in situ therapeutic synthesis*: a reprogrammable composite molecular machine that uses the body's own biochemistry as feedstock to manufacture therapeutic molecules directly inside target cells — turning the body into a precision pharmacological manufacturing system.

**CHIMERA is the complete computational design engine for the PSC's catalytic core (Layer 1).**

Layer 1 is an engineered NRPS/PKS hybrid assembly line operating inside mammalian cells. Building it requires designing not just one enzyme domain but an entire coordinated molecular factory: every domain in the assembly line, every junction between modules, every novel catalytic insert, and the full multi-module architecture that strings them together into a continuous synthesis pathway.

CHIMERA solves the 30-year unsolved problem of NRPS mammalian expression and module incompatibility by bridging bacterial NRPS chemistry toward mammalian-functional designs through SE(3) OT-Flow Matching, evolutionary context from animal NRPS homologs, and iterative learning from PROTEUS experimental results.

---

## What CHIMERA Engineers

CHIMERA is not an A-domain engineer. It is a **full NRPS machinery design system** covering every component of the Layer 1 catalytic core:

### Canonical NRPS Domains

| Domain | Function | CHIMERA's Role |
|--------|----------|----------------|
| **A-domain** (Adenylation) | Substrate recognition and activation as aminoacyl-AMP | Selectivity code transplant from bacterial analogs; PoET-scored against animal NRPS family; substrate-conditioned backbone generation |
| **T-domain** (Thiolation / PCP) | Tethers substrate via 20Å phosphopantetheine arm; shuttles intermediates between catalytic domains | PPant attachment site design; co-evolving interface with A-domain enforced via EvoFormer pair representation |
| **C-domain** (Condensation) | Catalyzes peptide/ester/C-C bond formation between tethered intermediates | Standard amide, ester, and C-C bond variants; novel bond chemistries via engineered C-domain variants; split-reporter selection in PROTEUS |
| **TE-domain** (Thioesterase) | Releases finished product; controls linear vs cyclic product geometry | Cyclization vs linear release engineering; tunable release kinetics that determine product Cmax and local concentration profile |
| **E-domain** (Epimerization) | Converts L-amino acids to D-configuration | D-amino acid incorporation for products with improved protease resistance |
| **Cy-domain** (Cyclization) | Heterocyclization of Cys/Ser/Thr residues | Thiazoline/oxazoline ring formation for cyclic peptide natural product analogs |
| **Mt-domain** (N-Methylation) | N-methylates backbone amides | Increased membrane permeability and protease resistance in product peptides |

### De Novo Enzyme Insert Domains

CHIMERA designs domains from scratch using a theozyme → RFdiffusion → ProteinMPNN workflow for reaction chemistries **not present in any natural NRPS/PKS**:

- Novel ring closures not achievable by natural TE-domains
- Bioorthogonal reactions using endogenous cofactors (SAM, NADPH)
- Non-standard functional group additions (fluorination, phosphorylation)
- Reductive chemistry beyond natural PKS ketoreductases

These are the source of Tier 3 and Tier 4 PSC outputs — molecular architectures that no existing biosynthetic machinery produces.

### Module-Module Interface Engineering

The **30-year NRPS module incompatibility problem**: swapping modules between NRPS assembly lines breaks the condensation interface geometry, destroying activity. CHIMERA's multi-scale hierarchical designer directly solves this:

- **Scale 3** (Module attention): Explicit module-module interface attention learns which linker geometries support productive condensation between adjacent modules
- **RFdiffusion linker design**: Generates new inter-module linkers conditioned on EvoFormer pair representations encoding co-evolutionary constraints between flanking domains
- **Pairwise interface scoring**: Each module pair gets an explicit compatibility score; incompatible combinations are rejected before PROTEUS

### NRPS-PKS Hybrid Modules

For Tier 2 PSC outputs (enhanced resolvins, macrolide variants, neurosteroids, kinase inhibitors), CHIMERA designs hybrid NRPS-PKS modules that combine:

- NRPS adenylation + PKS ketosynthase extensions
- Polyketide chain extension with amino acid incorporation
- Reductive loop domains (KR, DH, ER) for saturated/unsaturated polyketide products
- Full hybrid module backbone geometry via flow matching conditioned on both NRPS and PKS MSAs simultaneously

### PPTase Engineering

The phosphopantetheinyl transferase that activates all T-domains is itself a design target. CHIMERA's PPTase sub-campaign uses PROTEUS to evolve the native mammalian ACSF4 enzyme toward broader NRPS T-domain specificity — avoiding immunogenicity from bacterial Sfp while maintaining the post-translational modification that makes the entire assembly line functional.

### Full Assembly-Line Design

CHIMERA can design complete multi-module NRPS systems — not just individual domains:

- Multi-module polycistronic mRNA encoding (up to 5 NRPS modules + PPTase in a single construct)
- Substrate channeling architecture across the full assembly line
- Stoichiometry balancing via IRES strength calibration
- Icosahedral face compatibility at Scale 4 (ensuring each designed module integrates correctly into the PSC's 240nm icosahedral self-assembly)

---

## Architecture

```
Animal NRPS MSA (Stage 0: NCBI, antiSMASH, Suring et al. 2023)
including A, T, C, TE, E, Cy, Mt domain sequences
       │
 EvoFormer [FROZEN ~700M params]
 Captures co-evolutionary constraints across ALL domain types
       │
 TriangularPairUpdateConnector    ← TRAINABLE ~12M params total
 SubstratePocketConditioner       ←  substrate conditioning for A-domain
 EvolCrossAttentionConnector      ←  noise-adaptive evolutionary guidance
       │
 SE(3) OT-Flow Matching [FROZEN — RFdiffusion base]
 Bridge: bacterial NRPS backbone → mammalian-functional design
 Works on any domain type: A, T, C, TE, linker, insert
 20 NFE with RK4 (10x faster than DDPM)
       │
 Multi-Scale Hierarchical Sequence Designer
 Scale 1: Residue  — ProteinMPNN GNN (catalytic residue precision)
 Scale 2: Domain   — A/T/C/TE/linker domain attention
 Scale 3: Module   — module-module interface compatibility (solves 30yr problem)
 Scale 4: Assembly — icosahedral face constraint (PSC Layer 1 integration)
 Bidirectional: bottom-up and top-down message passing
       │
 Pareto Multi-Objective Head (5 objectives)
 F1: Evolutionary plausibility (PoET)
 F2: Structural stability (predicted pLDDT)
 F3: Mammalian expression efficiency (CodonOptimizer critic)
 F4: Substrate/product selectivity (domain function match)
 F5: Icosahedral assembly compatibility
       │
 Bayesian Uncertainty + Expected Improvement
 → Ranked Pareto frontier — optimal batch for PROTEUS
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

# Load model
model = CHIMERAv2.from_pretrained(
    flow_ckpt = "weights/rfdiffusion_base.pt",
    mpnn_ckpt = "weights/proteinmpnn_v48_020.pt",
)

# Example: design a complete A+T+C+TE module for phenylalanine activation
constraints = NRPSConstraints(
    # Stachelhaus selectivity code positions (A-domain substrate pocket)
    stachelhaus_positions = torch.tensor([235,236,239,278,299,301,322,330,517,518]),
    # All four domain boundaries in the module
    domain_boundaries     = torch.tensor([[[0,300],    # A-domain
                                           [300,400],  # T-domain
                                           [400,500],  # C-domain
                                           [500,580],  # TE-domain
                                           [580,600]]]), # linker
    module_boundaries     = torch.tensor([[[0,600],[0,0],[0,0],[0,0],[0,0]]]),
    icosahedral_face      = torch.tensor([7]),
    # PPant attachment serine on T-domain
    ppt_serine_position   = 519,
    fixed_mask            = None,
    hotspot_coords        = None,
    hotspot_indices       = None,
    target_substrate      = "PHE",
)

# Design: bridge bacterial PheA (1AMU) → mammalian-functional
results = model.design(
    nrps_msa        = msa_tokens,
    source_backbone = (bacterial_R, bacterial_t),   # from PDB 1AMU
    initial_pair_features = pair_features,
    target_substrate = "PHE",
    n_designs        = 500,
    n_pareto_samples = 50,
)

# Update from PROTEUS results (any domain campaign)
model.update_from_proteus(
    survivors = sequences_that_expressed_and_functioned,
    failures  = sequences_that_failed,
    msa       = msa_tokens,
    pair_features = pair_features,
)
```

---

## Companion Model: CodonOptimizer

Prepares **any** CHIMERA-designed NRPS sequence for mRNA delivery — A-domain, T-domain, C-domain, TE-domain, de novo inserts, or full modules.

```python
from chimera import optimize_nrps_for_mammalian_expression

# Works on any NRPS domain or full module sequence
result = optimize_nrps_for_mammalian_expression(
    aa_sequence = your_full_module_sequence,   # A+T+C+TE or any sub-domain
    beam_search = True,
    verbose     = True,
)

print(f"CAI: {result['cai']:.3f}")         # target ≥ 0.96
print(f"GC:  {result['gc_content']:.3f}")  # target 0.58-0.65
print(f"Bad motifs: {result['n_bad_motifs']}")  # target = 0
# result['dna_sequence'] → Trilink/Aldevron for mRNA synthesis with N1mΨ
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

### 3. Install external dependencies (production)

```bash
# OpenFold (EvoFormer backbone — covers all domain types)
git clone https://github.com/aqlaboratory/openfold.git && pip install -e openfold/

# RFdiffusion (backbone generation for all NRPS domains)
git clone https://github.com/RosettaCommons/RFdiffusion.git && pip install -e RFdiffusion/

# ProteinMPNN (sequence design for all domain types)
git clone https://github.com/dauparas/ProteinMPNN.git
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## PSC Pipeline

```
Stage 0:  Sequence retrieval (all domain types)
          NCBI: XP_018648700 (full NRPS), NIUQ01002120.1 (ACVS/IPNS/TE cluster)
          antiSMASH DB: pre-annotated A/T/C/TE/E/Cy/Mt domain clusters
          Suring et al. 2023 — 199 confirmed animal NRPS clusters
          MIBiG: 3000+ A-domain selectivity labels for training

Stage 1:  CHIMERA v2 computational design        ← THIS REPO
          Designs: A-domains, T-domains, C-domains, TE-domains,
                   E/Cy/Mt tailoring domains, de novo inserts,
                   module-module linkers, NRPS-PKS hybrids,
                   full multi-module assembly lines
          Output: 500 candidates → Pareto frontier → 50 for PROTEUS

Stage 1.5: PoET evolutionary plausibility scoring
           Scores any NRPS domain sequence against its family MSA

Stage 2:  CodonOptimizer mRNA preparation        ← THIS REPO
          Any domain or full module → CAI ≥ 0.96, GC 58-65%
          Outputs polycistronic mRNA encoding full NRPS assembly line

Stage 3:  PROTEUS directed evolution
          Domain-by-domain campaigns: A+T → C → TE → linkers → integration
          4-6 rounds per domain; CHIMERA updated via DPO after each round

Stage 4:  Polycistronic mRNA → intracellular icosahedral self-assembly
          Single LNP → 240nm PSC forms inside target cell
          Full NRPS assembly line operational: substrate → product

Stage 5:  Functional validation
          LC-MS: confirm product identity and yield
          PPant loading assay: confirm T-domain activation
          Sensor ring activation: confirm conditional synthesis
```

---

## PROTEUS Campaign Structure

Each domain type runs its own PROTEUS campaign, in this order:

```
Campaign 1 (A+T module):   5 rounds — substrate activation + PPant loading
Campaign 2 (C-domain):     3-5 rounds — bond formation + novel chemistry
Campaign 3 (TE-domain):    3 rounds — product release + cyclization geometry
Campaign 4 (De novo inserts): 5+ rounds — novel chemistry selection
Campaign 5 (Linkers):      3-4 rounds — module-module interface compatibility
Campaign 6 (Integration):  2-3 rounds — full assembly-line function
─────────────────────────────────────────────────────────────────
Total:                     ~21-30 rounds, parallel where possible
```

CHIMERA is updated via DPO after every campaign. Each round, the model gets better at predicting what works in mammalian cells for every domain type.

---

## Production Setup

The codebase uses stubs for the three pretrained backbones. See the production guide for Items 1-9:

| Item | Component | Time | GPU needed |
|------|-----------|------|-----------|
| 7, 8, 9 | Code fixes (random sampling, Hamming diversity, block counts) | 30 min | No |
| 4 | ESM-2 150M → CodonOptimizer | ~2 hr | 4GB |
| 3 | ProteinMPNN integration | ~3 hr | 2GB |
| 1 | EvoFormer / OpenFold | ~1 day | 6GB |
| 2 | RFdiffusion weight transfer | ~1 day | 4GB |

Hardware: **MacBook Pro M4 Max 48GB** (best) or RTX 4090 Laptop (16GB VRAM).
Cloud: **RunPod** (~$0.74/hr RTX 4090) or Google Colab Pro ($10/mo) while waiting.

---

## Roadmap

- [x] CHIMERA v1 — EvoFormer + SE3Denoiser + ProteinMPNN connectors
- [x] CHIMERA v2 — flow matching, multi-scale designer, RAG, DPO, Pareto, Bayesian uncertainty
- [x] Full NRPS machinery scope — A/T/C/TE/E/Cy/Mt + de novo inserts + linkers + hybrids
- [x] CodonOptimizer — autoregressive + expression critic
- [x] Test suite — shape/integration/math tests + CI/CD
- [ ] Production weight loading — Items 1-9 (in progress, collaborative)
- [ ] Stage 0 data pipeline — full NRPS domain sequence retrieval from all databases
- [ ] C-domain training data — MIBiG condensation domain annotations
- [ ] TE-domain cyclization training — cyclic vs linear product geometry labels
- [ ] De novo insert theozyme library — reaction geometry database
- [ ] First training run on Fath et al. codon optimization data
- [ ] First PROTEUS round — A+T domain campaign
- [ ] Wet lab validation of first CHIMERA-designed module

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Suring et al. 2023 *Genes* | Animal NRPS sequences (all domain types), Stage 0 templates |
| Fath et al. 2011 *PLoS ONE* | 9-parameter codon optimization for any NRPS sequence |
| Lipman et al. 2022 *ICLR* | OT-Flow Matching replacing DDPM |
| Yim et al. 2023 | SE(3) flow matching for protein backbones |
| Rafailov et al. 2023 *NeurIPS* | DPO — learning from PROTEUS preference pairs |
| Truong Jr & Bepler 2023 *NeurIPS* | PoET — evolutionary fitness for any NRPS domain |
| Watson et al. 2023 *Nature* | RFdiffusion — backbone generation |
| Jumper et al. 2021 *Nature* | AlphaFold2 EvoFormer |
| Dauparas et al. 2022 *Science* | ProteinMPNN — sequence design |
| Miller & Gulick 2016 *Methods Mol Biol* | NRPS structural biology (A/T/C/TE domain architecture) |
| Mootz et al. 2002 *PNAS* | NRPS module incompatibility (the 30-year problem) |
| Bozhüyük et al. 2018 *Nat Chem* | Modular NRPS recombination — closest experimental precedent |

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
<sub>PSC Engineering Pipeline · CHIMERA v2 · Theoretical Biomedical Engineering Framework · github.com/Izik-us/psc-chimera</sub>
</div>
