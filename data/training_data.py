"""
CHIMERA + CodonOptimizer — Training Data Specification
=======================================================
Complete guide to what data to use, where to get it, how to format it,
and how to build the datasets for both models in the PSC pipeline.

Author: PSC Engineering Pipeline
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: CHIMERA TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

"""
CHIMERA needs three data types:
  (A) MSA sequences → EvoFormer input conditioning
  (B) Structural data → SE3Denoiser supervision (backbone ground truth)
  (C) Functional labels → PoET feedback + NRPS constraint enforcement

The connectors are fine-tuned; the pretrained backbones are frozen.
So the data volume needed is modest — ~500-2000 examples suffices for
fine-tuning 7M connector parameters.
"""

CHIMERA_DATA_SOURCES = {

    # ─── A. MSA Sequence Data (EvoFormer Input) ─────────────────────────────

    "animal_nrps_sequences": {
        "description": "Animal NRPS A-domain sequences — the core evolutionary family",
        "sources": [
            {
                "name": "Suring et al. 2023 Supplementary Table S4",
                "url": "https://www.mdpi.com/article/10.3390/genes14091741/s1",
                "content": "All 199 confirmed animal NRPS clusters with GenBank accessions",
                "how_to_get": "Download supplement from MDPI open-access page (CC-BY)",
                "format": "Excel/CSV with accession numbers → use efetch to pull sequences",
                "n_sequences": 199,
            },
            {
                "name": "NCBI Protein — direct accessions from Suring paper",
                "accessions": [
                    "XP_018648700",   # Schistosoma mansoni NRPS (Sm-NRPS seed)
                    "NIUQ01002120.1", # Plectus sambesii ACVS gene region
                    "LNIX01000001.1", # Folsomia candida NRPS scaffold
                ],
                "fetch_command": "efetch -db protein -id XP_018648700 -format fasta",
            },
            {
                "name": "Adineta vaga genome (79 NRPS clusters)",
                "publication": "Simion et al. 2021 Science Advances",
                "ncbi_bioproject": "PRJEB11619",
                "url": "https://www.ncbi.nlm.nih.gov/bioproject/PRJEB11619",
                "n_sequences": "~79 NRPS + 5 NRPS-PKS hybrid clusters",
            },
            {
                "name": "Rotaria macrura genome (richest source: 79 clusters)",
                "url": "search NCBI for Rotaria macrura NRPS",
                "n_sequences": "~79 clusters",
            },
            {
                "name": "C. elegans nemamide NRPS",
                "publication": "Shou et al. 2016 Nature Chemical Biology",
                "uniprot_search": "organism:'Caenorhabditis elegans' AND domain:'NRPS'",
                "n_sequences": 2,  # two NRPS genes producing nemamide
            },
        ],
        "retrieval_script": """
# Pull all animal NRPS A-domains in one pass
from Bio import Entrez, SeqIO
Entrez.email = "your@email.com"

# BLAST from Sm-NRPS seed, restrict to Metazoa
handle = Entrez.esearch(
    db="protein",
    term="XP_018648700[BLASTSEQ] AND Metazoa[Organism] AND nonribosomal[Title]",
    usehistory="y",
    retmax=500,
)
record = Entrez.read(handle)

# Fetch all hits
seqs = Entrez.efetch(
    db="protein",
    webenv=record["WebEnv"],
    query_key=record["QueryKey"],
    rettype="fasta",
    retmode="text",
)
with open("animal_nrps_seeds.fasta", "w") as f:
    f.write(seqs.read())
        """,
    },

    "bacterial_nrps_a_domains": {
        "description": "Bacterial NRPS A-domain sequences for merged MSA (needed for chimeric design)",
        "sources": [
            {
                "name": "MIBiG database (Minimum Information about a Biosynthetic Gene cluster)",
                "url": "https://mibig.secondarymetabolites.org/",
                "content": "All characterized NRPS BGCs with annotated A-domain specificities",
                "how_to_get": "Download full database as JSON, extract NRPS modules",
                "n_sequences": "~3000 A-domain sequences with known substrate",
            },
            {
                "name": "NCBI Reference Sequence — Landmark NRPS structures",
                "entries": [
                    ("PheA",    "AAC44543.1",  "Phe-activating A-domain, crystal structure 1AMU"),
                    ("EntF",    "P0ADI4",      "EntF enterobactin NRPS, broad specificity"),
                    ("TycA",    "Q9L9I8",      "Tyrocidine A synthetase, crystal 1MDB"),
                    ("HMWP2",   "P0C6E0",      "Yersiniabactin NRPS, crystal 2GRY"),
                    ("GrsA",    "P0C062",      "Gramicidin S synthetase A"),
                    ("SrfAA",   "P12944",      "Surfactin synthetase A-domain"),
                    ("NpuA",    "Q8YSL1",      "Nostoc NRPS — plant-adjacent, structurally interesting"),
                ],
                "pdb_structures": ["1AMU", "2GRY", "1MDB", "3E7W", "4ZXH", "5N9T"],
            },
        ],
        "a_domain_extraction": """
# Extract just A-domain sequences from full NRPS proteins using hmmscan
# Pfam model for NRPS A-domain: PF00501 (AMP-binding)
# Run: hmmscan --domtblout domains.txt Pfam-A.hmm all_nrps.fasta
# Then extract domain coordinates and slice sequences
        """,
    },

    # ─── B. Structural Data (SE3Denoiser Supervision) ───────────────────────

    "nrps_structures": {
        "description": "3D crystal/cryo-EM structures for backbone generation ground truth",
        "sources": [
            {
                "name": "PDB — all solved NRPS structures",
                "pdb_ids": [
                    "1AMU",  # PheA A-domain + phenylalanine (2.0Å) — THE landmark A-domain structure
                    "2GRY",  # HMWP2 A-domain + salicylate
                    "1MDB",  # TycA A-domain
                    "3E7W",  # EntF A-domain (open conformation)
                    "4ZXH",  # EntF A+T domain
                    "5N9T",  # SrfAA A+T, phosphopantetheinylated — critical for PPant arm geometry
                    "6MFX",  # GrsB C-domain — condensation domain structure
                    "6N8E",  # PvdQ A-domain + pyoverdine substrate
                    "7LY1",  # Recent full NRPS module with all three A+T+C domains
                    "7SX1",  # Adenylation domain + adenosine-5'-phosphosulfate (APS) intermediate
                ],
                "fetch_command": "for PDB in 1AMU 2GRY 1MDB 3E7W 4ZXH 5N9T; do wget https://files.rcsb.org/download/${PDB}.pdb; done",
                "n_structures": 10,
                "resolution_range": "1.8–3.2 Angstroms",
            },
            {
                "name": "AlphaFold Database — predicted structures for animal NRPSs",
                "url": "https://alphafold.ebi.ac.uk/",
                "search_strategy": "Pull AF2 predictions for all C. elegans and rotifer NRPS accessions",
                "quality_filter": "pLDDT > 70 for A-domain regions; pLDDT > 85 for catalytic pockets",
                "n_expected": "~50-80 high-confidence animal NRPS structure predictions",
                "command": """
# Batch download AF2 predictions for your accession list
import requests
accessions = ["Q9XTF5", "Q9XUL2"]  # C. elegans NRPS accessions
for acc in accessions:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
    response = requests.get(url)
    with open(f"{acc}_AF2.pdb", "wb") as f:
        f.write(response.content)
                """,
            },
            {
                "name": "Chimeric NRPS validation structures (to generate in Phase 0)",
                "description": "Express A-domain chimeras in E. coli, purify, solve by cryo-EM/X-ray",
                "purpose": "Ground truth for whether chimeric sequences fold as predicted",
                "timeline": "Generate during Phase 0 in parallel with PROTEUS campaigns",
                "n_needed": "20-50 chimera structures for meaningful connector training",
            },
        ],
        "preprocessing": """
# Standardize PDB files to extract backbone atoms only (N, CA, C, O)
# and align all structures by A-domain core to canonical 1AMU reference

from chimera.structure_utils import (
    extract_backbone_coords,   # returns (L, 4, 3) N/CA/C/O per residue
    align_to_reference,        # superpose onto reference structure
    compute_se3_frames,        # convert coords to rotation+translation per residue
)

structure = extract_backbone_coords("1AMU.pdb")
aligned   = align_to_reference(structure, reference_pdb="1AMU.pdb")
frames    = compute_se3_frames(aligned)  # (L, 4, 4) homogeneous transform
        """,
    },

    # ─── C. Functional / PoET Feedback Data ─────────────────────────────────

    "functional_labels": {
        "description": "Activity data that tells CHIMERA which designs actually work",
        "sources": [
            {
                "name": "Deep Mutational Scanning (DMS) — ProteinGym database",
                "url": "https://proteingym.org/",
                "relevant_datasets": [
                    "BLAT_ECOLX — beta-lactamase (AMP-binding enzyme, structurally related to A-domains)",
                    "AMIE_PSEAE — amidase (C-domain adjacent function)",
                ],
                "how_to_use": "Pre-train PoET's scoring head on ProteinGym DMS data before NRPS fine-tuning",
                "note": "No direct NRPS DMS available yet — these are structural proxies",
            },
            {
                "name": "MIBiG A-domain selectivity database",
                "url": "https://mibig.secondarymetabolites.org/",
                "content": "Substrate specificity labels for 3000+ A-domains (Stachelhaus code → amino acid)",
                "how_to_use": "Training labels for A-domain → substrate prediction task (validates CHIMERA's A-domain designs)",
            },
            {
                "name": "PROTEUS experimental results (active learning loop)",
                "description": "After each PROTEUS round, sequence → activity data feeds back into CHIMERA training",
                "format": "sequence (string) + binary activity label (expressed + functional in mammalian cells)",
                "n_per_round": "~100-500 sequences per PROTEUS round",
                "active_learning_loop": """
# After each PROTEUS round:
# 1. Surviving sequences are 'positive' examples
# 2. Failed sequences are 'negative' examples  
# 3. Add to CHIMERA connector fine-tuning set
# 4. Re-fine-tune connectors for 100 steps
# 5. Generate next round's library with updated CHIMERA

def update_chimera_from_proteus_round(
    surviving_seqs: list,   # sequences that passed PROTEUS selection
    failed_seqs: list,      # sequences that failed
    chimera_model,
    poet_model,
    n_update_steps: int = 100,
):
    # Surviving sequences are ground truth for 'correct' CHIMERA outputs
    # Update connector weights to make CHIMERA generate surviving-like sequences
    optimizer = torch.optim.AdamW(
        [p for p in chimera_model.parameters() if p.requires_grad], lr=1e-5
    )
    for step in range(n_update_steps):
        for seq in surviving_seqs:
            # Compute CHIMERA's probability of generating this sequence
            logits = chimera_model(msa=nrps_msa, ...)['sequences']
            target = tokenize(seq)
            loss = F.cross_entropy(logits.view(-1,20), target.view(-1))
            loss.backward()
        optimizer.step()
                """,
            },
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: CODON OPTIMIZER TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

"""
The CodonOptimizer needs three data types:
  (X) Paired sequences: wildtype CDS → optimized CDS (supervised learning)
  (Y) Expression data: DNA sequence → measured protein yield (critic training)
  (Z) Motif annotation: sequences labeled for bad motif presence (auxiliary)
"""

CODON_OPTIMIZER_DATA_SOURCES = {

    # ─── X. Paired Sequence Data (Core Supervised Training) ─────────────────

    "paired_sequences": {
        "primary": {
            "name": "Fath et al. 2011 — 50 paired wildtype vs optimized sequences",
            "paper": "PLoS ONE 6(3):e17596",
            "url": "https://doi.org/10.1371/journal.pone.0017596",
            "supplement": "File S1 — all wildtype and sequence-optimized constructs (FASTA format)",
            "how_to_get": "Download directly from PLoS ONE open-access page (CC-BY)",
            "content": """
50 pairs covering five protein classes:
  - Transcription factors (TF): TFIIB, CREB1, ZNRD1, Lef1, EMG1
  - Ribosomal proteins (RB): SMARCD1
  - Protein kinases (PK): PIM1, PIM2, PIM3, Trb1, JAK2, HCK, LCK, FLT1,
                           MAP2K1, PLK3, Erk2, p38a, JNK1, JNK3, CLK3, CK1a
  - Membrane proteins (MP): GAT1, Serotonin-TP, TLR10, SLC39A1, KCNJ1, AQP5,
                             CAV1, TAP1, TAP2, LAMP1, LAMP2, LAMP3, CLN3, LEMD3,
                             OPRM1, TAS2R10, VKORC1, NGFR
  - Immunomodulators (IM): IL-15, Mip1a, IL-2, IFNg, IL-24, IL-10, RANTES, IFNa
  - Other: BIRC5, CDC2
            """,
            "n_pairs": 50,
            "gold_standard": True,
        },

        "supplementary": [
            {
                "name": "HIV-1 Rev-independent expression studies",
                "papers": [
                    "Graf et al. 2000 (J Virol 74:10822) — HIV-1 gag/pol codon optimization",
                    "Schneider et al. 1997 (J Virol 71:4892) — HIV-1 inhibitory element removal",
                    "Nguyen et al. 2004 (Virology 319:163) — HIV-1 vpu/vif optimization",
                ],
                "content": "Wildtype HIV-1 sequences + codon-optimized versions for human expression",
                "n_pairs": "~20 unique pairs",
                "note": "HIV sequences are AT-rich (bacterial-like CAI), excellent training for NRPS optimization",
            },
            {
                "name": "COVID-19 mRNA vaccine sequences (publicly disclosed)",
                "pairs": [
                    {
                        "name": "BNT162b2 (Pfizer-BioNTech)",
                        "wildtype": "SARS-CoV-2 spike protein gene (NC_045512.2 positions 21563-25384)",
                        "optimized": "Published in Vogel et al. 2021 Molecular Therapy; full optimized sequence disclosed in Lamb 2021 Current Biology",
                        "modifications": "Codon optimized + P2 stabilizing mutations + mRNA modifications",
                    },
                    {
                        "name": "mRNA-1273 (Moderna)",
                        "wildtype": "SARS-CoV-2 spike gene",
                        "optimized": "Published in Jackson et al. 2020 NEJM supplementary",
                    },
                ],
                "n_pairs": 2,
                "value": "Large gene (3822 nt spike), real mRNA optimization, N1mΨ context",
            },
            {
                "name": "Gene therapy expression cassette database",
                "source": "ClinicalTrials.gov + FDA biologics license applications",
                "examples": [
                    "AAV-RPE65 (Luxturna): wildtype RPE65 vs codon-optimized for photoreceptors",
                    "SMA therapy (Zolgensma): SMN1 codon-optimized for CNS",
                    "Hemophilia B (Etranacogene dezaparvovec): codon-optimized FIX",
                ],
                "how_to_get": "European Medicines Agency (EMA) assessment reports list optimized sequences",
                "n_pairs": "~15-20",
            },
            {
                "name": "Kosovac et al. 2010 — EPO electrogene transfer study",
                "paper": "Gene Therapy (2010) pp 1-10 — cited as [50] in Fath et al.",
                "content": "Codon-optimized EPO gene showing 3-4x expression increase in skeletal muscle",
                "value": "In vivo expression data (not just cell culture)",
                "n_pairs": 1,
            },
            {
                "name": "NCBI RefSeq — automated codon usage pairs",
                "description": """
For every human gene in NCBI RefSeq with known expression data:
- Collect the native CDS (wildtype)
- Collect any published recombinant/optimized versions from the literature
- Use ProteomicsDB or PaxDb for expression abundance as proxy label
                """,
                "search_command": """
# Find codon-optimized sequences in NCBI literature
from Bio import Entrez
Entrez.email = "your@email.com"
handle = Entrez.esearch(
    db="nucleotide",
    term="codon optimized[Title/Abstract] AND human[Organism] AND 2015:2024[PDAT]",
    retmax=5000,
)
                """,
                "n_expected": "~500-2000 additional pairs from automated search",
            },
        ],

        "nrps_specific": {
            "name": "NRPS heterologous expression studies",
            "papers": [
                "Baltz 2011 (J Ind Microbiol Biotechnol) — Review of NRPS heterologous expression",
                "Ongley et al. 2013 (ACS Chem Biol) — NRPS refactoring in E. coli",
                "Bozhuyuk et al. 2018 (Nat Chem) — NRPS module swapping (chimeras)",
                "Yuzawa et al. 2018 (Nat Chem Biol) — Modular PKS engineering precedent",
            ],
            "note": """
No published mammalian NRPS expression papers as of training cutoff.
This is EXACTLY why the PROTEUS campaign is needed.
After Phase 0 PROTEUS rounds, the successful sequences become the first
mammalian NRPS expression training pairs — pioneering data.
Use them to fine-tune the CodonOptimizer for NRPS-specific codon preferences.
            """,
            "n_pairs": 0,  # None exist yet — PSC project will create these
        },
    },

    # ─── Y. Expression Data (Critic / Expression Predictor Training) ─────────

    "expression_data": {
        "description": "Paired (DNA sequence → measured protein yield) for critic training",
        "sources": [
            {
                "name": "ProteomicsDB — protein expression across human tissues",
                "url": "https://www.proteomicsdb.org/",
                "content": "Protein abundance (in nM) for 12,000+ human proteins across 64 tissues",
                "how_to_use": """
1. Download ProteomicsDB protein abundance data (free API)
2. For each protein, get its CDS from RefSeq
3. Pair: (CDS sequence, protein abundance in HEK293 cells)
4. This gives ~12,000 (DNA, expression) pairs for critic pre-training
                """,
                "note": "These are ENDOGENOUS expression levels, not recombinant expression efficiency",
                "n_examples": 12000,
            },
            {
                "name": "Codon usage vs expression rate — tissue-specific data",
                "paper": "Dittmar et al. 2006 PLoS Genetics — tissue-specific tRNA expression",
                "content": "Correlation between codon usage and expression level in specific cell types",
                "use": "Pre-compute expected expression efficiency for any CDS in HEK293 based on tRNA availability",
            },
            {
                "name": "High-throughput codon expression studies",
                "papers": [
                    "Kudla et al. 2009 Science 324:255 — 154 GFP variants, codon usage vs expression",
                    "Goodman et al. 2013 Science 342:475 — 14,000 variants, codon pair bias",
                    "Cambray et al. 2018 Nature Comm — 20,000 5'UTR variants vs expression",
                ],
                "content": "High-throughput data linking specific sequence features to measured fluorescence",
                "n_examples": "~35,000 across all three studies",
                "value": "BEST training data for critic — direct (sequence → expression) with no confounds",
            },
            {
                "name": "Fath et al. 2011 expression ratios (Table 1)",
                "content": "50 optimized/wildtype expression ratios (1:1.04 to 1:14.73)",
                "use": "Direct training labels for relative expression prediction",
                "format": "wildtype_seq, optimized_seq, expression_ratio",
                "n_examples": 50,
            },
        ],
        "critic_pretraining": """
# Three-phase critic training:

# Phase 1: Pre-train on large codon-expression datasets (Kudla, Goodman, Cambray)
# Target: predict relative GFP fluorescence from CDS sequence
# n_examples: ~35,000
# Architecture: CNN + Transformer hybrid (captures both local motifs + global GC)

# Phase 2: Transfer to human protein expression (ProteomicsDB)
# Fine-tune on (human CDS, HEK293 expression level) pairs
# n_examples: ~12,000
# This shifts the critic to understand mammalian expression specifically

# Phase 3: Fine-tune on Fath et al. wildtype vs optimized pairs
# The critic learns to distinguish optimized from wildtype sequences
# n_examples: 50 (but gold standard — high quality)
        """,
    },

    # ─── Z. Motif Annotation Data (Auxiliary Tasks) ─────────────────────────

    "motif_annotation": {
        "description": "Labeled sequences for bad motif detection (auxiliary training signal)",
        "motifs_to_annotate": {
            "cryptic_splice_sites": {
                "tool": "MaxEntScan (http://genes.mit.edu/burgelab/maxent/)",
                "description": "Score-based splice site predictor",
                "labeling": "Run MaxEntScan on every window of every CDS; label windows with score > 3.0",
            },
            "poly_A_signals": {
                "patterns": ["AATAAA", "ATTAAA", "AGTAAA", "TATAAA"],
                "labeling": "Exact string search; label positions of any match",
            },
            "AU_rich_elements": {
                "patterns": ["AUUUA", "UUAUUUAUUU", "AUUUAUUUAUUU"],
                "description": "mRNA destabilizing elements in 3'UTR",
                "labeling": "Search in RNA sequence (T→U)",
            },
            "UpA_dinucleotides": {
                "description": "XA dinucleotides in RNA (preferred RNase targets)",
                "labeling": "Count [ACGU]A dinucleotides per codon window",
            },
            "direct_repeats": {
                "tool": "RepFind or self-BLAST",
                "threshold": "repeats > 8 nt within same sequence",
            },
        },
        "annotation_pipeline": """
# Annotate any CDS sequence with all 9 Fath motif classes
def annotate_sequence(dna_seq: str) -> dict:
    rna_seq = dna_seq.replace('T', 'U')
    return {
        'cai':          compute_cai(dna_seq, HUMAN_CODON_TABLE),
        'gc_content':   gc_fraction(dna_seq),
        'n_UpA':        count_UpA(rna_seq),
        'n_CpG':        count_CpG(dna_seq),
        'n_ARE':        count_AU_rich_elements(rna_seq),
        'n_splice':     maxent_splice_score(dna_seq),
        'n_polyA':      count_polyA_signals(dna_seq),
        'n_repeats':    count_direct_repeats(dna_seq),
        'mfe':          rnafold_mfe(rna_seq),  # requires ViennaRNA
        'n_IRES':       count_IRES_motifs(dna_seq),
    }
        """,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: DATA SPLITS AND CURRICULUM
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CURRICULUM = {

    "chimera_training_schedule": """
CHIMERA connector fine-tuning curriculum:

Step 1 — Structural alignment (100 epochs):
    Data:     PDB NRPS structures (10 solved structures + AF2 predictions)
    Loss:     L_struct only (FAPE loss on backbone coordinates)
    Purpose:  Teach pair_connector to generate NRPS-compatible backbone geometry
    LR:       1e-4 with warmup

Step 2 — Sequence recovery (200 epochs):
    Data:     All animal + bacterial NRPS sequences with AF2 structure predictions
    Loss:     L_struct + L_seq (add sequence cross-entropy)
    Purpose:  Teach node_connector to inform sequence design from evolution
    LR:       1e-4 with cosine decay

Step 3 — Evolutionary consistency (100 epochs):
    Data:     Animal NRPS MSAs, PoET scoring of generated sequences
    Loss:     L_struct + L_seq + L_evol (add PoET feedback)
    Purpose:  Pull generated sequences toward high-PoET-score region
    LR:       1e-5

Step 4 — NRPS constraint enforcement (50 epochs):
    Data:     Annotated A-domain sequences with Stachelhaus code labels
    Loss:     Full loss + L_nrps (enforce catalytic residue conservation)
    Purpose:  Ensure designs respect NRPS domain constraints
    LR:       1e-5

Step 5 — Active learning from PROTEUS (ongoing):
    Data:     Each PROTEUS round adds 100-500 (sequence, activity) pairs
    Loss:     Full loss, weight positive examples more heavily
    Purpose:  Ground CHIMERA in what actually works in mammalian cells
    Frequency: Re-fine-tune for 50-100 steps after each PROTEUS round
    """,

    "codon_optimizer_training_schedule": """
CodonOptimizer training curriculum:

Phase A — Critic pre-training (train ExpressionPredictor separately):
    Data:     Kudla 2009 + Goodman 2013 + Cambray 2018 (35,000 examples)
    Task:     Predict GFP fluorescence from CDS sequence
    Epochs:   100
    LR:       3e-4

Phase B — Transfer critic to human expression:
    Data:     ProteomicsDB (12,000 human CDS, HEK293 protein abundance)
    Task:     Predict log(protein abundance) in HEK293T
    Epochs:   50
    LR:       1e-4 (fine-tune from Phase A)

Phase C — Supervised seq2seq training:
    Data:     All paired sequences (Fath 50 + HIV pairs + vaccine pairs ≈ 100 pairs)
    Loss:     L_CE + λ_cai * L_CAI + λ_gc * L_GC + λ_expr * L_expr_critic
    Epochs:   500 (small dataset → many epochs needed)
    LR:       1e-4 with warmup + cosine decay
    Augment:  Random synonymous back-mutations to prevent overfitting

Phase D — NRPS-specific fine-tuning (after Phase 0 PROTEUS):
    Data:     First mammalian-expressed NRPS module sequences (novel training data)
    Loss:     Same as Phase C
    Epochs:   50-100 per PROTEUS round
    Purpose:  Specialize optimizer for NRPS codon context (long GC-rich domains)
    """,
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: DATA GAPS AND HOW TO FILL THEM
# ═══════════════════════════════════════════════════════════════════════════════

DATA_GAPS = {
    "no_mammalian_nrps_expression_data": """
CRITICAL GAP: No published data exists for NRPS module expression in mammalian cells.
This is the PSC project's founding problem.

MITIGATION:
  1. Pre-train CodonOptimizer on analogous large bacterial proteins expressed in mammals:
     - Sfp (PPTase, 245 aa, bacterial → mammalian expression published)
     - EntD (PPTase homolog, similar size)
     - These have similar codon bias challenges to NRPS modules
  2. Use PKS systems as structural proxies (similar megasynthetase architecture)
  3. After Phase 0 PROTEUS: PSC project generates this data first in the field
     → publish these pairs, enabling the community and improving the model
    """,

    "animal_nrps_structures": """
CRITICAL GAP: No crystal structures exist for animal NRPS enzymes.
All structural data comes from bacterial/fungal NRPSs.

MITIGATION:
  1. AF2 predictions for all 199 animal NRPS sequences — adequate for sequence design,
     insufficient for high-resolution binding pocket engineering
  2. Phase 0 milestone: obtain cryo-EM structure of C. elegans nemamide NRPS
     (expected 3.0-3.5 Å resolution if sample quality is good)
  3. This structure becomes the centerpiece of CHIMERA's structural training data
    """,

    "selectivity_code_chimera_data": """
MODERATE GAP: Limited data on bacterial A-domain selectivity codes transplanted
into animal A-domain scaffolds.

MITIGATION:
  1. Generate 50-100 chimeric sequences computationally (CHIMERA Stage 1.5)
  2. Express 20-30 highest-PoET-scoring chimeras in E. coli for rapid assessment
     (not mammalian — just to confirm folding and substrate activation)
  3. Feed results back to CHIMERA as supervised labels
  4. Only take validated chimeric backbones into PROTEUS mammalian campaign
    """,
}

if __name__ == '__main__':
    print("CHIMERA + CodonOptimizer Training Data Summary")
    print("=" * 60)

    print("\n[CHIMERA]")
    print(f"  Animal NRPS sequences:   {CHIMERA_DATA_SOURCES['animal_nrps_sequences']['sources'][0]['n_sequences']} confirmed + genome assemblies")
    print(f"  PDB structures:          {len(CHIMERA_DATA_SOURCES['nrps_structures']['sources'][0]['pdb_ids'])} solved + AF2 predictions")
    print(f"  Functional labels:       MIBiG selectivity database + PROTEUS active learning")

    print("\n[CodonOptimizer]")
    print(f"  Gold standard pairs:     {CODON_OPTIMIZER_DATA_SOURCES['paired_sequences']['primary']['n_pairs']} (Fath et al.)")
    print(f"  Extended pairs:          ~150 total from HIV, vaccine, gene therapy sources")
    print(f"  Expression critic data:  ~47,000 (sequence → expression) examples")
    print(f"  Motif annotation:        Any CDS automatically annotated via annotation_pipeline")

    print("\n[Key Gap]")
    print("  No mammalian NRPS expression data exists anywhere.")
    print("  Phase 0 PROTEUS generates this — PSC project creates its own training data.")

    print("\n[Data Retrieval]")
    print("  Run: python training_data.py --download-all")
    print("  This executes all fetch commands above and builds the dataset.")
