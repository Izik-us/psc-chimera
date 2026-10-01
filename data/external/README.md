# External datasets

## Kudla et al. 2009

- Study: Coding-sequence determinants of gene expression in *E. coli*
- DOI: https://doi.org/10.1126/science.1170160
- Supplement URL: https://pmc.ncbi.nlm.nih.gov/articles/instance/3902468/bin/NIHMS543681-supplement-Supplemental.doc
- Status: download endpoint returned an NCBI proof-of-work challenge in this environment; no records were converted.
- Intended use: weak/species-specific codon-context data after the supplementary table is obtained and manually verified.

Do not add external records to the merged training set unless the coding sequence, host, assay label, and provenance are all available.

## Pouyet et al. human codon-usage archive

- Archive: `data_HumanCodonUsage.zip` (Zenodo record 835063).
- Transcript/CDS sequence: Ensembl release 83 transcript identifiers in `human_genes_summary`; coding sequences are retrieved from Ensembl REST and cached locally.
- Default expression column: `Exp_Meiosis`, defined by the archive as the mean of female primordial germ-cell expression at 17 weeks and male pachytene-spermatocyte and round-spermatid expression (FPKM).
- Label semantics: endogenous human germline-expression proxy, not general mammalian expression and not a controlled synonymous-variant assay. Records retain `label_type=proxy`, phenotype components, source column, and normalization metadata.
- Intended use: separate proxy training/evaluation only; do not combine its metrics with measured synonymous-variant results without reporting label type.
