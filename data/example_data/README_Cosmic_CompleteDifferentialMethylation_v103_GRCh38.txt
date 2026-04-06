------------------
COSMIC Methylation
------------------
 TCGA Level 3 methylation data from the ICGC portal for the current release in a tab separated table. More information on the methylation data is available from https://cancer.sanger.ac.uk/cosmic/analyses. [ Cosmic_CompleteDifferentialMethylation_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                            Description
--------------------------------------------------------------------------------------------------------
[1:A]                 COSMIC_STUDY_ID                    A unique COSMIC study identifier (COSU) is used to identify a study that have involved this methylation data.
[2:B]                 COSMIC_SAMPLE_ID                   A unique COSMIC sample identifier (COSS) is used to identify a sample. This identifier can be used to retrieve additional Sample information from the Cosmic_Sample file.
[3:C]                 SAMPLE_NAME                        The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process.
[4:D]                 FRAGMENT_ID                        The unique probe Id for a specific CpG.
[5:E]                 CHROMOSOME                         The chromosome location of the probe (1-22, X or Y).
[6:F]                 POSITION                           The genome location of the CpG targeted by the probe (1-based coordinates).
[7:G]                 STRAND                             Positive or negative (+1/-1).
[8:H]                 GENE_SYMBOL                        The gene name (if the probe falls within the coding region of a COSMIC gene) or the probe annotation as described by Illumina.
[9:I]                 METHYLATION                        The methylation level; H (High, beta-value >0.8) or L (Low, beta-value < 0.2).
[10:J]                AVG_BETA_VALUE_NORMAL              The average beta-value across the normal population. The beta-value of the tumour must differ from this value by >0.5 to be considered a variant.
[11:K]                BETA_VALUE                         The beta-value for the probe in the tumour sample. Only values >0.8 (High) or <0.2 (Low) are included.
[12:L]                TWO_SIDED_P_VALUE                  The two sided p-value.
[13:M]                COSMIC_PHENOTYPE_ID                A unique COSMIC identifier (COSO) for the classification. This identifier can be used to retrieve tissue and histology information from the classification file.
