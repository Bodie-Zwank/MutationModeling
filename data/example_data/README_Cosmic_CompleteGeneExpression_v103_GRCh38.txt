-------------------------------
COSMIC Complete Gene Expression
-------------------------------
 All gene expression level 3 data from the TCGA portal for the current release in a tab separated table. Please note : The platform codes currently used to produce the COSMIC gene expression values are: IlluminaGA_RNASeqV2, IlluminaHiSeq_RNASeqV2, AgilentG4502A_07_2, AgilentG4502A_07_3. For more information on the gene expression data, please see  https://cancer.sanger.ac.uk/cosmic/analyses. [ Cosmic_CompleteGeneExpression_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                            Description
--------------------------------------------------------------------------------------------------------
[1:A]                 COSMIC_SAMPLE_ID                   A unique COSMIC sample identifier (COSS) is used to identify a sample. This identifier can be used to retrieve additional Sample information from the Cosmic_Sample file.
[2:B]                 SAMPLE_NAME                        The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process.
[3:C]                 COSMIC_GENE_ID                     A unique COSMIC gene identifier (COSG) is used to identify a gene within the file. This identifier can be used to retrieve additional Gene information from the Cosmic_Genes file.
[4:D]                 GENE_SYMBOL                        The gene name for which the data has been curated in COSMIC. In most cases this is the accepted HGNC identifier.
[5:E]                 REGULATION                         The regulation can be over or under depending on the scores from different platforms if they are above or below the threshold.
[6:F]                 Z_SCORE                            z_score serves as an indicative score taken from the gene_expression from different platforms in order of preference: IlluminaHiSeq_RNASeqV2, IlluminaGA_RNASeqV2, AgilentG4502A_07_3.
[7:G]                 COSMIC_STUDY_ID                    A unique COSMIC study identifier (COSU) is used to identify a study that have involved this gene expression data.
