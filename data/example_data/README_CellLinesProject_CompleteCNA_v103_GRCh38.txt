---------------------------------------
Cell Lines Project Copy Number Variants
---------------------------------------
 All copy number variants from the current release for each cell line identified by PICNIC analysis of the Affymetrix SNP6.0 array data in a tab separated table. For more information on copy number data, please see https://cancer.sanger.ac.uk/cell_lines/analyses. [ CellLinesProject_CompleteCNA_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                            Description
--------------------------------------------------------------------------------------------------------
[1:A]                 COSMIC_CNV_ID                      A Copy number variant identifier (COSCNV) is used to identify the copy number variants within the file.
[2:B]                 COSMIC_GENE_ID                     A unique COSMIC gene identifier (COSG) is used to identify a gene within the file. This identifier can be used to retrieve additional Gene information from the Cosmic_Genes file.
[3:C]                 GENE_SYMBOL                        The gene name for which the data has been curated in COSMIC. In most cases this is the accepted HGNC identifier.
[4:D]                 COSMIC_SAMPLE_ID                   A unique COSMIC sample identifier (COSS) is used to identify a sample. This identifier can be used to retrieve additional Sample information from the CellLinesProject_Sample file.
[5:E]                 SAMPLE_NAME                        The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process..
[6:F]                 COSMIC_PHENOTYPE_ID                A unique COSMIC identifier (COSO) for the classification. This identifier can be used to retrieve tissue and histology information from the classification file.
[7:G]                 TOTAL_CN                           The sum of the major and minor allele counts e.g. if ABB, total copy number = 3.
[8:H]                 MINOR_ALLELE                       The number of copies of the least frequent allele e.g. if ABB, minor allele = A ( 1 copy) and major allele = B ( 2 copies).
[9:I]                 MUT_TYPE                           Defined as Gain or Loss. For ICGC samples; as defined in the original data. For TCGA samples reanalysed with ASCAT -
                                                            * LOSS = average genome ploidy <= 2.7 AND total copy number = 0 OR average genome ploidy > 2.7 AND total copy number < ( average genome ploidy - 2.7 )
                                                            * GAIN = average genome ploidy <= 2.7 AND total copy number >= 5 OR average genome ploidy > 2.7  AND total copy number >= 9
[10:J]                COSMIC_STUDY_ID                    A unique COSMIC study identifier (COSU) is used to identify a study that have involved this copy number variation.
[11:K]                CHROMOSOME                         The chromosome location of a given copy number variant (1-22, X, Y or MT)
[12:L]                GENOME_START                       The start coordinate of a given copy number variant
[13:M]                GENOME_STOP                        The end coordinate of a given copy number variant
