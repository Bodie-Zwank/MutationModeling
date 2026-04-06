------------------
COSMIC Breakpoints
------------------
 All breakpoint data from the current release in a tab separated table. [ Cosmic_Breakpoints_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                             Description
--------------------------------------------------------------------------------------------------------
[1:A]                SAMPLE_NAME                          A sample is an instance of a portion of a tumour being examined for mutations. The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process. A number of samples can be taken from a single tumour and a number of tumours can be obtained from one individual. There can be multiple ids, if the same sample has been entered into the database multiple times from different papers.
[2:B]                COSMIC_SAMPLE_ID                     A unique COSMIC sample identifier (COSS) is used to identify a sample. This identifier can be used to retrieve additional sample information from the Cosmic_Sample file.
[3:C]                COSMIC_PHENOTYPE_ID                  A unique COSMIC identifier (COSO) for the classification. This identifier can be used to retrieve tissue and histology information from the classification file
[4:D]                COSMIC_STRUCTURAL_ID                 A COSMIC structural identifier (COST). This identifier can be used to retrieve structural variants from the Cosmic_StructuralVariants file
[5:E]                MUTATION_TYPE                        Types of mutation: Intra/Inter (chromosomal), tandem duplication, deletion, inversion, complex substitutions, complex amplicons.
[6:F]                CHROM_FROM                           The chromosome where the first variant/breakpoint occurs.
[7:G]                LOCATION_FROM_MIN                    The first position in breakpoint range.
[8:H]                LOCATION_FROM_MAX                    The last position in breakpoint range.
[9:I]                STRAND_FROM                          Positive or negative (+1/-1).
[10:J]               CHROM_TO                             The chromosome where the last variant/breakpoint occurs.
[11:K]               LOCATION_TO_MIN                      The first position in breakpoint range.
[12:L]               LOCATION_TO_MAX                      The last position in breakpoint range.
[13:M]               STRAND_TO                            Positive or negative (+1/-1).
[14:N]               NON_TEMPLATED_INS_SEQ                Non Templated Sequence (if any) which is inserted at the breakpoint. The sequence is not encoded.
[15:O]               PUBMED_PMID                          The PUBMED ID for the paper that the sample was noted in.
[16:P]               COSMIC_STUDY_ID                      A unique COSMIC study identifier (COSU) is used to identify a study that have involved this structural mutation.
