-------------------------
Cell Lines Project Sample
-------------------------
 All the features data related to a Cell Lines Project sample from the current release in a tab separated file. [ CellLinesProject_Sample_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                                           Description
--------------------------------------------------------------------------------------------------------
[1:A]                 COSMIC_SAMPLE_ID                                   A unique COSMIC sample identifier (COSS) is used to identify a sample. Other download files can be linked to this file using this identifier.
[2:B]                 SAMPLE_NAME                                        The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process.
[3:C]                 COSMIC_PHENOTYPE_ID                                A unique COSMIC identifier (COSO) for the classification. This identifier can be used to retrieve tissue and histology information from the classification file.
[4:D]                 TUMOUR_ID                                          A unique identifier used to identify a tumour. A number of tumours can be obtained from one individual and a number of samples can be taken from a single tumour.
[5:E]                 SAMPLE_TYPE                                        Describes the source (e.g. blood, surgery, cell line) and method of preparation (fixed, fresh/frozen) of the sample.
[6:F]                 INDIVIDUAL_ID                                      A unique ID to identify an individual
[7:G]                 WHOLE_GENOME_SCREEN                                Was the sample whole genome screened (y/n).
[8:H]                 WHOLE_EXOME_SCREEN                                 Was the sample whole exome sequenced (y/n).
[9:I]                 TARGETED_SCREEN                                    Was the sample targeted screened (y/n).
[10:J]                RNASEQ_SCREEN                                      Was the sample RNASeq screened (y/n).
[11:K]                REARRANGEMENT_SCREEN                               Was the sample rearrangement screened (y/n)
[12:L]                TUMOUR_SOURCE                                      Source of tumour tissue sample e.g. primary, metastasis.
[13:M]                NORMAL_TISSUE_TESTED                               If normal tissue from the same individual has been screened for mutations.
[14:N]                GENDER                                             Sex of individual.
[15:O]                AGE                                                Age (in years) of individual at diagnosis or at the earliest tumour presentation.
[16:P]                THERAPY_RELATIONSHIP                               Relates the time-point of tissue sampling to the drug therapy used to treat the tumour.
[17:Q]                SAMPLE_DIFFERENTIATOR                              Gives additional information if more than one sample (e.g. carcinomatous and sarcomatous components) from a tumour has been screened for mutations or if samples from a tumour were taken at different time points.
[18:R]                MUTATION_ALLELE_SPECIFICATION                      Where a publication has information on more than one mutation for one gene in a sample and reports whether or not the mutations occurred on the same or different chromosomes.
[19:S]                MSI                                                If microsatellite instability data is given in the publication per sample then High, Low, Stable/Low, MSI or Stable is reported in COSMIC. Unknown is the default.
[20:T]                AVERAGE_PLOIDY                                     The average ploidy of the sample, calculated from copy number data (where available).
[21:U]                SAMPLE_REMARK                                      Any additional sample information e.g. % mutant allele burden.
[22:V]                DRUG_RESPONSE                                      Clinical and in vitro responses to drugs (particularly targeted drugs). Phrasing based on RECIST guidelines.  Note that in COSMIC, SD (stable disease) and PD (progressive disease) = clinical primary non response.
[23:W]                GRADE                                              Grade of tumour. The phrase 'Some Grade data are given in publication' is used when publication reports grade data or when data hasn't been given per sample. More detailed data follow commonly used grading systems in tumours.
[24:X]                AGE_AT_TUMOUR_RECURRENCE                           Where both primary and recurrent tumour samples from an individual have been screened for mutations and the age (in years or months) of the patient at the time of the recurrence is different to that at diagnosis.
[25:Y]                STAGE                                              Stage of tumour. The phrase 'Some Stage data are given in publication' is used when publication reports stage data or when data hasn't been given per sample. More detailed data follow commonly used staging systems in tumours.
[26:Z]                CYTOGENETICS                                       Karyotype of the tumour.
[27:AA]               METASTATIC_SITE                                    Tissue site of any metastases identified in an individual.
[28:AB]               TUMOUR_REMARK                                      Any additional tumour information e.g. metachronous tumour.
[29:AC]               ETHNICITY                                          Ethnicity (e.g. Caucasian) of individual.
[30:AD]               ENVIRONMENTAL_VARIABLES                            Environmental variables to which an individual has been exposed (e.g. viral exposure, smoking status).
[31:AE]               GERMLINE_MUTATION                                  Gene name/mutation if a germline mutation as well as a somatic mutation has been detected in the same gene in the same tumour sample.
[32:AF]               THERAPY                                            Any significant treatment an individual has received prior to mutation screening.
[33:AG]               FAMILY                                             Any familial cancer history for an individual or familial relationships of individuals screened for mutations in the same publication.
[34:AH]               INDIVIDUAL_REMARK                                  Any additional individual information (e.g. age group, hereditary syndromes).
