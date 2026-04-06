----------------------------
COSMIC Resistance Mutations
----------------------------
 A tab separated table listing the details of all mutations in COSMIC which are known to confer drug resistance. [ Cosmic_ResistanceMutations_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                                           Description
--------------------------------------------------------------------------------------------------------
[1:A]                 SAMPLE_NAME                  The sample name can be derived from a number of sources. In many cases it originates from the cell line name. Other sources include names assigned by the annotators, or an incremented number assigned during an anonymization process.
[2:B]                 COSMIC_SAMPLE_ID             A unique COSMIC sample identifier (COSS) is used to identify a sample. This identifier can be used to retrieve additional Sample information from the Cosmic_Sample file.
[3:C]                 GENE_SYMBOL                  The gene name for which the data has been curated in COSMIC. In most cases this is the accepted HGNC identifier.
[4:D]                 COSMIC_GENE_ID               A unique COSMIC gene identifier (COSG) is used to identify a gene within the file. This identifier can be used to retrieve additional Gene information from the Cosmic_Genes file.
[5:E]                 TRANSCRIPT_ACCESSION         Unique Ensembl Transcript identifier (ENST). For details see: https://www.ensembl.org/info/genome/stable_ids/index.html. This identifier can be used to retrieve additional Transcript information from the Cosmic_Transcripts file.
[6:F]                 CENSUS_GENE                  Is the gene in the Cancer Gene Census (Yes/No).
[7:G]                 DRUG_NAME                    The name of the drug which the mutation confers resistance to.
[8:H]                 DRUG_RESPONSE                The response for the drug which the mutation confers resistance to.
[9:I]                 GENOMIC_MUTATION_ID          Genomic mutation identifier (COSV) to indicate the definitive position of the variant on the genome. This identifier is trackable and stable between different versions of the release. This identifier can be used to retrieve additional legacy mutation ids from the Cosmic_MutationTracking file.
[10:J]                LEGACY_MUTATION_ID           Legacy mutation identifier (COSM) or (COSN) that will represent existing COSM or COSN mutation identifiers.
[11:K]                MUTATION_ID                  An internal mutation identifier to uniquely represent each mutation on a specific transcript on a given assembly build. This identifier can be used to retrieve additional legacy mutation ids from the Cosmic_MutationTracking file.
[12:L]                MUTATION_CDS                 The change that has occurred in the nucleotide sequence. Formatting is identical to the method used for the peptide sequence.
[13:M]                MUTATION_AA                  The change that has occurred in the peptide sequence. Formatting is based on the recommendations made by the Human Genome Variation Society. The description of each type can be found by following the link to the Mutation Overview page.
[14:N]                GENOMIC_WT_ALLELE            Genomic Wild type allele sequence.
[15:O]                GENOMIC_MUT_ALLELE           Genomic mutation allele sequence.
[16:P]                COSMIC_PHENOTYPE_ID          A unique COSMIC identifier (COSO) for the classification. This identifier can be used to retrieve tissue and histology information from the classification file.
[17:Q]                PUBMED_PMID                  The PUBMED ID for the paper that the sample was noted in, linking to pubmed to provide more details of the publication.
[18:R]                COSMIC_STUDY_ID              A unique COSMIC study identifier (COSU) is used to identify a study that have involved this sample.
[19:S]                MUTATION_ZYGOSITY            Information on whether the mutation was reported to be homozygous , heterozygous or unknown within the sample.
[20:T]                CHROMOSOME                   The chromosome location of a given resistance mutation (1-22, X, Y or MT).
[21:U]                GENOME_START                 The start coordinate of a given resistance mutation.
[22:V]                GENOME_STOP                  The end coordinate of a given resistance mutation.
[23:W]                STRAND                       Positive or negative (+/-).
[24:X]                HGVSP                        Human Genome Variation Society peptide syntax.
[25:Y]                HGVSC                        Human Genome Variation Society coding dna sequence syntax (CDS).
[26:Z]                HGVSG                        Human Genome Variation Society genomic syntax (3' shifted).
[27:AA]               MUTATION_SOMATIC_STATUS      Information on whether the sample was reported to be Confirmed somatic variant, Reported in another cancer sample as somatic or Variant of unknown origin:
                                                      * Reported in another cancer sample as somatic = when the mutation has been reported as somatic previously but not in current paper
                                                      * Confirmed somatic variant = if the mutation has been confirmed to be somatic in the experiment by sequencing both the tumour and a matched normal from the same patient
                                                      * Variant of unknown origin = When the tumour has been sequenced without a matched normal tissue from the same individual, the somatic status of the variant cannot be assessed
