------------------------------------
Cell Lines Project Mutation Tracking
------------------------------------
 A tab separated table listing the mapping of all COSMIC's legacy mutations (COSMs or COSNs) to the new genomic identifiers (COSVs). This file also helps to identify the transcripts and the accession numbers on which the current mutation is annotated on, along with the mutation type. [ CellLinesProject_MutationTracking_v103_GRCh38.tsv.gz ]

  File Description

[column number:label] Heading                                           Description
--------------------------------------------------------------------------------------------------------
[1:A]                 COSMIC_GENE_ID               A unique COSMIC gene identifier (COSG) is used to identify a gene within the file. This identifier can be used to retrieve additional Gene information from the Cosmic_Genes file.
[2:B]                 TRANSCRIPT_ACCESSION         Unique Ensembl Transcript identifier (ENST). For details see: https://www.ensembl.org/info/genome/stable_ids/index.html. This identifier can be used to retrieve additional Transcript information from the Cosmic_Transcripts file.
[3:C]                 GENOMIC_MUTATION_ID          Genomic mutation identifier (COSV) to indicate the definitive position of the variant on the genome. This identifier is trackable and stable between different versions of the release.
[4:D]                 LEGACY_MUTATION_ID           Legacy mutation identifier (COSM) or (COSN) that will represent existing COSM or COSN mutation identifiers.
[5:E]                 MUTATION_ID                  An internal mutation identifier to uniquely represent each mutation on a specific transcript on a given assembly build.
[6:F]                 MUTATION_NC_ID               An internal mutation identifier to uniquely represent each non-coding mutation on a specific transcript on a given assembly build.
[7:G]                 MUTATION_TYPE                Type of mutation (coding or non-coding)
[8:H]                 IS_CANONICAL                 The Ensembl Canonical transcript is a single, representative transcript identified at every locus. For details see: https://www.ensembl.org/info/genome/genebuild/canonical.html
