import pandas as pd
import numpy as np
import os

"""
this file contains:
- GENOMIC_START (optionally GENOME_STOP) to get position of mutation
- SAMPLE_NAME to identify the person with the mutation
- COSMIC SAMPLE ID and TRANSCRIPT_ACCESSION to optionally retrieve additional data on gene
- CHROMOSOME to filter by chromosome if desired
"""
TSV_FILENAME = "Cosmic_MutantCensus_v103_GRCh38.tsv"
SAMPLE_DATA_PATH = "data/example_data"
# number of base pairs to be put in each bin
BIN_SIZE = 1_000_000
# number of base pairs in human genome
GENOME_SIZE = 3_000_000_000

sample_data_file = os.path.join(SAMPLE_DATA_PATH, TSV_FILENAME)

df = pd.read_csv(sample_data_file, sep='\t')
df['bin'] = (df['GENOME_START'] // BIN_SIZE).astype(int)

bin_counts = np.array([0 for _ in range(GENOME_SIZE // BIN_SIZE)])


for mutation_bin in df['bin']:
    bin_counts[mutation_bin] += 1

print(bin_counts)
