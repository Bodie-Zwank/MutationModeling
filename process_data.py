import pandas as pd
import numpy as np
import os
from itertools import product

"""
this file contains:
- GENOMIC_START (optionally GENOME_STOP) to get position of mutation
- SAMPLE_NAME to identify the person with the mutation
- COSMIC SAMPLE ID and TRANSCRIPT_ACCESSION to optionally retrieve additional data on gene
- CHROMOSOME to filter by chromosome if desired
"""
TSV_FILENAME = "Cosmic_MutantCensus_v103_GRCh38.tsv"
SAMPLE_DATA_PATH = "data/example_data"
REAL_DATA_PATH = "data/real_data"
PROCESSED_DATA_PATH = "data/processed_data/processed_data.parquet"
# number of base pairs to be put in each bin
BIN_SIZE = 1_000_000
# number of base pairs in human genome
GENOME_SIZE = 3_000_000_000

USE_SAMPLE_DATA = False

sample_data_file = os.path.join(SAMPLE_DATA_PATH, TSV_FILENAME)
real_data_file = os.path.join(REAL_DATA_PATH, TSV_FILENAME + ".gz")

if USE_SAMPLE_DATA:
    df = pd.read_csv(sample_data_file, sep='\t')
else:
    df = pd.read_csv(real_data_file, sep='\t', compression='gzip', low_memory=False)

# clean out unhelpful data points
df = df.dropna(subset=['CHROMOSOME', 'GENOME_START'])

# clean up chromosomes so all can be cast to int (including 'X', 'Y', and 'MT')
chromosome_map = {'X': '23', 'Y': '24', 'MT': '25'}
df['CHROMOSOME'] = df['CHROMOSOME'].astype(str).replace(chromosome_map)
df['CHROMOSOME'] = df['CHROMOSOME'].astype(int)


df['bin'] = (df['GENOME_START'] // BIN_SIZE).astype(int)

bin_counts = df.groupby(['CHROMOSOME', 'bin']).size().reset_index(name='total_mutations')
individuals = df.groupby(['CHROMOSOME', 'bin'])['SAMPLE_NAME'].nunique().reset_index(name='individuals_in_bin')

# combine number of individuals in each chromosome-bin pair with original bin dataframe
bin_counts = bin_counts.merge(individuals, on=['CHROMOSOME', 'bin'])
#bin_counts = bin_counts.groupby(['bin', 'chromosome']).sum().reset_index()

bin_counts['mutation_rates'] = bin_counts['total_mutations'] / bin_counts['individuals_in_bin']

# create bins with 0 mutations for all the indices along the genome with no marked mutations
# we do this for clarity on where the mutations are occurring, and to eliminate bias toward mutated regions
chromosomes = range(1, 26)  # 1-22 autosomes, 23=X, 24=Y, 25=MT
bins = range(GENOME_SIZE // BIN_SIZE)
all_bins = pd.DataFrame(list(product(chromosomes, bins)), columns=['CHROMOSOME', 'bin'])

bin_counts = all_bins.merge(bin_counts, on=['CHROMOSOME', 'bin'], how='left').fillna(0)

bin_counts.to_parquet(PROCESSED_DATA_PATH, index=False)
