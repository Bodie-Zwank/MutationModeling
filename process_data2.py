import pandas as pd
import numpy as np
import os

TSV_FILENAME = "Cosmic_MutantCensus_v103_GRCh38.tsv"
SAMPLE_DATA_PATH = "data/example_data"
REAL_DATA_PATH = "data/real_data"
PROCESSED_DATA_PATH = "data/processed_data/interarrivals.parquet"

PER_SAMPLE = True
USE_SAMPLE_DATA = False

# ── Filter to a specific chromosome arm ──────────────────────────────────────
# Set FILTER_CHROMOSOME to an int (1–22, 23=X, 24=Y) or None for all
# Set FILTER_ARM to 'p', 'q', or None for both arms
FILTER_CHROMOSOME = 7   # e.g. chromosome 7
FILTER_ARM = 'q' 

# ── GRCh38 centromere midpoints (bp) ─────────────────────────────────────────
# Source: UCSC centromere annotations for GRCh38.
# These are approximate midpoints; sufficient for p/q arm assignment.
CENTROMERE_POSITIONS = {
    1:  123400000,  2:  93900000,   3:  90900000,   4:  50400000,
    5:  48400000,   6:  61000000,   7:  59900000,   8:  45600000,
    9:  49000000,   10: 40200000,   11: 53700000,   12: 35800000,
    13: 17900000,   14: 17600000,   15: 19000000,   16: 36600000,
    17: 24000000,   18: 17200000,   19: 26500000,   20: 27500000,
    21: 13200000,   22: 14700000,
    23: 60600000,   # X
    24: 10400000,   # Y
    25: None,       # MT — mitochondrial, no centromere; treated as single arm
}

sample_data_file = os.path.join(SAMPLE_DATA_PATH, TSV_FILENAME)
real_data_file = os.path.join(REAL_DATA_PATH, TSV_FILENAME + ".gz")

if USE_SAMPLE_DATA:
    df = pd.read_csv(sample_data_file, sep='\t')
else:
    df = pd.read_csv(real_data_file, sep='\t', compression='gzip', low_memory=False)

df = df.dropna(subset=['CHROMOSOME', 'GENOME_START'])

chromosome_map = {'X': '23', 'Y': '24', 'MT': '25'}
df['CHROMOSOME'] = df['CHROMOSOME'].astype(str).replace(chromosome_map)
df['CHROMOSOME'] = df['CHROMOSOME'].astype(int)
df['GENOME_START'] = df['GENOME_START'].astype(int)

keep_cols = ['CHROMOSOME', 'GENOME_START', 'SAMPLE_NAME']
df = df[keep_cols].drop_duplicates()
df = df.drop_duplicates(subset=['SAMPLE_NAME', 'CHROMOSOME', 'GENOME_START'])

# ── Assign chromosome arm ─────────────────────────────────────────────────────
def assign_arm(row):
    centromere = CENTROMERE_POSITIONS.get(row['CHROMOSOME'])
    if centromere is None:   # MT has no arms
        return 'q'
    return 'p' if row['GENOME_START'] < centromere else 'q'

df['ARM'] = df.apply(assign_arm, axis=1)
df['CHROM_ARM'] = df['CHROMOSOME'].astype(str) + df['ARM']  # e.g. "1p", "1q"

# ── Optional filter ───────────────────────────────────────────────────────────
if FILTER_CHROMOSOME is not None:
    df = df[df['CHROMOSOME'] == FILTER_CHROMOSOME]
    print(f"Filtered to chromosome {FILTER_CHROMOSOME}: {len(df):,} mutations")
if FILTER_ARM is not None:
    df = df[df['ARM'] == FILTER_ARM]
    print(f"Filtered to {FILTER_ARM} arm: {len(df):,} mutations")

# ── Sort and compute interarrival distances ───────────────────────────────────
group_cols = (['SAMPLE_NAME', 'CHROM_ARM'] if PER_SAMPLE else ['CHROM_ARM'])

df = df.sort_values(group_cols + ['GENOME_START']).reset_index(drop=True)
df['interarrival_bp'] = df.groupby(group_cols)['GENOME_START'].diff()
df = df.dropna(subset=['interarrival_bp'])
df['interarrival_bp'] = df['interarrival_bp'].astype(int)

assert (df['interarrival_bp'] >= 0).all(), "Negative interarrivals found"

print(f"Total interarrival observations: {len(df):,}")
print(f"Mean interarrival distance:      {df['interarrival_bp'].mean():,.1f} bp")
print(df['interarrival_bp'].describe())

df.to_parquet(PROCESSED_DATA_PATH, index=False)
print(f"\nSaved to {PROCESSED_DATA_PATH}")