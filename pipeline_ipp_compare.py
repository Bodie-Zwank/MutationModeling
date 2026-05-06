"""Real vs IPP-simulated interarrival distances.

Builds the empirical rate map λ̂(x) from COSMIC, simulates per-sample
inhomogeneous Poisson processes against that rate map, then overlays the
real and simulated interarrival distributions.
"""
import os

from src.arms import assign_arms
from src.interarrivals import compute_interarrivals
from src.load_cosmic import load_cosmic
from src.rate import compute_rate_map
from src.simulate_ipp import simulate_ipp_interarrivals
from src.plots import plot_real_vs_simulated_interarrivals

COSMIC_FILENAME = 'Cosmic_MutantCensus_v103_GRCh38.tsv'

SAMPLE_DATA_DIR = 'data/example_data'
REAL_DATA_DIR = 'data/real_data'
PROCESSED_DIR = 'data/processed_data'
PLOTS_DIR = os.path.join(PROCESSED_DIR, 'plots')

USE_SAMPLE_DATA = False
BIN_SIZE_BP = 1_000_000
N_REPLICATES = 1
SEED = 0

REAL_INTERARRIVALS_PARQUET = os.path.join(PROCESSED_DIR, 'interarrivals.parquet')
SIM_INTERARRIVALS_PARQUET = os.path.join(PROCESSED_DIR, 'sim_interarrivals.parquet')
COMPARE_PLOT_PATH = os.path.join(PLOTS_DIR, 'real_vs_simulated_interarrivals.png')

if USE_SAMPLE_DATA:
    input_path, gzipped = os.path.join(SAMPLE_DATA_DIR, COSMIC_FILENAME), False
else:
    input_path, gzipped = os.path.join(REAL_DATA_DIR, COSMIC_FILENAME + '.gz'), True

os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f'[1/5] Loading {input_path}')
df_all = load_cosmic(input_path, gzipped=gzipped)
df_all = assign_arms(df_all)
n_samples = df_all['SAMPLE_NAME'].nunique()
print(f'      {len(df_all):,} mutations across {n_samples:,} samples')

print(f'[2/5] Computing rate map at {BIN_SIZE_BP // 1_000_000} Mb resolution')
rate_map = compute_rate_map(df_all, bin_size_bp=BIN_SIZE_BP)
print(f'      {len(rate_map):,} (arm, bin) pairs')

print('[3/5] Computing real per-sample interarrivals')
real_df = compute_interarrivals(df_all, per_sample=True)
real_df.to_parquet(REAL_INTERARRIVALS_PARQUET, index=False)
print(f'      {len(real_df):,} real interarrivals; saved: {REAL_INTERARRIVALS_PARQUET}')

print(f'[4/5] Simulating IPP interarrivals  '
      f'(n_samples={n_samples:,} × {N_REPLICATES} replicate(s))')
sim_df = simulate_ipp_interarrivals(
    rate_map, n_samples=n_samples, n_replicates=N_REPLICATES, seed=SEED,
)
sim_df.to_parquet(SIM_INTERARRIVALS_PARQUET, index=False)
print(f'      {len(sim_df):,} simulated interarrivals; saved: {SIM_INTERARRIVALS_PARQUET}')

print('[5/5] Plotting real vs simulated')
plot_real_vs_simulated_interarrivals(real_df, sim_df, COMPARE_PLOT_PATH)
print(f'      saved: {COMPARE_PLOT_PATH}')

print('\nDone.')
