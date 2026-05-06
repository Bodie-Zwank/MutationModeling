"""COSMIC TSV → cleaned variants → IPP rate map + time-rescaling + legacy plots."""
import os

from src.arms import assign_arms
from src.interarrivals import compute_interarrivals
from src.load_cosmic import load_cosmic
from src.rate import compute_rate_map, rescale_positions
from src.plots import (
    plot_threshold_decomposition,
    plot_interarrival_distribution,
    plot_interarrival_window,
    plot_window_with_normal_fit,
    plot_window_with_skew_normal_fit,
    plot_rate_map,
    plot_rescaled_interarrival_check,
)

COSMIC_FILENAME = 'Cosmic_MutantCensus_v103_GRCh38.tsv'

SAMPLE_DATA_DIR = 'data/example_data'
REAL_DATA_DIR = 'data/real_data'
PROCESSED_DIR = 'data/processed_data'
PLOTS_DIR = os.path.join(PROCESSED_DIR, 'plots')

USE_SAMPLE_DATA = False
BIN_SIZE_BP = 1_000_000

INTERARRIVALS_PARQUET = os.path.join(PROCESSED_DIR, 'interarrivals.parquet')
RATE_MAP_PARQUET = os.path.join(PROCESSED_DIR, 'rate_map.parquet')

THRESHOLD_PLOT_PATH = os.path.join(PLOTS_DIR, 'threshold_decomposition.png')
DISTRIBUTION_PLOT_PATH = os.path.join(PLOTS_DIR, 'interarrival_distribution.png')
WINDOW_PLOT_PATH = os.path.join(PLOTS_DIR, 'interarrivals_1e6_1e8.png')
LEFT_HUMP_PLOT_PATH = os.path.join(PLOTS_DIR, 'interarrivals_1_1e6.png')
LEFT_HUMP_FIT_PATH = os.path.join(PLOTS_DIR, 'interarrivals_1_1e6_normal_fit.png')
RIGHT_HUMP_FIT_PATH = os.path.join(PLOTS_DIR, 'interarrivals_1e6_1e8_normal_fit.png')
RATE_MAP_PLOT_PATH = os.path.join(PLOTS_DIR, 'rate_map.png')
RESCALED_PLOT_PATH = os.path.join(PLOTS_DIR, 'rescaled_interarrivals.png')

if USE_SAMPLE_DATA:
    input_path, gzipped = os.path.join(SAMPLE_DATA_DIR, COSMIC_FILENAME), False
else:
    input_path, gzipped = os.path.join(REAL_DATA_DIR, COSMIC_FILENAME + '.gz'), True

print(f'[1/6] Loading {input_path}')
df_all = load_cosmic(input_path, gzipped=gzipped)
print(f'      {len(df_all):,} cleaned variants')

print('[2/6] Assigning chromosome arms')
df_all = assign_arms(df_all)

os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f'[3/6] Building empirical rate map λ̂(x) at {BIN_SIZE_BP // 1_000_000} Mb resolution')
rate_map = compute_rate_map(df_all, bin_size_bp=BIN_SIZE_BP)
rate_map.to_parquet(RATE_MAP_PARQUET, index=False)
print(f'      saved: {RATE_MAP_PARQUET}  ({len(rate_map):,} (arm, bin) pairs)')
plot_rate_map(rate_map, RATE_MAP_PLOT_PATH)
print(f'      saved: {RATE_MAP_PLOT_PATH}')

print('[4/6] Time-rescaling test (IPP goodness-of-fit)')
rescaled = rescale_positions(df_all, rate_map, bin_size_bp=BIN_SIZE_BP)
print(f'      {len(rescaled):,} rescaled interarrivals; '
      f'mean={rescaled["rescaled_interarrival"].mean():.4f}  '
      f'(Exp(1) → 1)')
plot_rescaled_interarrival_check(rescaled, RESCALED_PLOT_PATH)
print(f'      saved: {RESCALED_PLOT_PATH}')

print('[5/6] Computing legacy per-sample interarrivals (renewal view)')
df = compute_interarrivals(df_all, per_sample=True)
print(f'      {len(df):,} interarrival observations')
df.to_parquet(INTERARRIVALS_PARQUET, index=False)
print(f'      saved: {INTERARRIVALS_PARQUET}')

print('[6/6] Plotting renewal-process diagnostics')
plot_threshold_decomposition(df, THRESHOLD_PLOT_PATH)
plot_interarrival_distribution(df, DISTRIBUTION_PLOT_PATH)
plot_interarrival_window(df, WINDOW_PLOT_PATH, lo=1e6, hi=1e8)
plot_interarrival_window(df, LEFT_HUMP_PLOT_PATH, lo=1, hi=1e6)
plot_window_with_skew_normal_fit(
    df, LEFT_HUMP_FIT_PATH, lo=1, hi=1e6, fit_lo=1e2, fit_hi=1e6,
)
plot_window_with_normal_fit(df, RIGHT_HUMP_FIT_PATH, lo=1e6, hi=1e8)

print('\nDone.')
