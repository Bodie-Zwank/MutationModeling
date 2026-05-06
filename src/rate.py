"""Inhomogeneous Poisson process: empirical rate map λ̂(x) and time-rescaling."""
import numpy as np
import pandas as pd

from src.arms import arm_boundaries


def compute_rate_map(df: pd.DataFrame, bin_size_bp: int = 1_000_000) -> pd.DataFrame:
    """Pooled empirical mutation rate per chromosome arm, per bin.

    Returns one row per (CHROM_ARM, bin_idx). λ̂ is in units of mutations per
    base pair per sample — i.e. ``n_mutations / (bin_size × n_samples)``.
    Empty bins between mutations and the arm's end-points are kept as zeros
    so the cumulative integral covers the full arm in `rescale_positions`.
    """
    n_samples = df['SAMPLE_NAME'].nunique()
    bounds = arm_boundaries()

    rows = []
    for chrom_arm, (arm_start, arm_end) in bounds.items():
        arm_df = df[df['CHROM_ARM'] == chrom_arm]
        n_bins = max(1, int(np.ceil((arm_end - arm_start) / bin_size_bp)))

        # Vectorised count per bin via np.bincount
        if len(arm_df):
            bin_idx_per_mut = ((arm_df['GENOME_START'].values - arm_start) // bin_size_bp).astype(int)
            bin_idx_per_mut = np.clip(bin_idx_per_mut, 0, n_bins - 1)
            counts = np.bincount(bin_idx_per_mut, minlength=n_bins)
        else:
            counts = np.zeros(n_bins, dtype=int)

        for j, n_mut in enumerate(counts):
            bin_start = arm_start + j * bin_size_bp
            bin_end = min(bin_start + bin_size_bp, arm_end)
            width = bin_end - bin_start
            lambda_hat = n_mut / (width * n_samples) if width > 0 else 0.0
            rows.append({
                'CHROM_ARM': chrom_arm,
                'bin_idx': j,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'n_mutations': int(n_mut),
                'lambda_hat': float(lambda_hat),
            })

    rate_map = pd.DataFrame(rows)
    # Cumulative integral up to bin_start within each arm: Σ λ̂_k × width_k for k < j.
    # Also stored as 'expected mutations per sample' which equals n_mut_k / n_samples.
    rate_map = rate_map.sort_values(['CHROM_ARM', 'bin_idx']).reset_index(drop=True)
    bin_integral = rate_map['lambda_hat'] * (rate_map['bin_end'] - rate_map['bin_start'])
    rate_map['cum_int_at_bin_start'] = (
        bin_integral.groupby(rate_map['CHROM_ARM']).cumsum() - bin_integral
    )
    return rate_map


def rescale_positions(df: pd.DataFrame, rate_map: pd.DataFrame,
                      bin_size_bp: int = 1_000_000) -> pd.DataFrame:
    """Apply the time-rescaling transform under the empirical λ̂.

    Decomposes each sample's rate as λ_i,a(x) = b_{i,a} · λ̂(x), where
    λ̂(x) is the pooled (per-sample-average) shape and b_{i,a} = k_{i,a} / k̄_a
    is the burden of sample i on arm a relative to the average sample on
    that arm. With this scaling, under an IPP each (sample, arm)'s rescaled
    interarrivals are iid Exp(1).

    Adds:
      - rescaled_position:    b_{i,a} · ∫₀^x λ̂(s) ds, in Exp(1)-mean units.
      - rescaled_interarrival: per-(SAMPLE_NAME, CHROM_ARM) successive diffs.
    """
    bounds = arm_boundaries()
    arm_starts = pd.Series({a: s for a, (s, _) in bounds.items()}, name='arm_start')

    df = df.copy()
    df['arm_start'] = df['CHROM_ARM'].map(arm_starts)
    df['bin_idx'] = ((df['GENOME_START'] - df['arm_start']) // bin_size_bp).astype(int)

    max_bin_per_arm = rate_map.groupby('CHROM_ARM')['bin_idx'].max().rename('max_bin')
    df = df.merge(max_bin_per_arm, on='CHROM_ARM', how='left')
    df['bin_idx'] = df[['bin_idx', 'max_bin']].min(axis=1).astype(int)
    df = df.drop(columns=['max_bin'])

    df = df.merge(
        rate_map[['CHROM_ARM', 'bin_idx', 'bin_start', 'lambda_hat', 'cum_int_at_bin_start']],
        on=['CHROM_ARM', 'bin_idx'], how='left',
    )
    df['rescaled_position_pooled'] = (
        df['cum_int_at_bin_start']
        + (df['GENOME_START'] - df['bin_start']) * df['lambda_hat']
    )

    # Genome-wide burden factor per sample: b_i = K_i / K̄. Using a single
    # per-sample multiplier (rather than per-sample-per-arm) avoids
    # conditioning on small per-arm counts, so under IPP the rescaled
    # interarrivals are Exp(1) iid (not conditionally Beta-spacings).
    sample_count = df.groupby('SAMPLE_NAME').size().rename('K_i').reset_index()
    K_bar = sample_count['K_i'].mean()
    df = df.merge(sample_count, on='SAMPLE_NAME')
    df['burden_factor'] = df['K_i'] / K_bar
    df['rescaled_position'] = df['rescaled_position_pooled'] * df['burden_factor']

    df = df.sort_values(['SAMPLE_NAME', 'CHROM_ARM', 'GENOME_START']).reset_index(drop=True)
    df['rescaled_interarrival'] = df.groupby(['SAMPLE_NAME', 'CHROM_ARM'])['rescaled_position'].diff()
    df = df.dropna(subset=['rescaled_interarrival'])
    df = df[df['rescaled_interarrival'] > 0]
    return df.drop(columns=['arm_start', 'bin_idx', 'bin_start']).reset_index(drop=True)
