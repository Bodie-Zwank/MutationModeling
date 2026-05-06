"""Simulate interarrival distances under the empirical IPP rate map λ̂(x)."""
import numpy as np
import pandas as pd


def simulate_ipp_interarrivals(
    rate_map: pd.DataFrame,
    n_samples: int,
    n_replicates: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate per-sample IPPs along each chromosome arm using λ̂(x).

    For every simulated sample × arm: K ~ Poisson(Λ_arm) where
    Λ_arm = Σ_b λ̂_b · w_b is the per-sample expected mutation count on the
    arm. Conditional on K, the K positions are iid from the piecewise-constant
    density λ̂(x)/Λ_arm — pick a bin with probability λ̂_b · w_b / Λ_arm,
    then place uniformly within it.

    Returns a DataFrame with column ``interarrival_bp`` containing
    consecutive-position diffs (>0) within each (simulated sample, arm).
    """
    rng = np.random.default_rng(seed)
    n_sim_samples = n_samples * n_replicates

    chunks = []
    for chrom_arm, sub in rate_map.groupby('CHROM_ARM', sort=False):
        sub = sub.sort_values('bin_idx')
        bin_start = sub['bin_start'].to_numpy(dtype=np.float64)
        bin_end = sub['bin_end'].to_numpy(dtype=np.float64)
        widths = bin_end - bin_start
        bin_mu = sub['lambda_hat'].to_numpy(dtype=np.float64) * widths
        Lambda_arm = float(bin_mu.sum())
        if Lambda_arm <= 0:
            continue
        bin_p = bin_mu / Lambda_arm

        Ks = rng.poisson(Lambda_arm, size=n_sim_samples)
        N = int(Ks.sum())
        if N < 2:
            continue

        chosen = rng.choice(len(bin_p), size=N, p=bin_p)
        offsets = rng.uniform(0.0, 1.0, size=N)
        positions = bin_start[chosen] + offsets * widths[chosen]
        sample_ids = np.repeat(np.arange(n_sim_samples), Ks)

        order = np.lexsort((positions, sample_ids))
        positions = positions[order]
        sample_ids = sample_ids[order]

        diffs = np.diff(positions)
        same_sample = sample_ids[1:] == sample_ids[:-1]
        diffs = diffs[same_sample]
        diffs = diffs[diffs > 0]
        if diffs.size:
            chunks.append(diffs)

    all_diffs = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    return pd.DataFrame({'interarrival_bp': all_diffs})
