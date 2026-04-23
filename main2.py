import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

PROCESSED_DATA_PATH = "data/processed_data/interarrivals.parquet"
PLOTS_OUTPUT_PATH = "data/processed_data/plots"
PER_SAMPLE = True  # should match what was used when generating the parquet

os.makedirs(PLOTS_OUTPUT_PATH, exist_ok=True)

df = pd.read_parquet(PROCESSED_DATA_PATH)
print(f"Loaded {len(df):,} interarrival observations")

# Filter out zero interarrivals — these indicate duplicate positions within a group
# and are not valid interarrival events for a Poisson process
n_before = len(df)
df = df[df['interarrival_bp'] > 0]
n_removed = n_before - len(df)
print(f"Removed {n_removed:,} zero interarrivals ({100 * n_removed / n_before:.2f}% of data)")

# Determine the grouping column present in the parquet
if 'CHROM_ARM' in df.columns:
    GEO_COL = 'CHROM_ARM'
    geo_label_prefix = 'chr'
else:
    GEO_COL = 'CHROMOSOME'
    geo_label_prefix = 'chr'

print(f"Grouping by: {GEO_COL}")
print(f"Total interarrival observations after filtering: {len(df):,}")


# ─────────────────────────────────────────────
# Helper: fit + KS test for one slice of data
# ─────────────────────────────────────────────
def fit_and_test(interarrivals: np.ndarray, label: str) -> dict:
    """
    Fits an exponential distribution to `interarrivals` via MLE and runs a
    one-sample KS test against the fitted distribution.

    The exponential is parameterized as: f(x) = λ·exp(-λx), x >= 0
    scipy uses the scale = 1/λ convention.

    Returns a dict with fit params and test results, or None if too few observations.
    """
    n = len(interarrivals)
    if n < 30:
        print(f"  [SKIP] {label}: too few observations ({n})")
        return None

    # MLE for exponential: λ_hat = 1 / mean; fix loc=0 (interarrivals start at 0)
    loc, scale = stats.expon.fit(interarrivals, floc=0)
    lambda_hat = 1.0 / scale

    ks_stat, p_value = stats.kstest(interarrivals, 'expon', args=(loc, scale))

    return {
        'label':      label,
        'n':          n,
        'mean_bp':    interarrivals.mean(),
        'lambda_hat': lambda_hat,   # mutations per bp
        'scale':      scale,        # mean interarrival in bp (= 1/λ)
        'ks_stat':    ks_stat,
        'p_value':    p_value,
        'reject_H0':  p_value < 0.05,  # H0: data is exponentially distributed
    }


# ─────────────────────────────────────────────
# 1. Global fit (all arms, all samples)
# ─────────────────────────────────────────────
print("\n── Global fit ──")
global_result = fit_and_test(df['interarrival_bp'].values, label='All data')
if global_result:
    print(f"  λ̂ = {global_result['lambda_hat']:.4e} mutations/bp  "
          f"(mean gap = {global_result['scale']:,.0f} bp)")
    print(f"  KS stat = {global_result['ks_stat']:.4f},  p = {global_result['p_value']:.4e}  "
          f"{'[REJECT H0]' if global_result['reject_H0'] else '[fail to reject H0]'}")


# ─────────────────────────────────────────────
# 2. Per-arm (or per-chromosome) fits
# ─────────────────────────────────────────────
print(f"\n── Per-{GEO_COL.lower()} fits ──")
geo_results = []
for geo_val, group in df.groupby(GEO_COL):
    label = f"{geo_label_prefix}{geo_val}"
    result = fit_and_test(group['interarrival_bp'].values, label=label)
    if result:
        result['geo_val'] = geo_val
        geo_results.append(result)
        flag = '  *** REJECT H0 ***' if result['reject_H0'] else ''
        print(f"  {label:<10}  n={result['n']:>8,}  λ̂={result['lambda_hat']:.3e}  "
              f"KS={result['ks_stat']:.4f}  p={result['p_value']:.3e}{flag}")

geo_df = pd.DataFrame(geo_results)


# ─────────────────────────────────────────────
# 3. Per-sample fits (if PER_SAMPLE=True)
# ─────────────────────────────────────────────
sample_df = None
if PER_SAMPLE and 'SAMPLE_NAME' in df.columns:
    print("\n── Per-sample fits (summary) ──")
    sample_results = []
    for (sample, geo_val), group in df.groupby(['SAMPLE_NAME', GEO_COL]):
        label = f"{sample}_{geo_label_prefix}{geo_val}"
        result = fit_and_test(group['interarrival_bp'].values, label=label)
        if result:
            result['geo_val'] = geo_val
            sample_results.append(result)

    sample_df = pd.DataFrame(sample_results)
    reject_pct = sample_df['reject_H0'].mean() * 100
    print(f"  Tested {len(sample_df):,} (sample, {GEO_COL.lower()}) pairs")
    print(f"  Rejected H0 (p<0.05) in {reject_pct:.1f}% of pairs")
    print(f"  Median KS stat: {sample_df['ks_stat'].median():.4f}")
    print(f"  Median p-value: {sample_df['p_value'].median():.4e}")


# ─────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────

def plot_exponential_fit(ax, interarrivals, result, title, color='steelblue'):
    """Overlays a histogram of interarrivals with the fitted exponential PDF."""
    x_max = np.percentile(interarrivals, 99)  # clip extreme outliers for readability
    clipped = interarrivals[interarrivals <= x_max]

    ax.hist(clipped, bins=80, density=True, alpha=0.6, color=color, label='Observed')

    x = np.linspace(0, x_max, 500)
    pdf = stats.expon.pdf(x, loc=0, scale=result['scale'])
    ax.plot(x, pdf, 'r-', linewidth=2,
            label=f"Exp fit (λ={result['lambda_hat']:.2e})")

    ax.set_title(
        f"{title}\nKS={result['ks_stat']:.3f}, p={result['p_value']:.2e}  "
        f"{'[REJECT]' if result['reject_H0'] else '[fail to reject]'}",
        fontsize=9
    )
    ax.set_xlabel("Interarrival distance (bp)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)


# ── Plot A: Global fit ──────────────────────────────────────────────────────
if global_result:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_exponential_fit(ax, df['interarrival_bp'].values, global_result,
                         title=f"All {GEO_COL.lower()}s — exponential fit")
    plt.tight_layout()
    path = os.path.join(PLOTS_OUTPUT_PATH, "global_fit.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close(fig)


# ── Plot B: Per-arm grid ─────────────────────────────────────────────────────
if geo_results:
    n_geos = len(geo_results)
    ncols = 5
    nrows = int(np.ceil(n_geos / ncols))

    # squeeze=False ensures axes is always 2D regardless of grid shape,
    # so flatten() reliably gives a 1D array of Axes objects
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2), squeeze=False)
    axes = axes.flatten()

    for i, result in enumerate(geo_results):
        data = df[df[GEO_COL] == result['geo_val']]['interarrival_bp'].values
        plot_exponential_fit(axes[i], data, result,
                             title=result['label'], color='steelblue')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Per-{GEO_COL.lower()} exponential fits", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_OUTPUT_PATH, f"per_{GEO_COL.lower()}_fits.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot C: Per-sample KS stat distribution ─────────────────────────────────
if sample_df is not None and len(sample_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(sample_df['ks_stat'], bins=40, color='steelblue', edgecolor='white')
    axes[0].set_title(f"Distribution of KS statistics\nacross (sample, {GEO_COL.lower()}) pairs")
    axes[0].set_xlabel("KS statistic")
    axes[0].set_ylabel("Count")

    axes[1].hist(sample_df['p_value'], bins=40, color='darkorange', edgecolor='white')
    axes[1].axvline(0.05, color='red', linestyle='--', label='p=0.05')
    axes[1].set_title(f"Distribution of KS p-values\nacross (sample, {GEO_COL.lower()}) pairs")
    axes[1].set_xlabel("p-value")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_OUTPUT_PATH, "per_sample_ks_distribution.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot D: Q-Q plot for global fit ─────────────────────────────────────────
if global_result:
    fig, ax = plt.subplots(figsize=(5, 5))
    sample = np.sort(df['interarrival_bp'].values)
    p = (np.arange(1, len(sample) + 1) - 0.5) / len(sample)
    theoretical = stats.expon.ppf(p, loc=0, scale=global_result['scale'])

    clip = np.percentile(sample, 99)
    mask = sample <= clip
    ax.scatter(theoretical[mask], sample[mask], s=1, alpha=0.3, color='steelblue')
    max_val = max(theoretical[mask].max(), sample[mask].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel("Theoretical quantiles (Exponential)")
    ax.set_ylabel("Observed quantiles (bp)")
    ax.set_title("Q-Q plot: observed interarrivals vs. exponential")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_OUTPUT_PATH, "global_qq_plot.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot E: λ across arms (mutation rate landscape) ─────────────────────────
if len(geo_df) > 1:
    fig, ax = plt.subplots(figsize=(max(8, len(geo_df) * 0.4), 4))
    colors = ['salmon' if r else 'steelblue' for r in geo_df['reject_H0']]
    ax.bar(geo_df['label'], geo_df['lambda_hat'], color=colors, edgecolor='white')
    ax.set_xlabel(GEO_COL)
    ax.set_ylabel("λ̂ (mutations / bp)")
    ax.set_title("Estimated mutation rate per chromosome arm\n"
                 "(red = exponential fit rejected at p<0.05)")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_OUTPUT_PATH, "lambda_by_arm.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

print("\nDone.")