import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

PROCESSED_DATA_PATH = "data/processed_data/interarrivals.parquet"
PLOTS_OUTPUT_PATH = "data/processed_data/plots"

os.makedirs(PLOTS_OUTPUT_PATH, exist_ok=True)

df = pd.read_parquet(PROCESSED_DATA_PATH)
print(f"Loaded {len(df):,} interarrival observations")

# Filter out zero interarrivals (duplicate positions)
df = df[df['interarrival_bp'] > 0]
print(f"After filtering zeros: {len(df):,} observations")

interarrivals = df['interarrival_bp'].values
log_interarrivals = np.log10(interarrivals)

# Determine grouping column
GEO_COL = 'CHROM_ARM' if 'CHROM_ARM' in df.columns else 'CHROMOSOME'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Panel 1: Log-scale histogram ──────────────────────────────────────────────
ax = axes[0, 0]
ax.hist(log_interarrivals, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.median(log_interarrivals), color='red', linestyle='--', linewidth=1.5,
           label=f'Median = {10**np.median(log_interarrivals):,.0f} bp')
ax.axvline(np.mean(log_interarrivals), color='orange', linestyle='--', linewidth=1.5,
           label=f'Mean = {interarrivals.mean():,.0f} bp')
ax.set_xlabel('log₁₀(Interarrival distance, bp)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Interarrival Distances')
ax.legend(fontsize=8)

# ── Panel 2: Empirical CDF ───────────────────────────────────────────────────
ax = axes[0, 1]
sorted_vals = np.sort(interarrivals)
ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
ax.plot(sorted_vals, ecdf_y, color='steelblue', linewidth=0.8)
ax.set_xscale('log')
ax.set_xlabel('Interarrival distance (bp)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Empirical CDF')
ax.grid(True, alpha=0.3)
# Mark key percentiles
for q, label in [(0.25, '25th'), (0.50, '50th'), (0.75, '75th')]:
    val = np.percentile(interarrivals, q * 100)
    ax.axhline(q, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'{label}: {val:,.0f} bp', xy=(val, q),
                fontsize=7, ha='left', va='bottom',
                xytext=(5, 3), textcoords='offset points')

# ── Panel 3: Per-arm boxplot ─────────────────────────────────────────────────
ax = axes[1, 0]
groups = df.groupby(GEO_COL)
labels_sorted = sorted(groups.groups.keys(),
                       key=lambda x: (int(''.join(c for c in str(x) if c.isdigit()) or 0),
                                      str(x)))
box_data = [np.log10(groups.get_group(g)['interarrival_bp'].values) for g in labels_sorted]
bp = ax.boxplot(box_data, tick_labels=labels_sorted, patch_artist=True,
                showfliers=False, medianprops=dict(color='red', linewidth=1.5))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.set_xlabel(GEO_COL)
ax.set_ylabel('log₁₀(Interarrival distance, bp)')
ax.set_title(f'Interarrival Distribution by {GEO_COL}')
ax.tick_params(axis='x', rotation=90, labelsize=7)

# ── Panel 4: Log-log survival plot ───────────────────────────────────────────
ax = axes[1, 1]
survival_y = 1.0 - ecdf_y
# Subsample for plotting efficiency (keep endpoints)
n = len(sorted_vals)
if n > 5000:
    idx = np.unique(np.concatenate([
        np.geomspace(1, n - 1, 5000).astype(int),
        [0, n - 1]
    ]))
    ax.plot(sorted_vals[idx], survival_y[idx], color='steelblue', linewidth=0.8)
else:
    ax.plot(sorted_vals, survival_y, color='steelblue', linewidth=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Interarrival distance (bp)')
ax.set_ylabel('P(X > x)')
ax.set_title('Survival Function (log-log)')
ax.grid(True, alpha=0.3, which='both')

fig.suptitle('Empirical Distribution of Mutation Interarrival Times', fontsize=14, y=1.01)
plt.tight_layout()
path = os.path.join(PLOTS_OUTPUT_PATH, "interarrival_distribution.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {path}")
plt.close(fig)

# ── Log-normal Q-Q plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

# Fit log-normal params to the data (fitting normal to log-transformed values)
ln_interarrivals = np.log(interarrivals)  # natural log for scipy convention
mu, sigma = ln_interarrivals.mean(), ln_interarrivals.std()

# Theoretical quantiles from fitted log-normal
sample_sorted = np.sort(interarrivals)
p = (np.arange(1, len(sample_sorted) + 1) - 0.5) / len(sample_sorted)
theoretical = stats.lognorm.ppf(p, s=sigma, scale=np.exp(mu))

# Use log-log scale so the full range is visible
ax.scatter(theoretical, sample_sorted, s=1, alpha=0.3, color='steelblue')
min_val = max(1, min(theoretical.min(), sample_sorted.min()))
max_val = max(theoretical.max(), sample_sorted.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y = x (perfect fit)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Theoretical quantiles (Log-normal)')
ax.set_ylabel('Observed quantiles (bp)')
ax.set_title('Q-Q Plot: Observed Interarrivals vs. Log-normal')
ax.legend()

plt.tight_layout()
path = os.path.join(PLOTS_OUTPUT_PATH, "lognormal_qq_plot.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
print(f"Saved: {path}")
plt.close(fig)

print("Done.")
