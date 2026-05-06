import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize, stats

from src.arms import CENTROMERE_POSITIONS_GRCH38, CHROMOSOME_LENGTHS_GRCH38


def plot_threshold_decomposition(
    df: pd.DataFrame,
    output_path: str,
    thresholds: tuple = (5, 10, 20, 50),
    bins: int = 80,
) -> None:
    """Decompose the bimodal interarrival distribution by per-sample mutation count.

    For each threshold N, samples are split into '>N mutations' and '≤N mutations'
    groups, and their interarrival distributions are overlaid as log10 histograms.
    Heavy-mutator samples populate the left mode; sparse samples populate the right.
    """
    sample_counts = df.groupby('SAMPLE_NAME').size()
    log_all = np.log10(df['interarrival_bp'].values)
    edges = np.linspace(log_all.min(), log_all.max(), bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, threshold in zip(axes.flat, thresholds):
        high_samples = sample_counts[sample_counts > threshold].index
        low_samples = sample_counts[sample_counts <= threshold].index

        high = df[df['SAMPLE_NAME'].isin(high_samples)]['interarrival_bp'].values
        low = df[df['SAMPLE_NAME'].isin(low_samples)]['interarrival_bp'].values

        ax.hist(np.log10(high), bins=edges, color='seagreen', alpha=0.6,
                label=f'>{threshold} mutations (n={len(high):,}, {len(high_samples):,} samples)')
        ax.hist(np.log10(low), bins=edges, color='salmon', alpha=0.6,
                label=f'≤{threshold} mutations (n={len(low):,}, {len(low_samples):,} samples)')

        ax.set_title(f'Threshold: {threshold} mutations per sample')
        ax.set_xlabel('log₁₀(Interarrival distance, bp)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    fig.suptitle('Bimodal Decomposition at Different Mutation Count Thresholds',
                 fontsize=14, y=1.00)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_interarrival_distribution(
    df: pd.DataFrame,
    output_path: str,
    bins: int = 80,
) -> None:
    """log10 histogram of all interarrival distances — full bimodal distribution."""
    log_vals = np.log10(df['interarrival_bp'].values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('log₁₀(Interarrival distance, bp)')
    ax.set_ylabel('Count')
    ax.set_title(f'Interarrival distance distribution  (n={len(df):,})')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_interarrival_window(
    df: pd.DataFrame,
    output_path: str,
    lo: float = 1e6,
    hi: float = 1e8,
    bins: int = 60,
) -> None:
    """log10 histogram of interarrivals restricted to [lo, hi] bp."""
    vals = df['interarrival_bp'].values
    vals = vals[(vals >= lo) & (vals <= hi)]
    edges = np.linspace(np.log10(lo), np.log10(hi), bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.log10(vals), bins=edges, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('log₁₀(Interarrival distance, bp)')
    ax.set_ylabel('Count')
    ax.set_title(f'Interarrival distances in [{lo:.0e}, {hi:.0e}] bp  (n={len(vals):,})')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def log10_stats(vals: np.ndarray) -> dict:
    """Standard descriptive statistics of log10(vals)."""
    log_vals = np.log10(vals)
    return {
        'n':      len(vals),
        'mean':   float(log_vals.mean()),
        'median': float(np.median(log_vals)),
        'std':    float(log_vals.std(ddof=1)),
        'var':    float(log_vals.var(ddof=1)),
        'min':    float(log_vals.min()),
        'max':    float(log_vals.max()),
    }


def plot_window_with_peak_centered_fit(
    df: pd.DataFrame,
    output_path: str,
    lo: float,
    hi: float,
    bins: int = 60,
) -> dict:
    """Histogram with a normal anchored at the histogram peak.

    The mean μ is fixed at the peak. To avoid biasing σ by the long left tail,
    σ is fit only on data in the symmetric window [μ - (hi_log - μ), hi_log] —
    i.e. the same distance below the peak that the upper truncation sits above
    it. The plotted normal is un-truncated and drawn across the full range.
    """
    all_vals = df['interarrival_bp'].values
    shown = all_vals[(all_vals >= lo) & (all_vals <= hi)]
    log_shown = np.log10(shown)
    descriptive = log10_stats(shown)

    lo_log, hi_log = np.log10(lo), np.log10(hi)
    edges = np.linspace(lo_log, hi_log, bins + 1)

    counts, _ = np.histogram(log_shown, bins=edges)
    peak_idx = int(np.argmax(counts))
    mu = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    fit_lo_log = max(2 * mu - hi_log, lo_log)
    fit_mask = (log_shown >= fit_lo_log) & (log_shown <= hi_log)
    log_fit = log_shown[fit_mask]
    sigma = float(np.sqrt(np.mean((log_fit - mu) ** 2)))

    f_fit = len(log_fit) / len(log_shown)
    mass_fit = float(
        stats.norm.cdf(hi_log, mu, sigma) - stats.norm.cdf(fit_lo_log, mu, sigma)
    )
    scale = f_fit / mass_fit

    x = np.linspace(lo_log, hi_log, 400)
    pdf = stats.norm.pdf(x, mu, sigma) * scale

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_shown, bins=edges, density=True, color='steelblue',
            edgecolor='white', alpha=0.7, label='Observed')
    ax.plot(x, pdf, color='crimson', linewidth=2,
            label=f'Normal (peak-anchored)  μ={mu:.3f}, σ²={sigma ** 2:.3f}')
    ax.axvline(fit_lo_log, color='gray', linestyle=':', linewidth=1,
               label='fit window (symmetric about μ)')
    ax.axvline(hi_log, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('log₁₀(Interarrival distance, bp)')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Interarrival distances in [{lo:.0e}, {hi:.0e}] bp  (n={descriptive["n"]:,})'
    )

    annotation = (
        f"n           = {descriptive['n']:,}\n"
        f"sample μ    = {descriptive['mean']:.4f}\n"
        f"sample σ    = {descriptive['std']:.4f}\n"
        f"median      = {descriptive['median']:.4f}\n"
        f"────── peak-anchored fit ──────\n"
        f"μ (peak)    = {mu:.4f}\n"
        f"fit window  = [{fit_lo_log:.3f}, {hi_log:.3f}]\n"
        f"fit n       = {len(log_fit):,}\n"
        f"σ̂          = {sigma:.4f}\n"
        f"σ̂²         = {sigma ** 2:.4f}"
    )
    ax.text(
        0.98, 0.97, annotation,
        transform=ax.transAxes, ha='right', va='top',
        family='monospace', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85),
    )
    ax.legend(loc='upper left')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    descriptive['mu_peak'] = mu
    descriptive['sigma_hat'] = sigma
    descriptive['fit_lo_log'] = fit_lo_log
    descriptive['fit_hi_log'] = hi_log
    return descriptive


def plot_window_with_skew_normal_fit(
    df: pd.DataFrame,
    output_path: str,
    lo: float,
    hi: float,
    fit_lo: float | None = None,
    fit_hi: float | None = None,
    bins: int = 60,
) -> dict:
    """Histogram with a skew-normal MLE overlaid — handles asymmetric humps.

    A symmetric normal can't fit data with a long tail on one side. The
    skew-normal adds a shape parameter α: α<0 → long left tail, α>0 → long
    right tail, α=0 → ordinary normal. MLE is fit only on data in
    [fit_lo, fit_hi] (defaults to [lo, hi]) so artifact spikes outside the
    main bulk don't fight the optimizer; the fitted curve is then drawn
    un-truncated across the full visible window.
    """
    fit_lo = fit_lo if fit_lo is not None else lo
    fit_hi = fit_hi if fit_hi is not None else hi

    all_vals = df['interarrival_bp'].values
    shown = all_vals[(all_vals >= lo) & (all_vals <= hi)]
    fit_vals = all_vals[(all_vals >= fit_lo) & (all_vals <= fit_hi)]
    log_shown = np.log10(shown)
    log_fit = np.log10(fit_vals)
    descriptive = log10_stats(shown)

    lo_log, hi_log = np.log10(lo), np.log10(hi)
    fit_lo_log, fit_hi_log = np.log10(fit_lo), np.log10(fit_hi)
    edges = np.linspace(lo_log, hi_log, bins + 1)

    alpha, xi, omega = stats.skewnorm.fit(log_fit)
    delta = alpha / np.sqrt(1 + alpha ** 2)
    sn_mean = xi + omega * delta * np.sqrt(2 / np.pi)
    sn_var = omega ** 2 * (1 - 2 * delta ** 2 / np.pi)
    sn_skew = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi)) ** 3 / (
        (1 - 2 * delta ** 2 / np.pi) ** 1.5
    )

    f_fit = len(log_fit) / len(log_shown)
    mass_fit = float(
        stats.skewnorm.cdf(fit_hi_log, alpha, xi, omega)
        - stats.skewnorm.cdf(fit_lo_log, alpha, xi, omega)
    )
    scale = f_fit / mass_fit

    x = np.linspace(lo_log, hi_log, 600)
    pdf = stats.skewnorm.pdf(x, alpha, xi, omega) * scale

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_shown, bins=edges, density=True, color='steelblue',
            edgecolor='white', alpha=0.7, label='Observed')
    ax.plot(x, pdf, color='crimson', linewidth=2,
            label=f'Skew-normal MLE  ξ={xi:.3f}, ω={omega:.3f}, α={alpha:.3f}')
    if fit_lo > lo:
        ax.axvline(fit_lo_log, color='gray', linestyle=':', linewidth=1,
                   label='fit window')
    if fit_hi < hi:
        ax.axvline(fit_hi_log, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('log₁₀(Interarrival distance, bp)')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Interarrival distances in [{lo:.0e}, {hi:.0e}] bp  (n={descriptive["n"]:,})'
    )

    annotation = (
        f"mean     = {sn_mean:.4f}\n"
        f"variance = {sn_var:.4f}"
    )
    legend = ax.legend(loc='upper left')
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(
        legend_bbox.x0, legend_bbox.y0 - 0.02, annotation,
        transform=ax.transAxes, ha='left', va='top',
        family='monospace', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    descriptive.update({
        'xi': float(xi), 'omega': float(omega), 'alpha': float(alpha),
        'sn_mean': float(sn_mean), 'sn_var': float(sn_var), 'sn_skew': float(sn_skew),
        'fit_lo_log': fit_lo_log, 'fit_hi_log': fit_hi_log,
    })
    return descriptive


def fit_truncated_normal(log_vals: np.ndarray, lo_log: float, hi_log: float) -> tuple:
    """MLE of (μ, σ) for a normal observed only on [lo_log, hi_log].

    The naive sample mean/std are biased when the true distribution extends
    beyond the window: this maximizes the truncated-normal log-likelihood
    instead, which corrects for the missing tails.
    """
    def neg_log_likelihood(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        a, b = (lo_log - mu) / sigma, (hi_log - mu) / sigma
        mass = stats.norm.cdf(b) - stats.norm.cdf(a)
        if mass <= 0:
            return np.inf
        ll = stats.norm.logpdf(log_vals, mu, sigma).sum() - len(log_vals) * np.log(mass)
        return -ll

    init = [float(log_vals.mean()), float(np.log(log_vals.std(ddof=1)))]
    result = optimize.minimize(neg_log_likelihood, init, method='Nelder-Mead')
    mu_hat, log_sigma_hat = result.x
    return float(mu_hat), float(np.exp(log_sigma_hat))


def plot_window_with_normal_fit(
    df: pd.DataFrame,
    output_path: str,
    lo: float,
    hi: float,
    fit_lo: float | None = None,
    fit_hi: float | None = None,
    bins: int = 60,
) -> dict:
    """log10 histogram in [lo, hi] with a truncated-normal MLE overlaid.

    The MLE is fit on data restricted to [fit_lo, fit_hi] (defaults to [lo, hi]),
    so artifact tails outside the bulk can be excluded from the fit while still
    being shown in the histogram.
    """
    fit_lo = fit_lo if fit_lo is not None else lo
    fit_hi = fit_hi if fit_hi is not None else hi

    all_vals = df['interarrival_bp'].values
    shown = all_vals[(all_vals >= lo) & (all_vals <= hi)]
    fit_vals = all_vals[(all_vals >= fit_lo) & (all_vals <= fit_hi)]
    log_shown = np.log10(shown)
    log_fit = np.log10(fit_vals)
    descriptive = log10_stats(shown)

    fit_lo_log, fit_hi_log = np.log10(fit_lo), np.log10(fit_hi)
    lo_log, hi_log = np.log10(lo), np.log10(hi)
    mu, sigma = fit_truncated_normal(log_fit, fit_lo_log, fit_hi_log)

    a, b = (fit_lo_log - mu) / sigma, (fit_hi_log - mu) / sigma
    edges = np.linspace(lo_log, hi_log, bins + 1)
    x = np.linspace(fit_lo_log, fit_hi_log, 400)
    # Scale truncnorm PDF to match the *fitted* subset's density level so
    # the curve sits on the actual bars when the fit window is narrower.
    pdf = stats.truncnorm.pdf(x, a, b, loc=mu, scale=sigma) * (len(fit_vals) / len(shown))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_shown, bins=edges, density=True, color='steelblue',
            edgecolor='white', alpha=0.7, label='Observed')
    ax.plot(x, pdf, color='crimson', linewidth=2,
            label=f'Truncated-normal MLE on [{fit_lo:.0e}, {fit_hi:.0e}]  '
                  f'μ={mu:.3f}, σ²={sigma ** 2:.3f}')
    if fit_lo > lo:
        ax.axvline(fit_lo_log, color='gray', linestyle=':', linewidth=1,
                   label=f'fit window')
    if fit_hi < hi:
        ax.axvline(fit_hi_log, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('log₁₀(Interarrival distance, bp)')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Interarrival distances in [{lo:.0e}, {hi:.0e}] bp  (n={descriptive["n"]:,})'
    )

    annotation = (
        f"mean     = {mu:.4f}\n"
        f"variance = {sigma ** 2:.4f}"
    )
    legend = ax.legend(loc='upper left')
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(
        legend_bbox.x0, legend_bbox.y0 - 0.02, annotation,
        transform=ax.transAxes, ha='left', va='top',
        family='monospace', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    descriptive['mu_hat'] = mu
    descriptive['sigma_hat'] = sigma
    return descriptive


def plot_rate_map(rate_map: pd.DataFrame, output_path: str) -> None:
    """λ̂(x) along the genome — one panel per chromosome (1–22, X, Y).

    p and q arms are drawn as a single continuous trace with a vertical
    dotted line at the centromere midpoint. y-axis is log-scaled (mutations
    per bp per sample); empty bins drop to a small floor for visibility.
    MT is omitted (only ~17 kb, doesn't merit its own panel).
    """
    chroms = list(range(1, 25))  # 1-22, X=23, Y=24
    chrom_label = {**{i: str(i) for i in range(1, 23)}, 23: 'X', 24: 'Y'}

    floor = rate_map.loc[rate_map['lambda_hat'] > 0, 'lambda_hat'].min() / 10
    rate_map = rate_map.copy()
    rate_map['lambda_plot'] = rate_map['lambda_hat'].clip(lower=floor)

    fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharey=True)
    for ax, chrom in zip(axes.flat, chroms):
        chrom_length = CHROMOSOME_LENGTHS_GRCH38[chrom]
        centromere = CENTROMERE_POSITIONS_GRCH38[chrom]

        for arm_letter in ('p', 'q'):
            arm = f'{chrom}{arm_letter}'
            sub = rate_map[rate_map['CHROM_ARM'] == arm]
            if len(sub) == 0:
                continue
            mid = (sub['bin_start'] + sub['bin_end']) / 2 / 1e6
            ax.plot(mid, sub['lambda_plot'], linewidth=0.8, color='steelblue')

        if centromere is not None:
            ax.axvline(centromere / 1e6, color='gray', linestyle=':', linewidth=0.8)

        ax.set_yscale('log')
        ax.set_xlim(0, chrom_length / 1e6)
        ax.set_title(f'chr {chrom_label[chrom]}', fontsize=10)
        ax.tick_params(labelsize=8)

    for ax in axes[-1]:
        ax.set_xlabel('Position (Mb)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('λ̂ (mut / bp / sample)', fontsize=9)

    fig.suptitle('Empirical mutation rate λ̂(x) per chromosome arm', fontsize=14, y=0.995)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rescaled_interarrival_check(rescaled_df: pd.DataFrame, output_path: str) -> None:
    """Goodness-of-fit for the IPP model via the time-rescaling theorem.

    Under an IPP with the true λ(x), rescaled interarrivals are iid Exp(1).
    Two panels: density histogram with Exp(1) PDF overlaid, and an Exp(1)
    Q-Q plot. KS statistic + p-value annotated.
    """
    vals = rescaled_df['rescaled_interarrival'].values
    n = len(vals)

    ks_stat, ks_p = stats.kstest(vals, 'expon')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    upper = float(np.quantile(vals, 0.999))  # clip extreme tail for readability
    edges = np.linspace(0, upper, 80)
    ax.hist(vals, bins=edges, density=True, color='steelblue',
            edgecolor='white', alpha=0.7, label='Rescaled interarrivals')
    x = np.linspace(0, upper, 400)
    ax.plot(x, stats.expon.pdf(x), color='crimson', linewidth=2, label='Exp(1) PDF')
    ax.set_xlabel('Rescaled interarrival (expected mutations/sample)')
    ax.set_ylabel('Density')
    ax.set_title('Rescaled interarrival histogram vs Exp(1)')
    ax.legend(loc='upper right')

    ax = axes[1]
    n_qq = min(50_000, n)
    rng = np.random.default_rng(0)
    sample = vals if n <= n_qq else rng.choice(vals, size=n_qq, replace=False)
    sample_sorted = np.sort(sample)
    p = (np.arange(1, len(sample_sorted) + 1) - 0.5) / len(sample_sorted)
    theoretical = stats.expon.ppf(p)
    ax.scatter(theoretical, sample_sorted, s=2, alpha=0.3, color='steelblue')
    line_max = float(min(sample_sorted.max(), theoretical.max()) * 1.05)
    ax.plot([0, line_max], [0, line_max], color='crimson', linestyle='--',
            linewidth=1.5, label='y = x')
    ax.set_xlabel('Exp(1) theoretical quantiles')
    ax.set_ylabel('Empirical quantiles')
    ax.set_title('Q-Q: rescaled interarrivals vs Exp(1)')
    ax.legend(loc='upper left')

    annotation = (
        f"n           = {n:,}\n"
        f"sample mean = {vals.mean():.4f}  (Exp(1) → 1)\n"
        f"sample var  = {vals.var(ddof=1):.4f}  (Exp(1) → 1)\n"
        f"KS stat     = {ks_stat:.4f}\n"
        f"KS p-value  = {ks_p:.2e}"
    )
    axes[0].text(
        0.98, 0.55, annotation,
        transform=axes[0].transAxes, ha='right', va='top',
        family='monospace', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85),
    )

    fig.suptitle('Time-rescaling test for inhomogeneous Poisson model', fontsize=13, y=1.00)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
