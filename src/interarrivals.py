import pandas as pd


def compute_interarrivals(df: pd.DataFrame, per_sample: bool = True) -> pd.DataFrame:
    """Base-pair gaps between consecutive mutations within each (sample, arm) group.

    The first mutation of each group has no predecessor and is dropped, as are
    zero-bp gaps (duplicate-position artifacts).
    """
    group_cols = ['SAMPLE_NAME', 'CHROM_ARM'] if per_sample else ['CHROM_ARM']
    df = df.sort_values(group_cols + ['GENOME_START']).reset_index(drop=True).copy()
    df['interarrival_bp'] = df.groupby(group_cols)['GENOME_START'].diff()
    df = df.dropna(subset=['interarrival_bp'])
    df = df[df['interarrival_bp'] > 0]
    df['interarrival_bp'] = df['interarrival_bp'].astype(int)
    return df.reset_index(drop=True)
