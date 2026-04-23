import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DATA_PATH = "data/processed_data/processed_data.parquet"

if __name__ == "__main__":
    bin_counts = pd.read_parquet(PROCESSED_DATA_PATH)
    #bin_counts = (bin_counts[bin_counts["total_mutations"] != 0])
    #bin_counts = bin_counts.drop([])
    # Get unique chromosomes and bins (preserving order)
    chromosomes = bin_counts['CHROMOSOME'].unique()
    bins = sorted(bin_counts['bin'].unique())

    x = np.arange(len(chromosomes))
    n_bins = len(bins)
    width = 0.8 / n_bins  # total bar group width = 0.8

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, bin_val in enumerate(bins):
        subset = bin_counts[bin_counts['bin'] == bin_val]
        # Align subset to chromosome order
        rates = [
            subset.loc[subset['CHROMOSOME'] == c, 'mutation_rates'].values[0]
            if c in subset['CHROMOSOME'].values else 0
            for c in chromosomes
        ]
        offset = (i - n_bins / 2 + 0.5) * width
        ax.bar(x + offset, rates, width=width, label=f'Bin {bin_val}')

    ax.set_xticks(x)
    ax.set_xticklabels(chromosomes)
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Mutation Rate')
    ax.set_title('Mutation Rates by Chromosome and Bin')
    ax.legend(title='Bin', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('mutation_rates.png', dpi=150, bbox_inches='tight')
    plt.show()