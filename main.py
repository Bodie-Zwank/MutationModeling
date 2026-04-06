import os
import pandas as pd

PROCESSED_DATA_PATH = "data/processed_data/processed_data.parquet"

if __name__ == "__main__":
    bin_counts = pd.read_parquet(PROCESSED_DATA_PATH)
    print(bin_counts)