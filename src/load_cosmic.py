import pandas as pd
from pyarrow import csv as pa_csv

CHROMOSOME_MAP = {'X': '23', 'Y': '24', 'MT': '25'}


def load_cosmic(path: str, gzipped: bool = True) -> pd.DataFrame:
    """Read a COSMIC Mutant Census TSV; return CHROMOSOME / GENOME_START / SAMPLE_NAME.

    X/Y/MT are mapped to 23/24/25 so CHROMOSOME stays integer-typed.
    Rows missing chromosome or position are dropped, and duplicate
    (sample, chromosome, position) rows are collapsed.
    """
    table = pa_csv.read_csv(
        path,
        parse_options=pa_csv.ParseOptions(delimiter='\t'),
        convert_options=pa_csv.ConvertOptions(
            include_columns=['CHROMOSOME', 'GENOME_START', 'SAMPLE_NAME'],
        ),
    )
    df = table.to_pandas()

    df = df.dropna(subset=['CHROMOSOME', 'GENOME_START'])
    df['CHROMOSOME'] = df['CHROMOSOME'].astype(str).replace(CHROMOSOME_MAP).astype(int)
    df['GENOME_START'] = df['GENOME_START'].astype(int)
    df = df.drop_duplicates(subset=['SAMPLE_NAME', 'CHROMOSOME', 'GENOME_START'])
    return df.reset_index(drop=True)
