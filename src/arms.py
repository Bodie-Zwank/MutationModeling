import pandas as pd

# GRCh38 centromere midpoints (bp), from UCSC. Approximate — sufficient for
# p / q arm assignment, not base-pair precision.
CENTROMERE_POSITIONS_GRCH38 = {
    1:  123_400_000,  2:   93_900_000,  3:   90_900_000,  4:   50_400_000,
    5:   48_400_000,  6:   61_000_000,  7:   59_900_000,  8:   45_600_000,
    9:   49_000_000,  10:  40_200_000,  11:  53_700_000,  12:  35_800_000,
    13:  17_900_000,  14:  17_600_000,  15:  19_000_000,  16:  36_600_000,
    17:  24_000_000,  18:  17_200_000,  19:  26_500_000,  20:  27_500_000,
    21:  13_200_000,  22:  14_700_000,
    23:  60_600_000,  # X
    24:  10_400_000,  # Y
    25:  None,        # MT — no centromere; treated as a single q arm
}

# GRCh38 chromosome lengths (bp), from UCSC seqlimits.
CHROMOSOME_LENGTHS_GRCH38 = {
    1:  248_956_422,  2:  242_193_529,  3:  198_295_559,  4:  190_214_555,
    5:  181_538_259,  6:  170_805_979,  7:  159_345_973,  8:  145_138_636,
    9:  138_394_717,  10: 133_797_422,  11: 135_086_622,  12: 133_275_309,
    13: 114_364_328,  14: 107_043_718,  15: 101_991_189,  16:  90_338_345,
    17:  83_257_441,  18:  80_373_285,  19:  58_617_616,  20:  64_444_167,
    21:  46_709_983,  22:  50_818_468,
    23: 156_040_895,  # X
    24:  57_227_415,  # Y
    25:      16_569,  # MT
}


def assign_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Add ARM ('p' | 'q') and CHROM_ARM (e.g. '7q') columns."""
    centromere = df['CHROMOSOME'].map(CENTROMERE_POSITIONS_GRCH38)
    is_q = df['GENOME_START'] >= centromere.fillna(0)
    df = df.copy()
    df['ARM'] = is_q.map({True: 'q', False: 'p'})
    df['CHROM_ARM'] = df['CHROMOSOME'].astype(str) + df['ARM']
    return df


def arm_boundaries() -> dict:
    """Return {CHROM_ARM: (start_bp, end_bp)} using the GRCh38 maps above.

    p arm: [0, centromere). q arm: [centromere, chrom_length]. Chromosomes
    with no centromere (MT) become a single q arm spanning the full length.
    """
    bounds = {}
    for chrom, length in CHROMOSOME_LENGTHS_GRCH38.items():
        centromere = CENTROMERE_POSITIONS_GRCH38[chrom]
        if centromere is None:
            bounds[f'{chrom}q'] = (0, length)
        else:
            bounds[f'{chrom}p'] = (0, centromere)
            bounds[f'{chrom}q'] = (centromere, length)
    return bounds
