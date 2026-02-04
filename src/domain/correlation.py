from __future__ import annotations

from itertools import combinations

import pandas as pd


def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    return returns_wide.corr()


def correlation_pairs(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for a, b in combinations(corr_matrix.columns, 2):
        value = corr_matrix.loc[a, b]
        pairs.append({"fund_a": a, "fund_b": b, "corr": value})
    return pd.DataFrame(pairs).sort_values("corr", ascending=False)
