from __future__ import annotations

import numpy as np
import pandas as pd

from src.types import StrategyParams


def select_universe_k(fund_codes: list[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    if k >= len(fund_codes):
        return fund_codes
    idx = rng.choice(len(fund_codes), size=k, replace=False)
    return [fund_codes[i] for i in idx]


def equal_weight_portfolio(
    returns_long: pd.DataFrame, params: StrategyParams
) -> pd.Series:
    """
    returns_long: [date, fund_code, ret]
    Output: portfolio daily return series indexed by date
    """
    df = returns_long.copy()

    all_codes = sorted(df["fund_code"].unique().tolist())
    chosen = select_universe_k(all_codes, params.k, params.seed)

    df = df[df["fund_code"].isin(chosen)]

    # date x fund pivot
    wide = df.pivot_table(
        index="date", columns="fund_code", values="ret", aggfunc="mean"
    )

    # equal-weight: her gün mevcut fonların ortalaması
    port_ret = wide.mean(axis=1, skipna=True)

    # NaN günleri temizle
    port_ret = port_ret.dropna().sort_index()
    port_ret.name = "portfolio_ret"
    return port_ret
