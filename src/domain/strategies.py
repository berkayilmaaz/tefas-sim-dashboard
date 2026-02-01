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


def compute_rebalance_dates(
    dates: pd.DatetimeIndex, frequency: str
) -> pd.DatetimeIndex:
    if dates.empty:
        return dates
    if frequency == "daily":
        return dates
    series = pd.Series(dates, index=dates)
    if frequency == "weekly":
        return series.groupby(dates.to_period("W")).last().values
    if frequency == "monthly":
        return series.groupby(dates.to_period("M")).last().values
    raise ValueError(f"Unknown rebalance frequency: {frequency}")


def _momentum_scores(window: pd.DataFrame) -> pd.Series:
    return (1.0 + window).prod(skipna=True) - 1.0


def _low_vol_scores(window: pd.DataFrame) -> pd.Series:
    return window.std(skipna=True)


def _sharpe_scores(window: pd.DataFrame) -> pd.Series:
    mean = window.mean(skipna=True)
    std = window.std(skipna=True)
    scores = mean / std.replace(0.0, np.nan)
    return scores


def _select_top_k(scores: pd.Series, k: int, low_is_best: bool) -> list[str]:
    scores = scores.dropna()
    if scores.empty:
        return []
    if low_is_best:
        return scores.nsmallest(k).index.tolist()
    return scores.nlargest(k).index.tolist()


def build_weights(
    returns_wide: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(returns_wide.index)
    rebalance_dates = compute_rebalance_dates(dates, params.rebalance)
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}

    for date in rebalance_dates:
        idx = dates.get_loc(date)
        if params.strategy == "equal_weight":
            selected = returns_wide.columns.tolist()
        else:
            if idx < params.lookback:
                continue
            window = returns_wide.iloc[idx - params.lookback : idx]
            if params.strategy == "momentum_top_k":
                scores = _momentum_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
            elif params.strategy == "low_vol_top_k":
                scores = _low_vol_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=True)
            elif params.strategy == "sharpe_top_k":
                scores = _sharpe_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
            else:
                raise ValueError(f"Unknown strategy: {params.strategy}")

        if not selected:
            continue
        weight = 1.0 / len(selected)
        weights_by_date[pd.Timestamp(date)] = pd.Series(
            weight, index=selected, name="weight"
        )

    if not weights_by_date:
        return pd.DataFrame(), pd.DataFrame()

    weights_daily = pd.DataFrame(
        index=dates, columns=returns_wide.columns, dtype=float
    )
    rows = []
    for date, weights in weights_by_date.items():
        weights_daily.loc[date, weights.index] = weights.values
        df = weights.reset_index()
        df.columns = ["fund_code", "weight"]
        df["rebalance_date"] = date
        rows.append(df)

    weights_daily = weights_daily.sort_index().ffill()
    first_date = min(weights_by_date.keys())
    weights_daily = weights_daily.loc[first_date:]

    weights_table = pd.concat(rows, ignore_index=True)[
        ["rebalance_date", "fund_code", "weight"]
    ]
    return weights_daily, weights_table


def compute_portfolio_returns(
    returns_wide: pd.DataFrame, weights_daily: pd.DataFrame
) -> pd.Series:
    if weights_daily.empty:
        return pd.Series(dtype=float, name="portfolio_ret")
    aligned_returns = returns_wide.loc[weights_daily.index]
    port_ret = (aligned_returns.fillna(0.0) * weights_daily.fillna(0.0)).sum(
        axis=1
    )
    port_ret = port_ret.dropna()
    port_ret.name = "portfolio_ret"
    return port_ret


def equal_weight_portfolio(
    returns_long: pd.DataFrame, params: StrategyParams
) -> pd.Series:
    """
    returns_long: [date, fund_code, ret]
    Output: portfolio daily return series indexed by date
    """
    returns_wide = returns_long.pivot_table(
        index="date", columns="fund_code", values="ret", aggfunc="mean"
    ).sort_index()
    weights_daily, _ = build_weights(returns_wide, params)
    return compute_portfolio_returns(returns_wide, weights_daily)
