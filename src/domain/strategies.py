# src/domain/strategies.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.types import StrategyParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monte Carlo result container
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation outputs."""

    simulated_returns: np.ndarray  # (n_sims,) array of portfolio total returns
    strategy_return: float  # the actual strategy return for comparison
    percentile_rank: float  # % of random portfolios beaten (0-100)
    median_random: float
    p5_random: float
    p95_random: float


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _momentum_scores(window: pd.DataFrame) -> pd.Series:
    return (1.0 + window).prod(skipna=True) - 1.0


def _low_vol_scores(window: pd.DataFrame) -> pd.Series:
    return window.std(skipna=True)


def _sharpe_scores(window: pd.DataFrame) -> pd.Series:
    mean = window.mean(skipna=True)
    std = window.std(skipna=True)
    scores = mean / std.replace(0.0, np.nan)
    return scores


def _risk_parity_weights(window: pd.DataFrame) -> pd.Series:
    vol = window.std(skipna=True)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    inv_vol = inv_vol.dropna()
    if inv_vol.empty:
        return pd.Series(dtype=float)
    weights = inv_vol / inv_vol.sum()
    return weights


def _min_variance_weights(window: pd.DataFrame) -> pd.Series:
    cov = window.cov()
    if cov.empty:
        return pd.Series(dtype=float)
    try:
        inv_cov = np.linalg.pinv(cov.values)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix inversion failed, returning empty weights")
        return pd.Series(dtype=float)
    ones = np.ones(inv_cov.shape[0])
    raw = inv_cov @ ones
    if np.all(np.isnan(raw)):
        return pd.Series(dtype=float)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() == 0.0:
        return pd.Series(dtype=float)
    weights = raw / raw.sum()
    return pd.Series(weights, index=cov.columns)


def _select_top_k(scores: pd.Series, k: int, low_is_best: bool) -> list[str]:
    scores = scores.dropna()
    if scores.empty:
        return []
    if low_is_best:
        return scores.nsmallest(k).index.tolist()
    return scores.nlargest(k).index.tolist()


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------
def build_weights(
    returns_wide: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns_wide.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = pd.DatetimeIndex(returns_wide.index)
    rebalance_dates = compute_rebalance_dates(dates, params.rebalance)
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}

    for date in rebalance_dates:
        idx = dates.get_loc(date)
        if params.strategy == "equal_weight":
            selected = returns_wide.columns.tolist()
            weights = None
        else:
            if idx < params.lookback:
                continue
            window = returns_wide.iloc[idx - params.lookback : idx]
            if params.strategy == "momentum_top_k":
                scores = _momentum_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
                weights = None
            elif params.strategy == "low_vol_top_k":
                scores = _low_vol_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=True)
                weights = None
            elif params.strategy == "sharpe_top_k":
                scores = _sharpe_scores(window)
                selected = _select_top_k(scores, params.k, low_is_best=False)
                weights = None
            elif params.strategy == "risk_parity":
                weights = _risk_parity_weights(window)
                selected = weights.index.tolist()
            elif params.strategy == "min_variance":
                weights = _min_variance_weights(window)
                selected = weights.index.tolist()
            else:
                raise ValueError(f"Unknown strategy: {params.strategy}")

        if not selected:
            continue
        if weights is None:
            weight = 1.0 / len(selected)
            weights_by_date[pd.Timestamp(date)] = pd.Series(
                weight, index=selected, name="weight"
            )
        else:
            weights_by_date[pd.Timestamp(date)] = weights.rename("weight")

    if not weights_by_date:
        return pd.DataFrame(), pd.DataFrame()

    weights_daily = pd.DataFrame(index=dates, columns=returns_wide.columns, dtype=float)
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
    port_ret = (aligned_returns.fillna(0.0) * weights_daily.fillna(0.0)).sum(axis=1)
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
    if returns_long.empty:
        return pd.Series(dtype=float, name="portfolio_ret")
    returns_wide = returns_long.pivot_table(
        index="date", columns="fund_code", values="ret", aggfunc="mean"
    ).sort_index()
    weights_daily, _ = build_weights(returns_wide, params)
    return compute_portfolio_returns(returns_wide, weights_daily)


# ---------------------------------------------------------------------------
# Asset-based backtest engine
# ---------------------------------------------------------------------------
def backtest_portfolio_assets(
    prices: pd.DataFrame, weights_table: pd.DataFrame, initial_capital: float
) -> pd.DataFrame:
    """
    Realistic backtest engine (Asset/Share based).

    Between rebalance dates the portfolio value drifts with prices
    (shares * current_price).  On rebalance dates the portfolio is
    re-allocated to target weights at zero transaction cost.

    Args:
        prices: Wide-format prices [index=date, columns=fund_code].
        weights_table: [rebalance_date, fund_code, weight].
        initial_capital: Starting capital in TL.

    Returns:
        pd.DataFrame with columns [equity, ret], indexed by date.
        Returns empty DataFrame on invalid inputs.
    """
    if weights_table.empty or prices.empty:
        return pd.DataFrame()

    if initial_capital <= 0:
        logger.warning("initial_capital must be > 0, got %s", initial_capital)
        return pd.DataFrame()

    # Sort rebalance dates
    rebalance_dates = sorted(weights_table["rebalance_date"].unique())

    # Start from first rebalance date
    start_date = rebalance_dates[0]
    prices = prices.loc[start_date:].copy()
    if prices.empty:
        return pd.DataFrame()

    dates = prices.index

    # Pivot target weights
    target_weights_df = weights_table.pivot(
        index="rebalance_date", columns="fund_code", values="weight"
    ).fillna(0.0)

    # Ensure all price columns exist in weights (fill missing with 0)
    for col in prices.columns:
        if col not in target_weights_df.columns:
            target_weights_df[col] = 0.0

    # Simulation state
    current_cash = initial_capital
    current_shares = pd.Series(0.0, index=prices.columns)
    p_values = {}
    reb_set = set(rebalance_dates)

    for d in dates:
        p = prices.loc[d].fillna(0.0)

        # Portfolio value BEFORE any rebalance
        val_assets = (current_shares * p).sum()
        total_value = current_cash + val_assets

        # Guard against degenerate values
        if np.isnan(total_value) or total_value < 0:
            total_value = 0.0

        # Rebalance if scheduled
        if d in reb_set and d in target_weights_df.index:
            w = target_weights_df.loc[d].reindex(prices.columns, fill_value=0.0)
            target_amounts = total_value * w

            # shares = amount / price; guard against division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                new_shares = target_amounts / p
            new_shares = new_shares.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            current_shares = new_shares
            current_cash = 0.0  # fully invested assumption

            # Recalculate after rebalance (sanity)
            val_assets = (current_shares * p).sum()
            total_value = current_cash + val_assets

        p_values[d] = total_value

    if not p_values:
        return pd.DataFrame()

    result = pd.Series(p_values, name="equity")
    df_res = pd.DataFrame(result)
    df_res["ret"] = df_res["equity"].pct_change().fillna(0.0)
    return df_res


# ---------------------------------------------------------------------------
# Monte Carlo simulation  (NEW)
# ---------------------------------------------------------------------------
def run_monte_carlo_simulation(
    prices_wide: pd.DataFrame,
    actual_k: int,
    strategy_total_return: float,
    n_sims: int = 1_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Vectorised Monte Carlo: randomly pick *actual_k* funds, compute
    equal-weight buy-and-hold total return, repeat *n_sims* times.

    Handles NaNs gracefully (funds that start on different dates):
    - Per-fund total return is computed only over the fund's own valid
      price range (first non-NaN to last non-NaN).
    - Funds with < 2 valid price observations are excluded.

    Args:
        prices_wide: Wide prices [index=date, columns=fund_code].
        actual_k:    Number of funds to pick per random portfolio.
        strategy_total_return: The real strategy's total return (for ranking).
        n_sims:      Number of Monte Carlo iterations.
        seed:        Optional RNG seed for reproducibility.

    Returns:
        MonteCarloResult dataclass.
    """
    if prices_wide.empty or actual_k < 1:
        return MonteCarloResult(
            simulated_returns=np.array([]),
            strategy_return=strategy_total_return,
            percentile_rank=np.nan,
            median_random=np.nan,
            p5_random=np.nan,
            p95_random=np.nan,
        )

    # ------------------------------------------------------------------
    # 1. Compute per-fund total return (handles NaN start dates)
    # ------------------------------------------------------------------
    fund_returns = {}
    for fund in prices_wide.columns:
        series = prices_wide[fund].dropna()
        if len(series) < 2:
            continue
        first_price = series.iloc[0]
        last_price = series.iloc[-1]
        if first_price == 0 or np.isnan(first_price):
            continue
        fund_returns[fund] = (last_price / first_price) - 1.0

    if not fund_returns:
        return MonteCarloResult(
            simulated_returns=np.array([]),
            strategy_return=strategy_total_return,
            percentile_rank=np.nan,
            median_random=np.nan,
            p5_random=np.nan,
            p95_random=np.nan,
        )

    valid_returns = np.array(list(fund_returns.values()), dtype=np.float64)
    n_funds = len(valid_returns)
    draw_k = min(actual_k, n_funds)

    # ------------------------------------------------------------------
    # 2. Vectorised random sampling
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # Build (n_sims, draw_k) index matrix via argsort trick on uniform randoms
    # This is much faster than a Python loop.
    rand_matrix = rng.random((n_sims, n_funds))
    # For each row, the first `draw_k` indices of argsort give a random
    # sample without replacement.
    idx_matrix = np.argpartition(rand_matrix, draw_k, axis=1)[:, :draw_k]

    # Gather returns and compute equal-weight portfolio return per sim
    sampled_returns = valid_returns[idx_matrix]  # (n_sims, draw_k)
    sim_portfolio_returns = sampled_returns.mean(axis=1)  # (n_sims,)

    # ------------------------------------------------------------------
    # 3. Statistics
    # ------------------------------------------------------------------
    percentile_rank = float(
        (strategy_total_return > sim_portfolio_returns).mean() * 100
    )
    median_random = float(np.median(sim_portfolio_returns))
    p5_random = float(np.percentile(sim_portfolio_returns, 5))
    p95_random = float(np.percentile(sim_portfolio_returns, 95))

    return MonteCarloResult(
        simulated_returns=sim_portfolio_returns,
        strategy_return=strategy_total_return,
        percentile_rank=percentile_rank,
        median_random=median_random,
        p5_random=p5_random,
        p95_random=p95_random,
    )
