# src/domain/stats.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, kurtosis


def equity_curve(daily_ret: pd.Series, start_value: float = 1.0) -> pd.Series:
    eq = (1.0 + daily_ret).cumprod() * start_value
    eq.name = "equity"
    return eq


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    dd.name = "drawdown"
    return dd


def annualized_vol(daily_ret: pd.Series, trading_days: int = 252) -> float:
    return float(daily_ret.std(ddof=1) * np.sqrt(trading_days))


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def sharpe(daily_ret: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    excess = daily_ret - (rf / trading_days)
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((excess.mean() / denom) * np.sqrt(trading_days))


def rolling_volatility(
    daily_ret: pd.Series, window: int, trading_days: int = 252
) -> pd.Series:
    rolling_std = daily_ret.rolling(window=window, min_periods=window).std(ddof=1)
    return rolling_std * np.sqrt(trading_days)


def rolling_sharpe(
    daily_ret: pd.Series, window: int, trading_days: int = 252
) -> pd.Series:
    rolling_mean = daily_ret.rolling(window=window, min_periods=window).mean()
    rolling_std = daily_ret.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe_series = rolling_mean / rolling_std.replace(0.0, np.nan)
    return sharpe_series * np.sqrt(trading_days)


def rolling_max_drawdown(daily_ret: pd.Series, window: int) -> pd.Series:
    def _window_mdd(values: np.ndarray) -> float:
        equity = np.cumprod(1.0 + values)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        return float(np.min(dd))

    return daily_ret.rolling(window=window, min_periods=window).apply(
        _window_mdd, raw=True
    )


def rolling_metrics(
    daily_ret: pd.Series, window: int, trading_days: int = 252
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rolling_vol": rolling_volatility(daily_ret, window, trading_days),
            "rolling_sharpe": rolling_sharpe(daily_ret, window, trading_days),
            "rolling_max_drawdown": rolling_max_drawdown(daily_ret, window),
        }
    )


def check_normality(daily_ret: pd.Series) -> dict:
    data = daily_ret.dropna()
    if len(data) < 3:
        return {
            "is_normal": False,
            "p_value": 0.0,
            "statistic": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
        }
    stat, p_value = shapiro(data)
    s = skew(data)
    k = kurtosis(data)
    return {
        "is_normal": p_value > 0.05,
        "p_value": p_value,
        "statistic": stat,
        "skew": s,
        "kurtosis": k,
    }


# ===========================================================================
# PHYSICS-INSPIRED METRICS
# ===========================================================================


def hurst_exponent(series: pd.Series, max_lag: int | None = None) -> float:
    """
    Rescaled Range (R/S) analysis to estimate the Hurst exponent.

    H ≈ 0.5  → Random walk (Brownian motion, efficient market)
    H > 0.5  → Persistent / trending (momentum)
    H < 0.5  → Anti-persistent / mean-reverting

    This is analogous to classifying the "diffusion regime" of a
    stochastic process — sub-diffusive, normal, or super-diffusive.
    """
    data = series.dropna().values
    n = len(data)
    if n < 20:
        return float("nan")

    if max_lag is None:
        max_lag = min(n // 2, 100)
    max_lag = max(max_lag, 10)

    lags = range(10, max_lag + 1)
    rs_values = []
    lag_values = []

    for lag in lags:
        rs_list = []
        for start in range(0, n - lag, lag):
            chunk = data[start : start + lag]
            mean_chunk = chunk.mean()
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            lag_values.append(lag)

    if len(lag_values) < 3:
        return float("nan")

    log_lags = np.log(lag_values)
    log_rs = np.log(rs_values)

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(coeffs[0])


def shannon_entropy(daily_ret: pd.Series, n_bins: int = 50) -> float:
    """
    Shannon entropy of the return distribution.

    Higher entropy → more "disordered" / unpredictable returns.
    Lower entropy  → more concentrated / predictable pattern.

    From statistical mechanics: S = -Σ p_i * ln(p_i)
    """
    data = daily_ret.dropna()
    if len(data) < 10:
        return float("nan")

    counts, _ = np.histogram(data, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log(probs)))


def mean_reversion_halflife(series: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life estimation.

    Fits: ΔP(t) = θ * (μ - P(t-1)) + ε
    Half-life = -ln(2) / ln(1 + θ)

    Shorter half-life → faster mean reversion (good for contrarian strategies).
    Infinite / negative → trending or random walk.

    Analogous to the relaxation time of a damped harmonic oscillator.
    """
    data = series.dropna().values
    if len(data) < 20:
        return float("nan")

    y = np.diff(data)  # ΔP
    x = data[:-1]  # P(t-1)

    # OLS: y = a + b*x + ε  →  θ = b
    x_with_const = np.column_stack([np.ones_like(x), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")

    theta = coeffs[1]

    if theta >= 0:
        return float("inf")  # trending, no mean reversion

    half_life = -np.log(2) / np.log(1 + theta)
    return float(half_life)


def variance_ratio_test(daily_ret: pd.Series, holding_period: int = 5) -> dict:
    """
    Lo-MacKinlay Variance Ratio Test.

    Under random walk (efficient market): VR(q) ≈ 1
    VR > 1 → Positive autocorrelation (momentum / persistence)
    VR < 1 → Negative autocorrelation (mean reversion)

    Physics analogy: tests whether the "diffusion coefficient" is
    scale-invariant (Brownian) or anomalous.
    """
    data = daily_ret.dropna().values
    n = len(data)
    q = holding_period

    if n < 2 * q + 1:
        return {
            "variance_ratio": float("nan"),
            "z_stat": float("nan"),
            "regime": "Yetersiz veri",
        }

    # Variance of 1-period returns
    var_1 = np.var(data, ddof=1)
    if var_1 == 0:
        return {
            "variance_ratio": float("nan"),
            "z_stat": float("nan"),
            "regime": "Sıfır varyans",
        }

    # q-period overlapping returns
    q_returns = np.array([data[i : i + q].sum() for i in range(n - q + 1)])
    var_q = np.var(q_returns, ddof=1)

    vr = var_q / (q * var_1)

    # Asymptotic z-stat (heteroscedasticity-robust)
    nq = n - q + 1
    z_stat = (vr - 1.0) / np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3 * q * nq))

    if vr > 1.05:
        regime = "Momentum (VR > 1)"
    elif vr < 0.95:
        regime = "Mean Reversion (VR < 1)"
    else:
        regime = "Random Walk (VR ≈ 1)"

    return {"variance_ratio": float(vr), "z_stat": float(z_stat), "regime": regime}


def compute_fund_analytics(
    returns_long: pd.DataFrame,
    prices_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute comprehensive per-fund analytics for the fund table.
    Returns a DataFrame with one row per fund.
    """
    if returns_long.empty:
        return pd.DataFrame()

    fund_stats = []
    for fund_code, group in returns_long.groupby("fund_code"):
        rets = group["ret"]
        n_days = len(rets)
        if n_days < 2:
            continue

        total_ret = float((1 + rets).prod() - 1)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0
        sharpe_val = sharpe(rets)
        eq = equity_curve(rets)
        mdd = max_drawdown(eq)
        best_day = float(rets.max())
        worst_day = float(rets.min())
        mean_daily = float(rets.mean())

        # Monthly returns (approx 21 trading days)
        monthly_ret = (
            float((1 + mean_daily) ** 21 - 1) if n_days >= 21 else float("nan")
        )

        # Hurst (from price series if available)
        h = float("nan")
        if fund_code in prices_wide.columns:
            price_series = prices_wide[fund_code].dropna()
            if len(price_series) >= 20:
                h = hurst_exponent(price_series)

        fund_stats.append(
            {
                "fund_code": fund_code,
                "total_return": total_ret,
                "annualized_vol": ann_vol,
                "sharpe": sharpe_val,
                "max_drawdown": mdd,
                "best_day": best_day,
                "worst_day": worst_day,
                "mean_daily_ret": mean_daily,
                "est_monthly_ret": monthly_ret,
                "hurst": h,
                "obs_days": n_days,
            }
        )

    if not fund_stats:
        return pd.DataFrame()

    return pd.DataFrame(fund_stats).sort_values("total_return", ascending=False)
