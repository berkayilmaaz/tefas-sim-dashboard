# src/domain/stats.py
from __future__ import annotations

import numpy as np
import pandas as pd


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
    # rf = yıllık risksiz; sprint2: 0
    excess = daily_ret - (rf / trading_days)
    denom = excess.std(ddof=1)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((excess.mean() / denom) * np.sqrt(trading_days))
