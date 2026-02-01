# src/domain/returns.py
from __future__ import annotations

import pandas as pd


def compute_fund_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Input: normalized long prices
    Output: long returns with columns: [date, fund_code, ret]
    """
    df = prices[["date", "fund_code", "price"]].copy()
    df = df.sort_values(["fund_code", "date"])

    df["ret"] = df.groupby("fund_code")["price"].pct_change()
    df = df.dropna(subset=["ret"])

    return df[["date", "fund_code", "ret"]]
