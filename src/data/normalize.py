# src/data/normalize.py
from __future__ import annotations

import pandas as pd

REQUIRED = ["date", "fund_code", "fund_name", "price"]


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force normalized schema and types.
    """
    df = df.copy()

    # bazen fund_cc gibi dönmüş olabilir → fund_code’a çek
    if "fund_cc" in df.columns and "fund_code" not in df.columns:
        df = df.rename(columns={"fund_cc": "fund_code"})

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Got: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["fund_code"] = df["fund_code"].astype(str)
    df["fund_name"] = df["fund_name"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # optional kolonlar yoksa ekle (pipeline kırılmasın)
    for col in ["shares_outstanding", "investor_count", "aum", "category"]:
        if col not in df.columns:
            df[col] = None

    df = df.dropna(subset=["price"])
    df = df.sort_values(["fund_code", "date"]).reset_index(drop=True)

    return df
