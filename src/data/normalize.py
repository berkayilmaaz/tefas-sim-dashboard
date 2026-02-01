# src/data/normalize.py
from __future__ import annotations

import pandas as pd

REQUIRED = ["date", "fund_code", "fund_name", "price"]

# TEFAS → canonical mapping
COLUMN_MAP = {
    "Tarih": "date",
    "Fon Kodu": "fund_code",
    "Fon Adı": "fund_name",
    "Fiyat": "price",
    "Birim Pay Değeri": "price",
    "Tedavüldeki Pay Sayısı": "shares_outstanding",
    "Kişi Sayısı": "investor_count",
    "Fon Toplam Değer": "aum",
}


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force normalized schema and types.
    """
    df = df.copy()

    # 1️⃣ Türkçe → canonical kolon isimleri
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2️⃣ bazen fund_cc gibi dönmüş olabilir → fund_code’a çek
    if "fund_cc" in df.columns and "fund_code" not in df.columns:
        df = df.rename(columns={"fund_cc": "fund_code"})

    # 3️⃣ required kolonlar var mı?
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Got: {df.columns.tolist()}"
        )

    # 4️⃣ tip normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["fund_code"] = df["fund_code"].astype(str)
    df["fund_name"] = df["fund_name"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 5️⃣ opsiyonel kolonlar yoksa ekle
    for col in ["shares_outstanding", "investor_count", "aum", "category"]:
        if col not in df.columns:
            df[col] = None

    # 6️⃣ temizlik
    df = df.dropna(subset=["date", "price"])
    df = df.sort_values(["fund_code", "date"]).reset_index(drop=True)

    return df
