# src/data/normalize.py
from __future__ import annotations

import pandas as pd

REQUIRED = ["date", "fund_code", "fund_name", "price"]
OPTIONAL = [
    "shares_outstanding",
    "investor_count",
    "aum",
    "category",
    "allocation_stock",
    "allocation_bond",
    "allocation_cash",
    "allocation_fx",
    "allocation_fund",
]

# TEFAS → canonical mapping
COLUMN_MAP = {
    # Temel Alanlar
    "Tarih": "date",
    "Fon Kodu": "fund_code",
    "Fon Adı": "fund_name",
    "Fiyat": "price",
    "Birim Pay Değeri": "price",
    "Tedavüldeki Pay Sayısı": "shares_outstanding",
    "Kişi Sayısı": "investor_count",
    "Fon Toplam Değer": "aum",
    # Varlık Dağılımı (Asset Allocation) Eşleştirmeleri
    "Hisse Senedi": "allocation_stock",
    "Yabancı Hisse Senedi": "allocation_stock",
    "Devlet Tahvili": "allocation_bond",
    "Özel Sektör Tahvili": "allocation_bond",
    "Tahvil": "allocation_bond",
    "Bono": "allocation_bond",
    "Ters Repo": "allocation_cash",
    "Takasbank Para Piyasası": "allocation_cash",
    "Mevduat": "allocation_cash",
    "Eurobond": "allocation_fx",
    "Kıymetli Madenler": "allocation_fx",
    "Yatırım Fonları": "allocation_fund",
    "Yatırım Fonu Katılma Payları": "allocation_fund",
}


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelen veriyi standart şemaya ve tiplere zorlar.
    Mükerrer kayıtları (aynı gün, aynı fon) temizler.
    """
    df = df.copy()

    # 1️⃣ Türkçe → canonical kolon isimleri
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2️⃣ Bazen fund_cc gibi dönmüş olabilir → fund_code’a çek
    if "fund_cc" in df.columns and "fund_code" not in df.columns:
        df = df.rename(columns={"fund_cc": "fund_code"})

    # 3️⃣ Zorunlu (Required) kolonlar var mı?
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Got: {df.columns.tolist()}"
        )

    # 4️⃣ Tip normalizasyonu
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce", dayfirst=True
    ).dt.normalize()
    df["fund_code"] = df["fund_code"].astype(str)
    df["fund_name"] = df["fund_name"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 5️⃣ Opsiyonel kolonlar
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 6️⃣ Temizlik ve Mükerrer Kayıt Kontrolü (KRİTİK DÜZELTME)
    df = df.dropna(subset=["date", "price"])

    # Aynı gün aynı fon kodu varsa, sonuncusunu tut (duplicate cleanup)
    df = df.drop_duplicates(subset=["date", "fund_code"], keep="last")

    df = df.sort_values(["fund_code", "date"]).reset_index(drop=True)

    return df
