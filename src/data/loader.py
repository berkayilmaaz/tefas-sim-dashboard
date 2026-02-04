from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DATA_DIR = Path("data")
MASTER_PATH = DATA_DIR / "tefas_master_3y.parquet"
METADATA_PATH = DATA_DIR / "metadata.json"


@st.cache_data(ttl=3600, show_spinner=False)
def load_master_data() -> pd.DataFrame:
    return pd.read_parquet(MASTER_PATH, engine="pyarrow")


def Youtube() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def get_metadata() -> dict[str, Any]:
    return Youtube()


def filter_data(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fund_codes: list[str],
    fund_types: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered = filtered.loc[
        (filtered["date"] >= start_date) & (filtered["date"] <= end_date)
    ]

    if fund_codes:
        filtered = filtered[filtered["fund_code"].isin(fund_codes)]

    if fund_types:
        if "category" in filtered.columns:
            filtered = filtered[filtered["category"].isin(fund_types)]

    return filtered
