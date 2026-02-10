# src/data/loader.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MASTER_PATH = DATA_DIR / "tefas_master_3y.parquet"
METADATA_PATH = DATA_DIR / "metadata.json"
MACRO_PATH = DATA_DIR / "macro.parquet"

# Minimum row count to consider a parquet file valid (not corrupted/empty)
MIN_VALID_ROWS = 10


def validate_parquet(path: Path, min_rows: int = MIN_VALID_ROWS) -> bool:
    """Check that a parquet file exists, is readable, and has enough rows."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty or len(df) < min_rows:
            logger.warning("Parquet file %s has only %d rows", path, len(df))
            return False
        return True
    except Exception as exc:
        logger.error("Parquet validation failed for %s: %s", path, exc)
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def load_master_data() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        logger.error("Master data file not found: %s", MASTER_PATH)
        return pd.DataFrame()
    try:
        df = pd.read_parquet(MASTER_PATH, engine="pyarrow")
        if df.empty:
            logger.warning("Master data is empty")
        return df
    except Exception as exc:
        logger.error("Failed to read master data: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_data() -> pd.DataFrame:
    if not MACRO_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(MACRO_PATH, engine="pyarrow")
    except Exception as exc:
        logger.error("Failed to read macro data: %s", exc)
        return pd.DataFrame()


# FIX: Renamed from `Youtube()` (typo) to `_load_metadata()`
def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read metadata: %s", exc)
        return {}


def get_metadata() -> dict[str, Any]:
    return _load_metadata()


def filter_data(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fund_codes: list[str],
    fund_types: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")

    # Drop rows where date parsing failed
    filtered = filtered.dropna(subset=["date"])

    filtered = filtered.loc[
        (filtered["date"] >= start_date) & (filtered["date"] <= end_date)
    ]

    if fund_codes:
        filtered = filtered[filtered["fund_code"].isin(fund_codes)]

    if fund_types:
        if "category" in filtered.columns:
            filtered = filtered[filtered["category"].isin(fund_types)]

    return filtered
