from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _key_to_path(key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{h}.parquet"


def load_parquet(key: str) -> pd.DataFrame | None:
    path = _key_to_path(key)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_parquet(key: str, df: pd.DataFrame) -> None:
    path = _key_to_path(key)
    df.to_parquet(path)
