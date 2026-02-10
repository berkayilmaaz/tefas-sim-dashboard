# scripts/update_data.py
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.normalize import normalize_prices
from src.data.provider_tefaslib import TefasLibProvider
from src.types import FetchParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "tefas_master_3y.parquet"
METADATA_PATH = DATA_DIR / "metadata.json"
MACRO_PATH = DATA_DIR / "macro.parquet"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds; exponential: 2, 4, 8 …
LOOKBACK_YEARS = 3
MIN_VALID_ROWS = 10  # reject parquet writes with fewer rows than this


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:  # Feb 29
        return value.replace(month=2, day=28, year=value.year - years)


def _format_tefas_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _build_yearly_windows(window: DateWindow) -> Iterable[DateWindow]:
    current_year = window.start.year
    while current_year <= window.end.year:
        chunk_start = date(current_year, 1, 1)
        chunk_end = date(current_year, 12, 31)
        if current_year == window.start.year:
            chunk_start = window.start
        if current_year == window.end.year:
            chunk_end = window.end
        yield DateWindow(start=chunk_start, end=chunk_end)
        current_year += 1


# ---------------------------------------------------------------------------
# Retry-aware fetch
# ---------------------------------------------------------------------------
def _fetch_window_with_retry(
    provider: TefasLibProvider,
    window: DateWindow,
    universe: str,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """
    Fetch a single date window with exponential-backoff retry.

    Returns an empty DataFrame (never raises) so callers can decide
    how to proceed.
    """
    params = FetchParams(
        start=_format_tefas_date(window.start),
        end=_format_tefas_date(window.end),
        universe=universe,
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = provider.fetch_prices(params)
            if df is not None and not df.empty:
                return df
            # API returned empty – not necessarily an error (e.g. holiday range)
            print(
                f"  [Attempt {attempt}] Empty response for "
                f"{window.start} → {window.end}",
                file=sys.stderr,
            )
        except Exception as exc:
            last_exc = exc
            wait = RETRY_BACKOFF_BASE**attempt
            print(
                f"  [Attempt {attempt}/{max_retries}] Error: {exc}. "
                f"Retrying in {wait}s …",
                file=sys.stderr,
            )
            time.sleep(wait)

    if last_exc:
        print(
            f"  FAILED after {max_retries} attempts for "
            f"{window.start} → {window.end}: {last_exc}",
            file=sys.stderr,
        )
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Incremental load helpers
# ---------------------------------------------------------------------------
def _load_existing_data() -> tuple[pd.DataFrame | None, date | None]:
    """
    Load the existing parquet + metadata to determine what we already have.

    Returns (existing_df, last_date) or (None, None) when there is no
    usable cached data.
    """
    if not PARQUET_PATH.exists() or not METADATA_PATH.exists():
        return None, None

    try:
        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
        if df.empty or len(df) < MIN_VALID_ROWS:
            return None, None

        meta = json.loads(METADATA_PATH.read_text())
        last_end_str = meta.get("date_range_end")
        if not last_end_str:
            return None, None
        last_end = date.fromisoformat(last_end_str)
        return df, last_end

    except Exception as exc:
        print(f"  Could not load existing data: {exc}", file=sys.stderr)
        return None, None


def _validate_dataframe(df: pd.DataFrame, label: str = "data") -> bool:
    """Basic sanity checks before we persist."""
    if df.empty:
        print(f"  Validation FAILED: {label} is empty.", file=sys.stderr)
        return False
    if len(df) < MIN_VALID_ROWS:
        print(
            f"  Validation FAILED: {label} has only {len(df)} rows "
            f"(min {MIN_VALID_ROWS}).",
            file=sys.stderr,
        )
        return False

    required = {"date", "fund_code", "price"}
    present = set(df.columns)
    missing = required - present
    if missing:
        print(
            f"  Validation FAILED: {label} missing columns {missing}.",
            file=sys.stderr,
        )
        return False

    # Price sanity: at least some positive prices
    numeric_prices = pd.to_numeric(df["price"], errors="coerce")
    if numeric_prices.dropna().empty or (numeric_prices.dropna() <= 0).all():
        print(f"  Validation FAILED: {label} has no positive prices.", file=sys.stderr)
        return False

    return True


def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Atomic-ish write: write to a temp file first, then rename.
    Prevents corruption if the process is killed mid-write.
    """
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------
def _fetch_usdtry(window: DateWindow) -> pd.Series:
    import yfinance as yf

    fx = yf.download(
        "USDTRY=X",
        start=window.start.isoformat(),
        end=window.end.isoformat(),
        progress=False,
    )
    if fx.empty or "Close" not in fx.columns:
        raise RuntimeError("USD/TRY verisi alınamadı.")
    usdtry = fx["Close"].rename("usdtry")
    usdtry.index = pd.to_datetime(usdtry.index).normalize()
    return usdtry.sort_index()


def _fetch_cpi(window: DateWindow) -> pd.DataFrame:
    from pandas_datareader import data as web

    cpi = web.DataReader(
        ["CPIAUCSL", "TURCPIALLMINMEI"],
        "fred",
        start=window.start,
        end=window.end,
    )
    if cpi.empty:
        raise RuntimeError("FRED CPI verisi alınamadı.")
    cpi = cpi.rename(columns={"CPIAUCSL": "cpi_us", "TURCPIALLMINMEI": "cpi_tr"})
    cpi.index = pd.to_datetime(cpi.index).normalize()
    return cpi.sort_index()


def _build_macro_dataset(window: DateWindow) -> pd.DataFrame | None:
    try:
        usdtry = _fetch_usdtry(window)
        cpi = _fetch_cpi(window)
    except Exception as exc:
        print(f"  Makro veri hatası: {exc}", file=sys.stderr)
        return None

    daily_index = pd.date_range(
        start=usdtry.index.min(), end=usdtry.index.max(), freq="D"
    )
    usdtry = usdtry.reindex(daily_index).ffill()
    cpi = cpi.reindex(daily_index).ffill()

    macro = pd.concat([usdtry, cpi], axis=1).reset_index()
    macro = macro.rename(columns={"index": "date"})
    return macro


# ---------------------------------------------------------------------------
# Metadata writer
# ---------------------------------------------------------------------------
def _write_metadata(
    metadata_path: Path,
    df: pd.DataFrame,
    macro_rows: int,
    window: DateWindow,
    is_incremental: bool,
) -> None:
    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(df)),
        "macro_rows": int(macro_rows),
        "date_range_start": window.start.isoformat(),
        "date_range_end": window.end.isoformat(),
        "update_type": "incremental" if is_incremental else "full",
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> int:
    today = date.today()
    full_start = _subtract_years(today, LOOKBACK_YEARS)
    full_window = DateWindow(start=full_start, end=today)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    provider = TefasLibProvider()
    universe = "all"

    # ------------------------------------------------------------------
    # INCREMENTAL LOGIC: reuse existing data if possible
    # ------------------------------------------------------------------
    existing_df, last_date = _load_existing_data()
    is_incremental = False
    fetch_window = full_window

    if existing_df is not None and last_date is not None:
        gap_days = (today - last_date).days
        if gap_days <= 1:
            print("Data is already up-to-date. Nothing to fetch.")
            return 0
        # Only fetch from the day AFTER the last date we have
        incremental_start = last_date + timedelta(days=1)
        if incremental_start <= today:
            fetch_window = DateWindow(start=incremental_start, end=today)
            is_incremental = True
            print(
                f"Incremental update: fetching {incremental_start} → {today} "
                f"({gap_days} day gap)"
            )
        else:
            print("Data is already up-to-date.")
            return 0
    else:
        print(f"Full download: {full_start} → {today}")

    # ------------------------------------------------------------------
    # FETCH
    # ------------------------------------------------------------------
    try:
        chunks: list[pd.DataFrame] = []
        for chunk_window in _build_yearly_windows(fetch_window):
            print(f"  Fetching {chunk_window.start} → {chunk_window.end} …")
            chunk_df = _fetch_window_with_retry(provider, chunk_window, universe)
            if chunk_df.empty:
                print(f"    ⚠ No data for this window (skipping).")
                continue
            chunks.append(chunk_df)

        if not chunks and not is_incremental:
            print("ERROR: TEFAS returned no data at all.", file=sys.stderr)
            return 1

        new_data = pd.DataFrame()
        if chunks:
            new_data = pd.concat(chunks, ignore_index=True)
            new_data = normalize_prices(new_data)
            if "category" not in new_data.columns:
                new_data["category"] = universe

        # ------------------------------------------------------------------
        # MERGE with existing data (incremental) or use new data (full)
        # ------------------------------------------------------------------
        if is_incremental and existing_df is not None:
            if new_data.empty:
                print("  No new rows fetched. Keeping existing data.")
                combined = existing_df
            else:
                combined = pd.concat([existing_df, new_data], ignore_index=True)
                # De-dup: keep latest row for each (date, fund_code) pair
                combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
                combined = combined.drop_duplicates(
                    subset=["date", "fund_code"], keep="last"
                )
                combined = combined.sort_values(["fund_code", "date"]).reset_index(
                    drop=True
                )
        else:
            combined = new_data

        # ------------------------------------------------------------------
        # TRIM to lookback window (drop rows older than LOOKBACK_YEARS)
        # ------------------------------------------------------------------
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        cutoff = pd.Timestamp(full_start)
        combined = combined[combined["date"] >= cutoff].reset_index(drop=True)

        # ------------------------------------------------------------------
        # VALIDATE before writing
        # ------------------------------------------------------------------
        if not _validate_dataframe(combined, "combined TEFAS data"):
            print(
                "ERROR: Validation failed. Existing file NOT overwritten.",
                file=sys.stderr,
            )
            return 1

        _safe_write_parquet(combined, PARQUET_PATH)
        print(f"  ✓ Wrote {len(combined):,} rows to {PARQUET_PATH}")

        # ------------------------------------------------------------------
        # MACRO (best-effort, non-blocking)
        # ------------------------------------------------------------------
        macro_df = _build_macro_dataset(full_window)
        macro_rows = 0
        if macro_df is not None and not macro_df.empty:
            _safe_write_parquet(macro_df, MACRO_PATH)
            macro_rows = len(macro_df)
            print(f"  ✓ Macro data: {macro_rows:,} rows")

        _write_metadata(
            METADATA_PATH, combined, macro_rows, full_window, is_incremental
        )
        print("  ✓ Metadata written.")

    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
