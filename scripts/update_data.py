from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.normalize import normalize_prices
from src.data.provider_tefaslib import TefasLibProvider
from src.types import FetchParams


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
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


def _fetch_window(
    provider: TefasLibProvider, window: DateWindow, universe: str
) -> pd.DataFrame:
    params = FetchParams(
        start=_format_tefas_date(window.start),
        end=_format_tefas_date(window.end),
        universe=universe,
    )
    return provider.fetch_prices(params)


def _write_metadata(
    metadata_path: Path, df: pd.DataFrame, window: DateWindow
) -> None:
    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(df)),
        "date_range_start": window.start.isoformat(),
        "date_range_end": window.end.isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    today = date.today()
    start_date = _subtract_years(today, 3)
    window = DateWindow(start=start_date, end=today)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "tefas_master_3y.parquet"
    metadata_path = data_dir / "metadata.json"

    provider = TefasLibProvider()
    universe = "all"

    try:
        chunks: list[pd.DataFrame] = []
        for chunk_window in _build_yearly_windows(window):
            chunk_df = _fetch_window(provider, chunk_window, universe)
            if chunk_df.empty:
                continue
            chunks.append(chunk_df)

        if not chunks:
            raise RuntimeError("TEFAS verisi boş geldi.")

        combined = pd.concat(chunks, ignore_index=True)
        combined = normalize_prices(combined)
        if "category" not in combined.columns:
            combined["category"] = universe

        combined.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        _write_metadata(metadata_path, combined, window)
    except Exception as exc:
        print(f"Veri güncelleme hatası: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
