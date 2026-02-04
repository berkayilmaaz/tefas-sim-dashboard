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
    metadata_path: Path, df: pd.DataFrame, macro_rows: int, window: DateWindow
) -> None:
    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(df)),
        "macro_rows": int(macro_rows),
        "date_range_start": window.start.isoformat(),
        "date_range_end": window.end.isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


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
    cpi = cpi.rename(
        columns={
            "CPIAUCSL": "cpi_us",
            "TURCPIALLMINMEI": "cpi_tr",
        }
    )
    cpi.index = pd.to_datetime(cpi.index).normalize()
    return cpi.sort_index()


def _build_macro_dataset(window: DateWindow) -> pd.DataFrame | None:
    try:
        usdtry = _fetch_usdtry(window)
        cpi = _fetch_cpi(window)
    except Exception as exc:
        print(f"Makro veri hatası: {exc}", file=sys.stderr)
        return None

    daily_index = pd.date_range(
        start=usdtry.index.min(), end=usdtry.index.max(), freq="D"
    )
    usdtry = usdtry.reindex(daily_index).ffill()
    cpi = cpi.reindex(daily_index).ffill()

    macro = pd.concat([usdtry, cpi], axis=1).reset_index()
    macro = macro.rename(columns={"index": "date"})
    return macro


def main() -> int:
    today = date.today()
    start_date = _subtract_years(today, 3)
    window = DateWindow(start=start_date, end=today)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "tefas_master_3y.parquet"
    metadata_path = data_dir / "metadata.json"
    macro_path = data_dir / "macro.parquet"

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
        macro_df = _build_macro_dataset(window)
        macro_rows = 0
        if macro_df is not None and not macro_df.empty:
            macro_df.to_parquet(
                macro_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            macro_rows = len(macro_df)
        _write_metadata(metadata_path, combined, macro_rows, window)
    except Exception as exc:
        print(f"Veri güncelleme hatası: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
