from __future__ import annotations

import pandas as pd

from src.data.provider_base import FetchParams
from src.data.normalize import normalize_prices
from src.data.cache import load_parquet, save_parquet


class TefasLibProvider:
    """
    Real TEFAS provider using tefasfon.fetch_tefas_data
    """

    name = "tefasfon"

    def fetch_prices(self, params: FetchParams) -> pd.DataFrame:
        cache_key = f"{self.name}|{params.universe}|{params.start}|{params.end}"
        cached = load_parquet(cache_key)
        if cached is not None and len(cached) > 0:
            # Cached data is expected to already be normalized.
            # normalize again is safe BUT don't pass category kwarg.
            if "category" not in cached.columns:
                cached = cached.copy()
                cached["category"] = self._category_label(params.universe)
            return normalize_prices(cached)

        try:
            from tefasfon import fetch_tefas_data
        except Exception as e:
            raise RuntimeError("tefasfon could not be imported") from e

        # tefasfon expects date format: "dd.mm.yyyy"
        # and numeric fund_type_code / tab_code.
        raw = fetch_tefas_data(
            start_date=params.start,  # e.g. "01.01.2025"
            end_date=params.end,  # e.g. "18.01.2025"
            fund_type_code=self._fund_type_code(params.universe),
            tab_code=self._tab_code(params.universe),
        )

        df = pd.DataFrame(raw)

        # Add category as a COLUMN (normalize_prices does not accept category= kwarg)
        if "category" not in df.columns:
            df["category"] = self._category_label(params.universe)

        # normalize_prices will handle mapping TR columns like 'Fiyat' -> 'price'
        out = normalize_prices(df)

        save_parquet(cache_key, out)
        return out

    @staticmethod
    def _category_label(universe: str) -> str:
        # what we store in the 'category' column
        u = (universe or "").strip()
        return u if u else "tefasfon"

    @staticmethod
    def _fund_type_code(universe: str) -> int:
        """
        Map our universe name -> tefasfon fund_type_code.

        Valid codes observed: [0,1,2,3,4]
        We'll start with:
        - "serbest": 4
        - else: 0
        """
        u = (universe or "").lower().strip()
        if u in ["serbest", "free", "serbest_fon"]:
            return 4
        return 0

    @staticmethod
    def _tab_code(universe: str) -> int:
        """
        Map our universe name -> tefasfon tab_code.

        We'll keep default 0 for now.
        """
        return 0
