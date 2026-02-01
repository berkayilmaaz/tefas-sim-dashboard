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
            # cached is already normalized when we save it, but normalize again is safe
            return normalize_prices(cached, category=self.name)

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

        # tefasfon usually returns:
        # ['date','fund_code','fund_name','Fiyat','Tedavüldeki Pay Sayısı','Kişi Sayısı','Fon Toplam Değer']
        # normalize_prices will map 'Fiyat' -> 'price' etc.
        out = normalize_prices(df, category=params.universe)

        save_parquet(cache_key, out)
        return out

    @staticmethod
    def _fund_type_code(universe: str) -> int:
        """
        Map our universe name -> tefasfon fund_type_code.

        NOTE: You saw valid codes: [0,1,2,3,4]
        We'll start with a sane default:
        - "serbest": 4 (your assumption; if wrong we'll brute-force map later)
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
        If you want more tabs later, we can expose this in UI.
        """
        return 0
