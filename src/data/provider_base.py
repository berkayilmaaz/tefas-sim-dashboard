# src/data/provider_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from src.types import FetchParams


class ProviderBase(ABC):
    name: str

    @abstractmethod
    def fetch_prices(self, params: FetchParams) -> pd.DataFrame:
        """
        Must return normalized schema:
        [date, fund_code, fund_name, price, shares_outstanding, investor_count, aum, category]
        """
        raise NotImplementedError
