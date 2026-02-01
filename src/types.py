# src/types.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional


Universe = Literal["all", "free"]  # şimdilik


@dataclass(frozen=True)
class FetchParams:
    start: str  # "dd.mm.yyyy" (tefasfon formatı)
    end: str    # "dd.mm.yyyy"
    universe: Universe = "free"
    fund_type_code: int = 4  # 4: serbest fonlar (tefasfon)
    tab_code: int = 0        # default sekme


@dataclass(frozen=True)
class StrategyParams:
    k: int = 100  # kaç fon seçilecek
    seed: int = 42
    rebalance: Literal["none"] = "none"  # sprint2: tek dönem, rebalance yok


@dataclass(frozen=True)
class PortfolioResult:
    equity: "object"  # pd.Series
    daily_returns: "object"  # pd.Series
