# Python 3.12
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class Filters:
    # Date range (YYYY-MM-DD)
    start_date: str = ""
    end_date: str = ""

    # Pairing windows
    l1_window: str = "60s"
    l2_window: str = "10m"
    l3_window: str = "10m"

    # L3 specific
    delta_qty: int = 1

    # Toggles
    enable_L1: bool = True
    enable_L2: bool = True
    enable_L3: bool = False   # 🔴 OFF por defecto (clave para performance)


@dataclass
class AppState:
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None

    df_master: Optional[pd.DataFrame] = None

    pairs_L1: Optional[pd.DataFrame] = None
    groups_L2: Optional[pd.DataFrame] = None
    pairs_L3: Optional[pd.DataFrame] = None

    meta: Dict[str, Any] = field(default_factory=dict)
    filters: Filters = field(default_factory=Filters)

    def clear_results(self) -> None:
        self.df_master = None
        self.pairs_L1 = None
        self.groups_L2 = None
        self.pairs_L3 = None
