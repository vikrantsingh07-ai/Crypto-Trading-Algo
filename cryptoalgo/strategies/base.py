"""Strategy interface.

``signals(df)`` is VECTORISED and CAUSAL: row t is the decision made at the close
of bar t using only data <= t. The engine fills at the open of bar t+1. The same
function is used by the backtester (whole history at once) and the paper runner
(rolling window, last row).

Output columns
  signal      +1 enter long / -1 enter short / 0 nothing
  stop_dist   initial stop distance in PRICE units (applied to the fill price)
  tp_dist     take-profit distance in price units (NaN = none)
  trail_dist  chandelier trailing distance in price units (NaN = none)
  exit_long / exit_short   bool: close an open long/short at next open
  max_hold    time stop in bars (NaN = none)
  regime      label that produced the decision
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

SIGNAL_COLS = ["signal", "stop_dist", "tp_dist", "trail_dist", "exit_long", "exit_short", "max_hold", "regime"]


def empty_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"signal": 0, "stop_dist": np.nan, "tp_dist": np.nan, "trail_dist": np.nan,
                         "exit_long": False, "exit_short": False, "max_hold": np.nan, "regime": ""}, index=index)


class Strategy:
    name = "base"
    defaults: dict = {}
    grid: dict = {}          # param -> list of values; pre-declared, small

    def __init__(self, params: dict | None = None, base_tf: str = "1h", htf: str = "4h",
                 use_htf_confirmation: bool = False):
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError(f"unknown params for {self.name}: {sorted(unknown)}")
        self.params = {**self.defaults, **(params or {})}
        self.base_tf, self.htf, self.use_htf = base_tf, htf, use_htf_confirmation

    # -- to be implemented -------------------------------------------------
    def signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @property
    def min_bars(self) -> int:
        return 300

    def with_params(self, **kw) -> "Strategy":
        return type(self)({**self.params, **kw}, self.base_tf, self.htf, self.use_htf)

    @classmethod
    def param_grid(cls) -> list[dict]:
        keys = list(cls.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*(cls.grid[k] for k in keys))]

    def neighbours(self) -> list[dict]:
        """Single-parameter perturbations (~ -30% / +30%) of the current params, for sensitivity tests."""
        out = []
        for k, v in self.params.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            for f in (0.7, 1.3):
                nv = v * f
                nv = max(2, int(round(nv))) if isinstance(v, int) else round(nv, 4)
                if nv != v:
                    out.append({k: nv})
        return out
