"""Market-regime labelling.

* ``live_regime`` is causal and may drive trading decisions (trend / range / high_vol).
* ``label_blocks`` is an EX-POST label (bull / bear / sideways x vol tercile) used
  ONLY to report how results differ across market conditions. Never trade on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def live_regime(df: pd.DataFrame, adx_trend: float = 25.0, adx_range: float = 20.0,
                high_vol: float = 2.0) -> pd.Series:
    """Causal per-bar regime: 'high_vol', 'trend_up', 'trend_down', 'range', 'neutral'."""
    a, pdi, mdi = ind.adx(df, 14)
    atr_pct = ind.atr(df, 14) / df["close"]
    vs = ind.vol_state(atr_pct)
    out = pd.Series("neutral", index=df.index, dtype=object)
    out[(a < adx_range)] = "range"
    out[(a >= adx_trend) & (pdi > mdi)] = "trend_up"
    out[(a >= adx_trend) & (pdi <= mdi)] = "trend_down"
    out[vs > high_vol] = "high_vol"
    out[a.isna()] = "warmup"
    return out


def label_blocks(df: pd.DataFrame, block_days: int = 30, trend_thresh: float = 0.10) -> pd.Series:
    """Ex-post label per bar from its non-overlapping calendar block.

    direction: bull if block return > +trend_thresh, bear if < -trend_thresh, else sideways.
    volatility: block realised vol vs. whole-sample terciles -> high_vol / low_vol tags
    appended when in the top / bottom tercile (e.g. 'bull|high_vol').
    """
    close = df["close"]
    blk = ((df.index - df.index[0]) // pd.Timedelta(days=block_days)).astype(int)
    g = close.groupby(blk)
    ret = g.last() / g.first() - 1
    rv = close.pct_change().groupby(blk).std()
    lo, hi = rv.quantile(1 / 3), rv.quantile(2 / 3)
    lab = {}
    for b in ret.index:
        d = "bull" if ret[b] > trend_thresh else "bear" if ret[b] < -trend_thresh else "sideways"
        v = "high_vol" if rv[b] >= hi else "low_vol" if rv[b] <= lo else "mid_vol"
        lab[b] = f"{d}|{v}"
    return pd.Series([lab[b] for b in blk], index=df.index, name="block_regime")
