"""Data-integrity checks for candle frames."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import OHLCV, tf_timedelta


class DataIntegrityError(ValueError):
    pass


@dataclass
class IntegrityReport:
    ok: bool = True
    n_bars: int = 0
    gaps: list = field(default_factory=list)          # (start, end, missing_bars)
    issues: list = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.issues.append(msg)


def check_candles(df: pd.DataFrame, timeframe: str, max_abs_return: float = 0.5) -> IntegrityReport:
    rep = IntegrityReport(n_bars=len(df))
    if df.empty:
        rep.fail("empty frame")
        return rep
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        rep.fail(f"missing columns {missing}")
        return rep
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        rep.fail("index must be a tz-aware DatetimeIndex (UTC)")
        return rep
    if not df.index.is_monotonic_increasing:
        rep.fail("index not sorted")
    if df.index.has_duplicates:
        rep.fail(f"{int(df.index.duplicated().sum())} duplicate timestamps")
    if df[OHLCV].isna().any().any():
        rep.fail("NaN values present")
    if np.isinf(df[OHLCV].to_numpy(dtype=float)).any():
        rep.fail("infinite values present")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        rep.fail("non-positive prices")
    if (df["volume"] < 0).any():
        rep.fail("negative volume")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9).any():
        rep.fail("high below open/close/low")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9).any():
        rep.fail("low above open/close/high")
    step = tf_timedelta(timeframe)
    d = df.index.to_series().diff().dropna()
    off = d[d != step]
    if not off.empty:
        bad_small = off[off < step]
        if not bad_small.empty:
            rep.fail(f"{len(bad_small)} intervals shorter than timeframe (misaligned / overlapping)")
        for ts, delta in off[off > step].items():
            rep.gaps.append((ts - delta, ts, int(delta / step) - 1))
    r = df["close"].pct_change().abs()
    if (r > max_abs_return).any():
        rep.fail(f"{int((r > max_abs_return).sum())} bar returns exceed {max_abs_return:.0%} (bad ticks?)")
    return rep


def require_clean(df: pd.DataFrame, timeframe: str, allow_gaps: bool = False) -> IntegrityReport:
    rep = check_candles(df, timeframe)
    if not rep.ok or (rep.gaps and not allow_gaps):
        extra = f"; {len(rep.gaps)} gaps" if rep.gaps and not allow_gaps else ""
        raise DataIntegrityError("; ".join(rep.issues) + extra)
    return rep


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and drop duplicate timestamps (keep last). Does NOT invent missing bars."""
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()
