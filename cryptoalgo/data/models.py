"""Candle conventions.

A candle DataFrame is indexed by UTC *open* time (tz-aware DatetimeIndex) with
columns open, high, low, close, volume. A bar is "closed" (usable for decisions)
only once open_time + timeframe <= now.
"""
from __future__ import annotations

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

_TF = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
       "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400}


def tf_seconds(tf: str) -> int:
    try:
        return _TF[tf]
    except KeyError:
        raise ValueError(f"unsupported timeframe {tf!r}; use one of {sorted(_TF)}") from None


def tf_timedelta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=tf_seconds(tf))


def bars_per_day(tf: str) -> float:
    return 86400 / tf_seconds(tf)


def to_ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def closed_only(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the still-forming candle(s)."""
    return df[df.index + tf_timedelta(tf) <= now]
