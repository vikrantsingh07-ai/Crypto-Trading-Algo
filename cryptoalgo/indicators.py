"""Causal technical indicators (value at bar t uses data <= t only).

Every function is vectorised on pandas objects. Causality is enforced by test
(tests/test_causality.py): truncating future data must not change past values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.models import tf_timedelta


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna() & dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * rma(plus_dm, n) / a
    minus_di = 100 * rma(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx.fillna(0).where(a.notna()), n), plus_di, minus_di


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def donchian_prior(df: pd.DataFrame, n: int):
    """Highest high / lowest low of the *previous* n bars (excludes the current bar)."""
    return df["high"].shift(1).rolling(n, min_periods=n).max(), df["low"].shift(1).rolling(n, min_periods=n).min()


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"] / sma(df["volume"], n)


def vol_state(atr_pct: pd.Series, n: int = 500) -> pd.Series:
    """ATR% relative to its trailing median: >1 = volatile, <1 = calm."""
    return atr_pct / atr_pct.rolling(n, min_periods=n // 5).median()


def htf_trend(df: pd.DataFrame, base_tf: str, htf: str, fast: int = 20, slow: int = 50) -> pd.Series:
    """Higher-timeframe trend direction (+1/-1/0) using only *closed* HTF bars.

    HTF bars are built from base bars, stamped with their CLOSE time, and joined
    to each base bar by the latest HTF close time <= the base bar's close time.
    """
    h = df.resample(htf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    # drop the final HTF bar if incomplete
    n_per = int(tf_timedelta(htf) / tf_timedelta(base_tf))
    cnt = df["close"].resample(htf, label="left", closed="left").count().reindex(h.index)
    h = h[cnt >= n_per]
    ef, es = ema(h["close"], fast), ema(h["close"], slow)
    trend = pd.Series(np.where(ef > es, 1, np.where(ef < es, -1, 0)), index=h.index).where(es.notna())
    trend.index = trend.index + tf_timedelta(htf)          # available at HTF close time
    base_close = df.index + tf_timedelta(base_tf)
    left = pd.DataFrame({"t": base_close})
    right = pd.DataFrame({"t": trend.index, "trend": trend.values}).dropna()
    m = pd.merge_asof(left, right, on="t", direction="backward")
    return pd.Series(m["trend"].values, index=df.index, name="htf_trend").fillna(0).astype(int)
