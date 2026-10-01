"""SQLite candle store (idempotent upserts)."""
from __future__ import annotations

import sqlite3

import pandas as pd

from .models import OHLCV, from_ms, to_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles(
  symbol TEXT NOT NULL, timeframe TEXT NOT NULL, open_time_ms INTEGER NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
  PRIMARY KEY(symbol, timeframe, open_time_ms));
"""


class CandleStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.executescript(SCHEMA)

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        rows = [(symbol, timeframe, to_ms(ts), r.open, r.high, r.low, r.close, r.volume)
                for ts, r in zip(df.index, df.itertuples())]
        self.conn.executemany(
            "INSERT INTO candles VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,timeframe,open_time_ms) "
            "DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,"
            "close=excluded.close,volume=excluded.volume", rows)
        self.conn.commit()
        return len(rows)

    def load(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        q = ("SELECT open_time_ms,open,high,low,close,volume FROM candles "
             "WHERE symbol=? AND timeframe=? ORDER BY open_time_ms")
        rows = self.conn.execute(q, (symbol, timeframe)).fetchall()
        if limit:
            rows = rows[-limit:]
        if not rows:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(rows, columns=["t"] + OHLCV)
        df.index = pd.DatetimeIndex([from_ms(int(t)) for t in df.pop("t")])
        return df
