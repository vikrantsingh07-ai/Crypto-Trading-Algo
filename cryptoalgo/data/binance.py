"""Binance PUBLIC market-data client (klines only).

Unauthenticated GET requests to public endpoints. This class intentionally has
no parameter, header or code path that could carry credentials, and no order endpoint.
"""
from __future__ import annotations

import time

import pandas as pd

from .models import OHLCV, from_ms, tf_seconds, to_ms

HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")  # 2nd: public-data mirror
KLINE_PATH = "/api/v3/klines"


class MarketDataError(RuntimeError):
    pass


class BinancePublicClient:
    def __init__(self, session=None, hosts=HOSTS, timeout: float = 10.0, now_fn=time.time):
        if session is None:
            import requests  # optional dependency
            session = requests.Session()
        self.session = session
        self.hosts = tuple(hosts)
        self.timeout = timeout
        self._now = now_fn

    def _get(self, params: dict) -> list:
        last = None
        for host in self.hosts:
            try:
                r = self.session.get(host + KLINE_PATH, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code == 418:
                    raise MarketDataError(f"rate limited ({r.status_code})")
                if r.status_code != 200:
                    raise MarketDataError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                if not isinstance(data, list):
                    raise MarketDataError(f"unexpected payload: {str(data)[:200]}")
                return data
            except MarketDataError as e:
                last = e
            except Exception as e:  # network / JSON / timeout
                last = MarketDataError(f"{type(e).__name__}: {e}")
        raise last or MarketDataError("no hosts configured")

    @staticmethod
    def _to_frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC"))
        idx = pd.DatetimeIndex([from_ms(int(r[0])) for r in rows])
        df = pd.DataFrame({"open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                           "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                           "volume": [float(r[5]) for r in rows]}, index=idx)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def klines(self, symbol: str, timeframe: str, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None, limit: int = 1000, closed_only: bool = True) -> pd.DataFrame:
        tf_seconds(timeframe)
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        if start is not None:
            params["startTime"] = to_ms(start)
        if end is not None:
            params["endTime"] = to_ms(end)
        df = self._to_frame(self._get(params))
        if closed_only and not df.empty:
            now = pd.Timestamp(self._now(), unit="s", tz="UTC")
            df = df[df.index + pd.Timedelta(seconds=tf_seconds(timeframe)) <= now]
        return df

    def history(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
                sleep_s: float = 0.2) -> pd.DataFrame:
        """Paginated download of closed candles in [start, end)."""
        frames, cur = [], pd.Timestamp(start)
        step = pd.Timedelta(seconds=tf_seconds(timeframe))
        while cur < end:
            df = self.klines(symbol, timeframe, start=cur, end=end, limit=1000)
            if df.empty:
                break
            frames.append(df)
            nxt = df.index[-1] + step
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(sleep_s)
        if not frames:
            return self._to_frame([])
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()
