"""Market-data feeds for the paper trader. All read-only; none can place orders."""
from __future__ import annotations

import pandas as pd

from ..data.binance import BinancePublicClient, MarketDataError
from ..data.models import tf_timedelta


class FeedError(RuntimeError):
    """Temporary data problem (network, rate limit, bad payload). Callers retry with back-off."""


class BinanceFeed:
    """Closed candles from Binance public REST (no keys)."""

    def __init__(self, client: BinancePublicClient | None = None):
        self.client = client or BinancePublicClient()

    def fetch_closed(self, symbol: str, timeframe: str, limit: int = 1000, since: pd.Timestamp | None = None) -> pd.DataFrame:
        try:
            return self.client.klines(symbol, timeframe, start=since, limit=limit, closed_only=True)
        except MarketDataError as e:
            raise FeedError(str(e)) from e


class SimulatedFeed:
    """Replays a pre-generated candle history as if it were arriving live.

    ``release(n)`` makes n more bars "closed". Faults can be injected to exercise
    the reconnection / de-duplication paths: ``fail_next(n)`` raises FeedError,
    ``overlap`` returns extra already-seen bars, ``hold(symbol)`` stops one symbol.
    """

    def __init__(self, data: dict[str, pd.DataFrame], timeframe: str, start_pos: int):
        self.data, self.tf, self.pos = data, timeframe, start_pos
        self._fail = 0
        self.overlap = 0
        self._held: set[str] = set()
        self.calls = 0

    def fail_next(self, n: int = 1) -> None:
        self._fail += n

    def hold(self, symbol: str) -> None:
        self._held.add(symbol)

    def unhold(self, symbol: str) -> None:
        self._held.discard(symbol)

    def release(self, n: int = 1) -> int:
        n = min(n, min(len(d) for d in self.data.values()) - self.pos)
        self.pos += n
        return n

    @property
    def exhausted(self) -> bool:
        return self.pos >= min(len(d) for d in self.data.values())

    @property
    def now(self) -> pd.Timestamp:
        """Virtual clock: the close time of the newest released bar."""
        d = next(iter(self.data.values()))
        return d.index[self.pos - 1] + tf_timedelta(self.tf)

    def fetch_closed(self, symbol: str, timeframe: str, limit: int = 1000, since: pd.Timestamp | None = None) -> pd.DataFrame:
        self.calls += 1
        if self._fail > 0:
            self._fail -= 1
            raise FeedError("simulated network failure")
        pos = self.pos - (1 if symbol in self._held else 0)
        df = self.data[symbol].iloc[:pos]
        if since is not None:
            df = df[df.index >= since - tf_timedelta(timeframe) * self.overlap]
        return df.iloc[-limit:].copy()
