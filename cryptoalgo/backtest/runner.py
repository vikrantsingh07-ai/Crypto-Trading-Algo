"""Backtest runner: replays candles through the shared TradingEngine."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data.models import bars_per_day, from_ms, tf_timedelta, to_ms
from ..engine import TradingEngine, rows_from_signals
from ..events import MemorySink
from ..strategies.base import Strategy

TRADE_COLS = ["trade_id", "position_id", "symbol", "side", "qty", "entry_ts_ms", "entry_price", "exit_ts_ms",
              "exit_price", "gross_pnl", "fees", "funding", "net_pnl", "r_multiple", "exit_reason", "bars_held",
              "source", "notional_inr", "strategy"]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    initial_capital: float
    timeframe: str
    symbols: list
    strategy: str
    params: dict
    orders: list = field(default_factory=list)
    risk_events: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


def precompute(data: dict[str, pd.DataFrame], strategy: Strategy) -> dict[str, list]:
    """Causal signals for each symbol over its FULL history (so windows get proper warm-up)."""
    return {s: rows_from_signals(strategy.signals(df)) for s, df in data.items()}


def common_index(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return idx


def run_backtest(data: dict[str, pd.DataFrame], cfg: Config, strategy: Strategy, timeframe: str | None = None,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 rows: dict[str, list] | None = None, keep_decisions: bool = False,
                 liquidate_at_end: bool = True) -> BacktestResult:
    """Trade bars with open time in [start, end). ``rows`` = precomputed signals from precompute()."""
    tf = timeframe or cfg.market.timeframe
    rows = rows if rows is not None else precompute(data, strategy)
    idx = common_index(data)
    pos_of = {s: pd.Series(np.arange(len(df)), index=df.index) for s, df in data.items()}
    lo = 0 if start is None else idx.searchsorted(pd.Timestamp(start), side="left")
    hi = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end), side="left")
    arr = {s: df[["open", "high", "low", "close", "volume"]].to_numpy() for s, df in data.items()}
    sink = MemorySink(keep_decisions=keep_decisions)
    eng = TradingEngine(cfg, strategy_tag=strategy.name, sink=sink, bar_hours=tf_timedelta(tf).total_seconds() / 3600)
    minb = strategy.min_bars
    sel = {s: pos_of[s].reindex(idx).to_numpy() for s in data}
    for k in range(lo, hi):
        ts = idx[k]
        bars, sigs = {}, {}
        for s in data:
            j = int(sel[s][k])
            bars[s] = tuple(arr[s][j])
            sigs[s] = rows[s][j] if j >= minb else None
        eng.process_bar(to_ms(ts), bars, sigs)
    if liquidate_at_end and eng.last_ts_ms > 0:
        eng.liquidate("end_of_data")
    trades = pd.DataFrame(sink.trades, columns=TRADE_COLS) if sink.trades else pd.DataFrame(columns=TRADE_COLS)
    if len(sink.equity_curve):
        eq = pd.Series([e["equity"] for e in sink.equity_curve],
                       index=pd.DatetimeIndex([from_ms(e["ts_ms"]) for e in sink.equity_curve]), name="equity")
        eq = eq[~eq.index.duplicated(keep="last")]
    else:
        eq = pd.Series(dtype=float, name="equity")
    return BacktestResult(trades, eq, cfg.account.initial_capital, tf, list(data), strategy.name, dict(strategy.params),
                          sink.orders, sink.risk_events, sink.signals, sink.decisions,
                          idx[lo] if lo < len(idx) else None, idx[hi - 1] if hi > 0 and hi - 1 < len(idx) else None)
