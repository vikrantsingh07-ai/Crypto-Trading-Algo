"""Event sinks: how the engine reports what it did (backtest = memory, paper = SQLite)."""
from __future__ import annotations


class Sink:
    """No-op base. Override what you need."""

    def begin_bar(self, ts_ms: int) -> None: ...
    def signal(self, rec: dict) -> None: ...
    def decision(self, rec: dict) -> None: ...
    def order(self, rec: dict) -> None: ...
    def position(self, rec: dict) -> None: ...
    def trade(self, rec: dict) -> None: ...
    def risk_event(self, rec: dict) -> None: ...
    def equity(self, rec: dict) -> None: ...
    def end_bar(self, ts_ms: int, engine_state: dict) -> None: ...


class MemorySink(Sink):
    def __init__(self, keep_decisions: bool = True):
        self.keep_decisions = keep_decisions
        self.signals, self.decisions, self.orders = [], [], []
        self.positions, self.trades, self.risk_events, self.equity_curve = [], [], [], []

    def signal(self, rec):
        self.signals.append(rec)

    def decision(self, rec):
        if self.keep_decisions:
            self.decisions.append(rec)

    def order(self, rec):
        self.orders.append(rec)

    def position(self, rec):
        self.positions.append(rec)

    def trade(self, rec):
        self.trades.append(rec)

    def risk_event(self, rec):
        self.risk_events.append(rec)

    def equity(self, rec):
        self.equity_curve.append(rec)
