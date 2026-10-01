"""Simulated order execution (PAPER ONLY).

`PaperBroker` is a pure in-memory simulator: it turns an order plus a candle
into a simulated fill. It performs no I/O and has no connection to any exchange.
It enforces idempotency: a client_order_id can fill at most once.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .config import Costs


class DuplicateOrderError(RuntimeError):
    pass


def make_client_order_id(strategy: str, symbol: str, bar_ms: int, intent: str) -> str:
    """Deterministic id: the same decision on the same bar always yields the same id."""
    raw = f"{strategy}|{symbol}|{bar_ms}|{intent}"
    return "P" + hashlib.sha1(raw.encode()).hexdigest()[:19]


@dataclass
class Order:
    client_order_id: str
    ts_ms: int
    symbol: str
    side: str            # 'buy' | 'sell'
    kind: str            # 'market' | 'stop' | 'take_profit'
    qty: float
    purpose: str         # entry | signal_exit | stop | tp | trail_stop | time_stop | risk_flatten
    ref_price: float = 0.0
    reason: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Fill:
    client_order_id: str
    ts_ms: int
    symbol: str
    side: str
    kind: str
    purpose: str
    qty: float
    price: float
    fee_usdt: float
    slippage_cost_usdt: float   # cost of slippage+spread vs the reference price (informational)
    status: str = "filled"


class PaperBroker:
    def __init__(self, costs: Costs, seen_ids: set[str] | None = None):
        self.c = costs
        self._seen: set[str] = set(seen_ids or ())

    # -- idempotency -------------------------------------------------------
    def has_seen(self, client_order_id: str) -> bool:
        return client_order_id in self._seen

    def _claim(self, order: Order) -> None:
        if order.client_order_id in self._seen:
            raise DuplicateOrderError(order.client_order_id)
        self._seen.add(order.client_order_id)

    @property
    def seen_ids(self) -> set[str]:
        return set(self._seen)

    # -- price model -------------------------------------------------------
    def _bps(self, stop: bool) -> float:
        b = self.c.half_spread_bps + self.c.slippage_bps
        return b + (self.c.stop_extra_slippage_bps if stop else 0.0)

    def _fee(self, price: float, qty: float) -> float:
        return price * qty * self.c.taker_fee_bps / 1e4

    def _adverse(self, side: str, ref: float, bps: float) -> float:
        return ref * (1 + bps / 1e4) if side == "buy" else ref * (1 - bps / 1e4)

    # -- fills -------------------------------------------------------------
    def fill_market(self, order: Order, bar_open: float) -> Fill:
        self._claim(order)
        px = self._adverse(order.side, bar_open, self._bps(False))
        return Fill(order.client_order_id, order.ts_ms, order.symbol, order.side, order.kind, order.purpose,
                    order.qty, px, self._fee(px, order.qty), abs(px - bar_open) * order.qty)

    def fill_stop(self, order: Order, stop_price: float, bar_open: float) -> Fill:
        """Stop-market: gap-through fills at the open (worse than the stop), plus extra slippage."""
        self._claim(order)
        trigger = min(stop_price, bar_open) if order.side == "sell" else max(stop_price, bar_open)
        px = self._adverse(order.side, trigger, self._bps(True))
        return Fill(order.client_order_id, order.ts_ms, order.symbol, order.side, order.kind, order.purpose,
                    order.qty, px, self._fee(px, order.qty), abs(px - stop_price) * order.qty)

    def fill_take_profit(self, order: Order, tp_price: float, bar_open: float) -> Fill:
        """Limit order: fills at the limit (or better on a gap); no slippage, still charged taker fee (conservative)."""
        self._claim(order)
        px = max(tp_price, bar_open) if order.side == "sell" else min(tp_price, bar_open)
        return Fill(order.client_order_id, order.ts_ms, order.symbol, order.side, order.kind, order.purpose,
                    order.qty, px, self._fee(px, order.qty), 0.0)
