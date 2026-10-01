"""Virtual portfolio: cash, positions, equity. Account currency = INR; prices are in USDT.

Margin-style accounting: opening a position costs only its fee; equity = cash +
unrealised P&L. Exposure is bounded by the risk manager's leverage cap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Position:
    pos_id: str
    symbol: str
    side: int                 # +1 long, -1 short
    qty: float
    entry_price: float
    entry_ts_ms: int
    stop: float
    tp: float | None
    trail_dist: float | None
    best: float               # best price seen since entry (for chandelier trail)
    max_hold: int | None
    risk_inr: float
    source: str               # regime/sub-strategy that opened it
    bars_held: int = 0
    entry_fee_inr: float = 0.0
    funding_inr: float = 0.0   # positive = cost paid
    signal_ts_ms: int = 0
    initial_stop_dist: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class Portfolio:
    initial_capital: float
    fx: float
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cash:
            self.cash = self.initial_capital

    def unrealized_inr(self, prices: dict[str, float]) -> float:
        tot = 0.0
        for s, p in self.positions.items():
            px = prices.get(s, p.entry_price)
            tot += p.side * p.qty * (px - p.entry_price) * self.fx
        return tot

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.unrealized_inr(prices)

    def notional_inr(self, prices: dict[str, float]) -> float:
        return sum(p.qty * prices.get(s, p.entry_price) * self.fx for s, p in self.positions.items())

    def open_risk_inr(self) -> float:
        return sum(p.risk_inr for p in self.positions.values())

    def to_dict(self) -> dict:
        return {"initial_capital": self.initial_capital, "fx": self.fx, "cash": self.cash,
                "positions": {s: p.to_dict() for s, p in self.positions.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        pf = cls(d["initial_capital"], d["fx"], d["cash"])
        pf.positions = {s: Position.from_dict(p) for s, p in d["positions"].items()}
        return pf
