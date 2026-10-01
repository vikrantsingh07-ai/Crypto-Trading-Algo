"""The single trading core shared by the backtester and the paper runner.

Per bar (all symbols at timestamp ts), in this fixed order:
  1. fill orders decided at the PREVIOUS bar close, at THIS bar's open
  2. check stops / take-profits against this bar's high/low (stop wins ties)
  3. age positions, charge funding, ratchet trailing stops (effective next bar)
  4. mark to market at the close; update risk state (daily loss, drawdown, flatten)
  5. read signals computed from the closed bar -> schedule exits / entries for the NEXT open

Signals therefore never trade on the bar that produced them (no look-ahead).
Processing is idempotent: a timestamp <= last processed is ignored.
"""
from __future__ import annotations

from typing import Callable

from .config import Config
from .events import MemorySink, Sink
from .execution import DuplicateOrderError, Fill, Order, PaperBroker, make_client_order_id
from .portfolio import Portfolio, Position
from .risk import RiskManager

Bar = tuple  # (open, high, low, close, volume)


def _side_name(s: int) -> str:
    return "long" if s > 0 else "short"


class TradingEngine:
    def __init__(self, cfg: Config, strategy_tag: str = "strategy", sink: Sink | None = None,
                 broker: PaperBroker | None = None, bar_hours: float = 1.0):
        cfg.validate()
        self.cfg, self.tag = cfg, strategy_tag
        self.fx = cfg.account.usdt_inr
        self.sink = sink or MemorySink()
        self.broker = broker or PaperBroker(cfg.costs)
        self.pf = Portfolio(cfg.account.initial_capital, self.fx)
        self.bar_hours = bar_hours
        self.bar_idx = 0
        self.last_ts_ms = -1
        self.last_equity = cfg.account.initial_capital
        self.pending: dict[str, dict] = {}
        self.last_prices: dict[str, float] = {}
        self.n_trades = 0
        self.risk = RiskManager(cfg.risk, cfg.costs, cfg.account.initial_capital, self.fx, emit=self._risk_event)
        self._cur_ts = 0
        self.regime_tag = ""

    # ------------------------------------------------------------------ helpers
    def _risk_event(self, ts_ms: int, kind: str, detail: dict) -> None:
        self.sink.risk_event({"ts_ms": ts_ms, "kind": kind, "detail": detail})

    def _order_rec(self, o: Order, f: Fill | None, pos_id: str, status="filled") -> dict:
        return {"order_id": o.client_order_id, "ts_ms": o.ts_ms, "symbol": o.symbol, "side": o.side, "kind": o.kind,
                "purpose": o.purpose, "qty": o.qty, "price": f.price if f else None,
                "fee_usdt": f.fee_usdt if f else 0.0, "fee_inr": (f.fee_usdt * self.fx) if f else 0.0,
                "status": status, "reason": o.reason, "position_id": pos_id}

    def _close(self, sym: str, fill: Fill, order: Order, reason: str, ts_ms: int) -> None:
        p = self.pf.positions.pop(sym)
        gross = p.side * p.qty * (fill.price - p.entry_price) * self.fx
        exit_fee = fill.fee_usdt * self.fx
        fees = p.entry_fee_inr + exit_fee
        net = gross - fees - p.funding_inr
        self.pf.cash += gross - exit_fee
        self.n_trades += 1
        rec = {"trade_id": f"T-{p.pos_id}", "position_id": p.pos_id, "symbol": sym, "side": _side_name(p.side),
               "qty": p.qty, "entry_ts_ms": p.entry_ts_ms, "entry_price": p.entry_price, "exit_ts_ms": ts_ms,
               "exit_price": fill.price, "gross_pnl": gross, "fees": fees, "funding": p.funding_inr, "net_pnl": net,
               "r_multiple": net / p.risk_inr if p.risk_inr > 0 else 0.0, "exit_reason": reason,
               "bars_held": p.bars_held, "source": p.source,
               "notional_inr": p.qty * p.entry_price * self.fx, "strategy": self.tag}
        self.sink.order(self._order_rec(order, fill, p.pos_id))
        self.sink.trade(rec)
        self.sink.position({"ts_ms": ts_ms, "position_id": p.pos_id, "symbol": sym, "action": "close",
                            "price": fill.price, "qty": p.qty})
        self.risk.on_trade_closed(ts_ms, self.bar_idx, sym, net)

    def _exit_order(self, sym: str, p: Position, kind: str, purpose: str, ts_ms: int, reason: str) -> Order:
        cid = make_client_order_id(self.tag, sym, ts_ms, f"exit:{purpose}:{p.pos_id}")
        return Order(cid, ts_ms, sym, "sell" if p.side > 0 else "buy", kind, p.qty, purpose, reason=reason)

    # ------------------------------------------------------------------ main step
    def process_bar(self, ts_ms: int, bars: dict[str, Bar], signals: dict[str, dict | None] | None = None,
                    market_regime: str = "") -> bool:
        if ts_ms <= self.last_ts_ms:
            return False                       # replay / duplicate bar: ignore (idempotent)
        signals = signals or {}
        self.regime_tag = market_regime
        self._cur_ts = ts_ms
        self.bar_idx += 1
        self.sink.begin_bar(ts_ms)
        self.risk.start_bar(ts_ms, self.bar_idx, self.last_equity)

        # 1 ---- fill pending orders at this open (exits before entries)
        for sym in sorted(self.pending, key=lambda s: self.pending[s]["type"] != "exit"):
            if sym not in bars:
                continue
            pend = self.pending.pop(sym)
            o = bars[sym][0]
            if pend["type"] == "exit":
                p = self.pf.positions.get(sym)
                if p is None:
                    continue
                order = self._exit_order(sym, p, "market", pend["purpose"], ts_ms, pend["reason"])
                try:
                    fill = self.broker.fill_market(order, o)
                except DuplicateOrderError:
                    self._risk_event(ts_ms, "duplicate_order_blocked", {"order_id": order.client_order_id})
                    continue
                self._close(sym, fill, order, pend["reason"], ts_ms)
            else:
                self._fill_entry(sym, pend, o, ts_ms)

        # 2/3 ---- intrabar stop/TP, aging, funding, trailing
        for sym, p in list(self.pf.positions.items()):
            if sym not in bars:
                continue
            o, h, l, c, _ = bars[sym]
            self.last_prices[sym] = c
            hit = None
            if p.side > 0:
                if l <= p.stop:
                    hit = "stop"
                elif p.tp is not None and h >= p.tp:
                    hit = "tp"
            else:
                if h >= p.stop:
                    hit = "stop"
                elif p.tp is not None and l <= p.tp:
                    hit = "tp"
            if hit:
                if hit == "stop":
                    purpose = "trail_stop" if self._stop_is_trailed(p) else "stop"
                    order = self._exit_order(sym, p, "stop", purpose, ts_ms, purpose)
                    try:
                        fill = self.broker.fill_stop(order, p.stop, o)
                    except DuplicateOrderError:
                        continue
                    self._close(sym, fill, order, purpose, ts_ms)
                else:
                    order = self._exit_order(sym, p, "take_profit", "tp", ts_ms, "tp")
                    try:
                        fill = self.broker.fill_take_profit(order, p.tp, o)
                    except DuplicateOrderError:
                        continue
                    self._close(sym, fill, order, "tp", ts_ms)
                continue
            p.bars_held += 1
            fund = p.side * p.qty * c * self.fx * self.cfg.costs.funding_bps_per_8h / 1e4 * (self.bar_hours / 8)
            p.funding_inr += fund
            self.pf.cash -= fund
            if p.side > 0:
                p.best = max(p.best, h)
                if p.trail_dist:
                    p.stop = max(p.stop, p.best - p.trail_dist)
            else:
                p.best = min(p.best, l)
                if p.trail_dist:
                    p.stop = min(p.stop, p.best + p.trail_dist)
            if p.max_hold and p.bars_held >= p.max_hold and sym not in self.pending:
                self.pending[sym] = {"type": "exit", "purpose": "time_stop", "reason": "time_stop"}

        # 4 ---- mark to market, risk state
        for sym, b in bars.items():
            self.last_prices[sym] = b[3]
        equity = self.pf.equity(self.last_prices)
        if self.risk.update_equity(ts_ms, self.bar_idx, equity) and self.cfg.risk.flatten_on_drawdown_halt:
            for sym in list(self.pf.positions):
                self.pending[sym] = {"type": "exit", "purpose": "risk_flatten", "reason": "drawdown_halt_flatten"}
            for sym in [s for s, v in self.pending.items() if v["type"] == "entry"]:
                del self.pending[sym]
        self.last_equity = equity

        # 5 ---- decisions for the NEXT open
        for sym, sig in signals.items():
            if sig is None or sym not in bars:
                continue
            self._decide(sym, sig, bars[sym][3], ts_ms, equity)

        dd = (self.risk.true_peak - equity) / self.risk.true_peak * 100
        self.sink.equity({"ts_ms": ts_ms, "equity": equity, "cash": self.pf.cash,
                          "unrealized": equity - self.pf.cash, "n_open": len(self.pf.positions),
                          "drawdown_pct": dd, "regime": market_regime})
        self.last_ts_ms = ts_ms
        self.sink.end_bar(ts_ms, self)
        return True

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _stop_is_trailed(p: Position) -> bool:
        init = p.entry_price - p.side * p.initial_stop_dist
        return (p.stop > init + 1e-12) if p.side > 0 else (p.stop < init - 1e-12)

    def _fill_entry(self, sym: str, pend: dict, bar_open: float, ts_ms: int) -> None:
        if self.risk.dd_halted:
            self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "entry_cancelled", "reason": "drawdown_halt"})
            return
        if sym in self.pf.positions:
            return
        side = pend["side"]
        cid = make_client_order_id(self.tag, sym, pend["signal_ts_ms"], f"entry:{side}")
        order = Order(cid, ts_ms, sym, "buy" if side > 0 else "sell", "market", pend["qty"], "entry",
                      reason=pend["source"])
        try:
            fill = self.broker.fill_market(order, bar_open)
        except DuplicateOrderError:
            self._risk_event(ts_ms, "duplicate_order_blocked", {"order_id": cid})
            return
        fee_inr = fill.fee_usdt * self.fx
        sd = pend["stop_dist"]
        pos = Position(
            pos_id=f"{sym}-{ts_ms}", symbol=sym, side=side, qty=fill.qty, entry_price=fill.price, entry_ts_ms=ts_ms,
            stop=fill.price - side * sd, tp=(fill.price + side * pend["tp_dist"]) if pend.get("tp_dist") else None,
            trail_dist=pend.get("trail_dist"), best=fill.price,
            max_hold=int(pend["max_hold"]) if pend.get("max_hold") else None,
            risk_inr=pend["risk_inr"], source=pend["source"], entry_fee_inr=fee_inr,
            signal_ts_ms=pend["signal_ts_ms"], initial_stop_dist=sd)
        self.pf.positions[sym] = pos
        self.pf.cash -= fee_inr
        self.risk.trades_today += 1
        self.sink.order(self._order_rec(order, fill, pos.pos_id))
        self.sink.position({"ts_ms": ts_ms, "position_id": pos.pos_id, "symbol": sym, "action": "open",
                            "price": fill.price, "qty": fill.qty, "side": _side_name(side), "stop": pos.stop,
                            "tp": pos.tp})

    def _decide(self, sym: str, sig: dict, close: float, ts_ms: int, equity: float) -> None:
        pos = self.pf.positions.get(sym)
        # exits from strategy flags (mean-reversion positions exit via TP/stop/time only)
        if pos is not None and sym not in self.pending and pos.source != "mean_reversion":
            if (pos.side > 0 and sig.get("exit_long")) or (pos.side < 0 and sig.get("exit_short")):
                self.pending[sym] = {"type": "exit", "purpose": "signal_exit", "reason": "signal_exit"}
                self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "exit_scheduled", "reason": "signal_exit"})
        s = int(sig.get("signal", 0))
        if s == 0:
            return
        self.sink.signal({"ts_ms": ts_ms, "symbol": sym, "signal": s, "stop_dist": sig.get("stop_dist"),
                          "tp_dist": sig.get("tp_dist"), "regime": sig.get("regime", ""), "close": close})
        if pos is not None or sym in self.pending:
            self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "skip", "reason": "already_in_position_or_pending", "signal": s})
            return
        n_pend = sum(1 for v in self.pending.values() if v["type"] == "entry")
        ok, why = self.risk.can_enter(self.bar_idx, sym, len(self.pf.positions), n_pend, self.pf.open_risk_inr()
                                      + sum(v["risk_inr"] for v in self.pending.values() if v["type"] == "entry"),
                                      equity)
        if not ok:
            self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "skip", "reason": why, "signal": s})
            return
        pend_notional = sum(v["notional_inr"] for v in self.pending.values() if v["type"] == "entry")
        sz = self.risk.size(equity, close, float(sig["stop_dist"]),
                            self.pf.notional_inr(self.last_prices) + pend_notional,
                            self.pf.open_risk_inr() + sum(v["risk_inr"] for v in self.pending.values() if v["type"] == "entry"))
        if sz.qty <= 0:
            self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "skip", "reason": sz.reason, "signal": s})
            return
        tp = sig.get("tp_dist")
        self.pending[sym] = {"type": "entry", "side": s, "qty": sz.qty, "stop_dist": float(sig["stop_dist"]),
                             "tp_dist": None if tp is None or tp != tp else float(tp),
                             "trail_dist": None if sig.get("trail_dist") is None or sig["trail_dist"] != sig["trail_dist"] else float(sig["trail_dist"]),
                             "max_hold": None if sig.get("max_hold") is None or sig["max_hold"] != sig["max_hold"] else sig["max_hold"],
                             "risk_inr": sz.risk_inr, "notional_inr": sz.notional_inr,
                             "source": sig.get("regime", "") or "strategy", "signal_ts_ms": ts_ms}
        self.sink.decision({"ts_ms": ts_ms, "symbol": sym, "action": "enter_scheduled", "reason": sig.get("regime", ""),
                            "signal": s, "qty": sz.qty, "risk_inr": sz.risk_inr})

    def liquidate(self, reason: str = "end_of_data") -> None:
        """Close everything at the last seen close (used at the end of a backtest window)."""
        ts_ms = self.last_ts_ms
        for sym, p in list(self.pf.positions.items()):
            order = self._exit_order(sym, p, "market", reason, ts_ms, reason)
            fill = self.broker.fill_market(order, self.last_prices.get(sym, p.entry_price))
            self._close(sym, fill, order, reason, ts_ms)
        self.pending.clear()
        eq = self.pf.equity(self.last_prices)
        self.last_equity = eq
        self.sink.equity({"ts_ms": ts_ms, "equity": eq, "cash": self.pf.cash, "unrealized": 0.0, "n_open": 0,
                          "drawdown_pct": (self.risk.true_peak - eq) / self.risk.true_peak * 100, "regime": ""})

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"bar_idx": self.bar_idx, "last_ts_ms": self.last_ts_ms, "last_equity": self.last_equity,
                "pending": self.pending, "last_prices": self.last_prices, "n_trades": self.n_trades,
                "portfolio": self.pf.to_dict(), "risk": self.risk.to_dict()}

    def load_state(self, d: dict) -> None:
        self.bar_idx, self.last_ts_ms = d["bar_idx"], d["last_ts_ms"]
        self.last_equity, self.pending = d["last_equity"], d["pending"]
        self.last_prices, self.n_trades = d["last_prices"], d["n_trades"]
        self.pf = Portfolio.from_dict(d["portfolio"])
        self.risk.load_dict(d["risk"])

    # ------------------------------------------------------------------ convenience
    @property
    def equity(self) -> float:
        return self.pf.equity(self.last_prices)


def rows_from_signals(sig_df) -> list[dict | None]:
    """DataFrame -> per-bar dict (or None if nothing to do). Vectorised pre-extraction for speed."""
    import numpy as np
    n = len(sig_df)
    s = sig_df["signal"].to_numpy()
    xl = sig_df["exit_long"].to_numpy(dtype=bool)
    xs = sig_df["exit_short"].to_numpy(dtype=bool)
    sd, tp = sig_df["stop_dist"].to_numpy(float), sig_df["tp_dist"].to_numpy(float)
    tr, mh = sig_df["trail_dist"].to_numpy(float), sig_df["max_hold"].to_numpy(float)
    rg = sig_df["regime"].to_numpy(dtype=object)
    out: list[dict | None] = [None] * n
    for i in np.flatnonzero((s != 0) | xl | xs):
        out[i] = {"signal": int(s[i]), "stop_dist": sd[i], "tp_dist": tp[i], "trail_dist": tr[i],
                  "exit_long": bool(xl[i]), "exit_short": bool(xs[i]), "max_hold": mh[i], "regime": rg[i]}
    return out
