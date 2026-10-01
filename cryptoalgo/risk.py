"""Risk management. Every limit comes from `config.Risk`; nothing is hard-coded.

Controls
  * risk per trade (position size from stop distance, incl. round-trip cost allowance)
  * max open positions / max total open risk / max leverage / max single-position size
  * max daily loss  -> no new entries until the next UTC day
  * max drawdown    -> no new entries (optionally flatten) until reset
        reset modes: 'cooldown' (after N bars; re-baselines the peak, logged) or 'manual'
  * cooldown after N consecutive losing trades
  * per-symbol re-entry cooldown and max trades/day (anti-overtrading)
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Costs, Risk


@dataclass
class SizeResult:
    qty: float
    risk_inr: float
    notional_inr: float
    reason: str = ""


class RiskManager:
    def __init__(self, risk: Risk, costs: Costs, initial_equity: float, fx: float, emit=None):
        self.r, self.costs, self.fx = risk, costs, fx
        self._emit = emit or (lambda *a, **k: None)
        self.peak = initial_equity
        self.true_peak = initial_equity        # never re-baselined: used for reporting true drawdown
        self.day_key: str | None = None
        self.day_start_equity = initial_equity
        self.daily_halted = False
        self.dd_halted = False
        self.hard_halted = False
        self.dd_halt_bar = -1
        self.loss_streak = 0
        self.loss_pause_until = -1
        self.last_exit_bar: dict[str, int] = {}
        self.trades_today = 0

    # -- state -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "peak", "true_peak", "day_key", "day_start_equity", "daily_halted", "dd_halted", "hard_halted", "dd_halt_bar",
            "loss_streak", "loss_pause_until", "last_exit_bar", "trades_today")}

    def load_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    # -- lifecycle ---------------------------------------------------------
    def start_bar(self, ts_ms: int, bar_idx: int, prev_equity: float) -> None:
        import datetime as dt
        key = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        if key != self.day_key:
            if self.daily_halted:
                self._emit(ts_ms, "daily_loss_reset", {"day": key})
            self.day_key, self.day_start_equity = key, prev_equity
            self.daily_halted = False
            self.trades_today = 0

    def update_equity(self, ts_ms: int, bar_idx: int, equity: float) -> bool:
        """Returns True if a drawdown halt was *newly* triggered on this bar (caller may flatten)."""
        newly = False
        self.true_peak = max(self.true_peak, equity)
        true_dd = (self.true_peak - equity) / self.true_peak * 100 if self.true_peak > 0 else 0.0
        if not self.hard_halted and true_dd >= self.r.hard_stop_drawdown_pct:
            self.hard_halted = True
            self.dd_halted, self.dd_halt_bar = True, bar_idx
            newly = True
            self._emit(ts_ms, "hard_stop", {"drawdown_from_all_time_peak_pct": round(true_dd, 3),
                                            "limit_pct": self.r.hard_stop_drawdown_pct, "equity": equity,
                                            "note": "manual reset required"})
        if self.hard_halted:
            return newly
        if self.dd_halted:
            if self.r.drawdown_reset_mode == "cooldown" and bar_idx - self.dd_halt_bar >= self.r.drawdown_cooldown_bars:
                self.dd_halted = False
                self.peak = equity
                self._emit(ts_ms, "drawdown_reset", {"mode": "cooldown", "rebaselined_peak": equity})
            return False
        self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak * 100 if self.peak > 0 else 0.0
        if dd >= self.r.max_drawdown_pct:
            self.dd_halted, self.dd_halt_bar, newly = True, bar_idx, True
            self._emit(ts_ms, "drawdown_halt", {"drawdown_pct": round(dd, 3), "limit_pct": self.r.max_drawdown_pct,
                                                "equity": equity, "peak": self.peak})
        day_loss = (self.day_start_equity - equity) / self.day_start_equity * 100 if self.day_start_equity > 0 else 0.0
        if not self.daily_halted and day_loss >= self.r.max_daily_loss_pct:
            self.daily_halted = True
            self._emit(ts_ms, "daily_loss_halt", {"day_loss_pct": round(day_loss, 3),
                                                  "limit_pct": self.r.max_daily_loss_pct, "equity": equity})
        return newly

    def manual_reset(self, ts_ms: int, equity: float) -> None:
        self.dd_halted, self.daily_halted, self.hard_halted = False, False, False
        self.peak = self.true_peak = equity
        self.loss_pause_until = -1
        self._emit(ts_ms, "manual_reset", {"equity": equity})

    def on_trade_closed(self, ts_ms: int, bar_idx: int, symbol: str, net_pnl: float) -> None:
        self.last_exit_bar[symbol] = bar_idx
        if net_pnl < 0:
            self.loss_streak += 1
            if self.loss_streak >= self.r.max_consecutive_losses:
                self.loss_pause_until = bar_idx + self.r.loss_cooldown_bars
                self._emit(ts_ms, "loss_streak_cooldown", {"streak": self.loss_streak,
                                                           "pause_until_bar": self.loss_pause_until})
                self.loss_streak = 0
        else:
            self.loss_streak = 0

    # -- gating & sizing ---------------------------------------------------
    def can_enter(self, bar_idx: int, symbol: str, n_open: int, n_pending: int, open_risk_inr: float,
                  equity: float) -> tuple[bool, str]:
        r = self.r
        if self.hard_halted:
            return False, "hard_stop"
        if self.dd_halted:
            return False, "drawdown_halt"
        if self.daily_halted:
            return False, "daily_loss_halt"
        if bar_idx < self.loss_pause_until:
            return False, "loss_streak_cooldown"
        if bar_idx - self.last_exit_bar.get(symbol, -10**9) < r.symbol_cooldown_bars:
            return False, "symbol_cooldown"
        if self.trades_today >= r.max_trades_per_day:
            return False, "max_trades_per_day"
        if n_open + n_pending >= r.max_open_positions:
            return False, "max_open_positions"
        if open_risk_inr >= equity * r.max_total_open_risk_pct / 100 - 1e-9:
            return False, "max_total_open_risk"
        return True, ""

    def size(self, equity: float, ref_price: float, stop_dist: float, open_notional_inr: float,
             open_risk_inr: float) -> SizeResult:
        r, c = self.r, self.costs
        if not (stop_dist > 0 and ref_price > 0 and equity > 0):
            return SizeResult(0, 0, 0, "invalid_stop")
        cost_buf = 2 * (c.taker_fee_bps + c.half_spread_bps + c.slippage_bps + c.stop_extra_slippage_bps) / 1e4 * ref_price
        eff = stop_dist + cost_buf
        risk_inr = equity * r.risk_per_trade_pct / 100
        room = equity * r.max_total_open_risk_pct / 100 - open_risk_inr
        risk_inr = min(risk_inr, room)
        if risk_inr <= 0:
            return SizeResult(0, 0, 0, "no_risk_budget")
        qty = risk_inr / (eff * self.fx)
        cap_pos = equity * r.max_position_pct / 100
        cap_lev = equity * r.max_leverage - open_notional_inr
        cap = min(cap_pos, cap_lev)
        notional = qty * ref_price * self.fx
        if notional > cap:
            if cap <= 0:
                return SizeResult(0, 0, 0, "leverage_cap")
            qty = cap / (ref_price * self.fx)
            notional = cap
            risk_inr = qty * eff * self.fx
        if notional < r.min_notional_inr:
            return SizeResult(0, 0, 0, "below_min_notional")
        return SizeResult(qty, risk_inr, notional)
