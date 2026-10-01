"""Builds the JSON the dashboard shows. Read-only against the paper DB."""
from __future__ import annotations

import json
import sqlite3
import time

import pandas as pd

from ..backtest.metrics import compute_metrics, daily_pnl_series
from ..data.models import from_ms
from ..safety import PAPER_BANNER


def _status(conn, key, default=None):
    r = conn.execute("SELECT value, updated_ms FROM status WHERE key=?", (key,)).fetchone()
    return (json.loads(r["value"]), r["updated_ms"]) if r else (default, None)


def build_state(conn: sqlite3.Connection, initial_capital: float) -> dict:
    bot, bot_ts = _status(conn, "bot", {})
    risk, _ = _status(conn, "risk", {})
    valid, _ = _status(conn, "validation", {})
    strat, _ = _status(conn, "strategy", {})
    cfg, _ = _status(conn, "config", {})
    initial = (cfg or {}).get("initial_capital", initial_capital)

    eq = pd.read_sql_query("SELECT ts_ms, equity, drawdown_pct FROM equity ORDER BY ts_ms", conn)
    trades = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_ts_ms", conn)
    equity_s = pd.Series(eq["equity"].to_numpy(), index=pd.DatetimeIndex([from_ms(t) for t in eq["ts_ms"]])) if len(eq) else pd.Series(dtype=float)
    tf = (strat or {}).get("timeframe", "1h")
    m = compute_metrics(trades, equity_s, initial, tf) if len(equity_s) > 1 else None

    equity_now = float(eq["equity"].iloc[-1]) if len(eq) else initial
    day_start = (risk or {}).get("day_start_equity", initial)
    state_row = conn.execute("SELECT state FROM engine_state WHERE id=1").fetchone()
    positions, cash = [], initial
    if state_row:
        st = json.loads(state_row["state"])
        cash = st["portfolio"]["cash"]
        fx = st["portfolio"]["fx"]
        for sym, p in st["portfolio"]["positions"].items():
            px = st["last_prices"].get(sym, p["entry_price"])
            positions.append({"symbol": sym, "side": "long" if p["side"] > 0 else "short", "qty": p["qty"],
                              "entry_price": p["entry_price"], "mark": px, "stop": p["stop"], "tp": p["tp"],
                              "unrealized_inr": p["side"] * p["qty"] * (px - p["entry_price"]) * fx,
                              "opened": str(from_ms(p["entry_ts_ms"])), "bars_held": p["bars_held"], "source": p["source"]})
    step = max(1, len(eq) // 1000)
    curve = [[int(t), float(v)] for t, v in zip(eq["ts_ms"].iloc[::step], eq["equity"].iloc[::step])]
    if len(eq) and (len(eq) - 1) % step:
        curve.append([int(eq["ts_ms"].iloc[-1]), float(eq["equity"].iloc[-1])])
    daily = []
    if len(equity_s) > 1:
        d = daily_pnl_series(equity_s, initial)
        daily = [{"day": k.strftime("%Y-%m-%d"), "pnl": float(v)} for k, v in d.tail(60).items()]

    def rows(sql, n=40):
        return [dict(r) for r in conn.execute(sql, (n,)).fetchall()]

    recent = rows("SELECT trade_id,symbol,side,qty,entry_price,exit_price,net_pnl,fees,exit_reason,exit_ts_ms,r_multiple "
                  "FROM trades ORDER BY exit_ts_ms DESC LIMIT ?", 25)
    for r in recent:
        r["exit_time"] = str(from_ms(r.pop("exit_ts_ms")))
    errors = rows("SELECT ts_ms,component,message FROM errors ORDER BY id DESC LIMIT ?", 25)
    revents = rows("SELECT ts_ms,kind,detail FROM risk_events ORDER BY id DESC LIMIT ?", 25)
    decisions = rows("SELECT ts_ms,symbol,action,reason FROM decisions ORDER BY id DESC LIMIT ?", 25)
    for lst in (errors, revents, decisions):
        for r in lst:
            r["time"] = str(from_ms(r.pop("ts_ms"))) if r["ts_ms"] > 10**11 else r.pop("ts_ms")
    n_open_unreal = sum(p["unrealized_inr"] for p in positions)
    return {
        "banner": PAPER_BANNER, "generated": time.time(),
        "bot": bot, "bot_updated_ms": bot_ts, "risk": risk, "validation": valid, "strategy": strat,
        "balance": cash, "equity": equity_now, "initial_capital": initial, "unrealized": n_open_unreal,
        "today_pnl": equity_now - day_start, "total_pnl": equity_now - initial,
        "metrics": None if m is None else {k: m[k] for k in (
            "win_rate_pct", "loss_rate_pct", "profit_factor", "max_drawdown_pct", "sharpe", "sortino", "trades",
            "avg_trade", "risk_reward", "daily_avg_pnl", "total_fees", "max_consecutive_wins", "max_consecutive_losses")},
        "monthly": (m or {}).get("monthly", []),
        "current_drawdown_pct": float(eq["drawdown_pct"].iloc[-1]) if len(eq) else 0.0,
        "regimes": (bot or {}).get("regimes", {}),
        "positions": positions, "recent_trades": recent, "equity_curve": curve, "daily_pnl": daily,
        "errors": errors, "risk_events": revents, "decisions": decisions,
    }
