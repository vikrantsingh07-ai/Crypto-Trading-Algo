"""Report helpers: JSON-safe conversion, markdown tables, static HTML/SVG equity charts."""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd

PAPER_LABEL = "BACKTEST / PAPER-TRADING RESEARCH - VIRTUAL FUNDS ONLY"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return jsonable(o.to_dict())
    return o


def dump_json(obj, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def inr(x: float) -> str:
    s = f"{abs(x):,.0f}"
    return f"-₹{s}" if x < 0 else f"₹{s}"


def metrics_md(m: dict) -> str:
    rows = [
        ("Initial capital", inr(m["initial_capital"])), ("Final balance", inr(m["final_balance"])),
        ("Net profit/loss", inr(m["net_pnl"])), ("Total return", f"{m['total_return_pct']:.2f}%"),
        ("Daily average P&L", inr(m["daily_avg_pnl"])), ("Trades", m["trades"]),
        ("Win rate / loss rate", f"{m['win_rate_pct']:.1f}% / {m['loss_rate_pct']:.1f}%"),
        ("Risk/reward (avg win / avg loss)", f"{m['risk_reward']:.2f}"), ("Profit factor", f"{m['profit_factor']:.2f}"),
        ("Average trade", inr(m["avg_trade"])), ("Expectancy (R)", f"{m['expectancy_r']:.3f}"),
        ("Max drawdown", f"{m['max_drawdown_pct']:.2f}% ({inr(m['max_drawdown_inr'])})"),
        ("Sharpe / Sortino (daily, ann.)", f"{m['sharpe']:.2f} / {m['sortino']:.2f}"),
        ("Max consecutive wins / losses", f"{m['max_consecutive_wins']} / {m['max_consecutive_losses']}"),
        ("Fees paid / funding", f"{inr(m['total_fees'])} / {inr(m['total_funding'])}"),
        ("Long: trades, net, PF", f"{m['long']['trades']}, {inr(m['long']['net_pnl'])}, {m['long']['profit_factor']:.2f}"),
        ("Short: trades, net, PF", f"{m['short']['trades']}, {inr(m['short']['net_pnl'])}, {m['short']['profit_factor']:.2f}"),
        ("Positive days / months", f"{m['positive_days_pct']:.0f}% / {m['positive_months_pct']:.0f}%"),
    ]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def monthly_md(m: dict) -> str:
    if not m.get("monthly"):
        return "_no data_"
    return "| Month | P&L | Return |\n|---|---:|---:|\n" + "\n".join(
        f"| {r['month']} | {inr(r['pnl'])} | {r['return_pct']:.2f}% |" for r in m["monthly"])


def equity_svg(equity: pd.Series, initial: float, width=900, height=260, title="Equity curve (virtual INR)") -> str:
    if len(equity) < 2:
        return "<p>no equity data</p>"
    step = max(1, len(equity) // 800)
    e = equity.iloc[::step]
    y = e.to_numpy()
    lo, hi = min(y.min(), initial), max(y.max(), initial)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    px = lambda i: 50 + i / (len(y) - 1) * (width - 70)
    py = lambda v: 20 + (hi - v) / (hi - lo) * (height - 50)
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(y))
    base = py(initial)
    labels = "".join(f'<text x="4" y="{py(v):.0f}" font-size="10" fill="#888">{v:,.0f}</text>' for v in (hi - pad, initial, lo + pad))
    t0, t1 = e.index[0].strftime("%Y-%m-%d"), e.index[-1].strftime("%Y-%m-%d")
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">'
            f'<text x="50" y="14" font-size="12" fill="currentColor">{html.escape(title)}</text>{labels}'
            f'<line x1="50" x2="{width-20}" y1="{base:.1f}" y2="{base:.1f}" stroke="#999" stroke-dasharray="4 3"/>'
            f'<polyline fill="none" stroke="#2a7de1" stroke-width="1.6" points="{pts}"/>'
            f'<text x="50" y="{height-8}" font-size="10" fill="#888">{t0}</text>'
            f'<text x="{width-90}" y="{height-8}" font-size="10" fill="#888">{t1}</text></svg>')


def html_page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;'
            f'color:#1c1c1c;background:#fff}}@media(prefers-color-scheme:dark){{body{{color:#e8e8e8;background:#141414}}}}'
            f'.banner{{background:#b00020;color:#fff;padding:8px 12px;font-weight:700;text-align:center;border-radius:6px}}'
            f'table{{border-collapse:collapse}}td,th{{border:1px solid #8884;padding:4px 10px;text-align:left}}'
            f'.pass{{color:#0a7d2c}}.fail{{color:#c62828}}</style></head><body>'
            f'<div class="banner">{PAPER_LABEL}</div>{body}</body></html>')
