#!/usr/bin/env python3
"""Backtest one strategy over a data set and print every required metric. (Exploratory: use run_validation for decisions.)"""
import argparse

from cryptoalgo.backtest import report as rp
from cryptoalgo.backtest.feasibility import assess_target
from cryptoalgo.backtest.metrics import by_label, compute_metrics
from cryptoalgo.backtest.runner import run_backtest
from cryptoalgo.config import load_config
from cryptoalgo.data.integrity import require_clean
from cryptoalgo.data.loader import load_data
from cryptoalgo.regime import label_blocks
from cryptoalgo.strategies.library import REGISTRY, get_strategy

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--strategy", default=None, choices=sorted(REGISTRY))
ap.add_argument("--htf", action="store_true")
ap.add_argument("--start", default=None)
ap.add_argument("--end", default=None)
ap.add_argument("--out", default="reports/backtest")
a = ap.parse_args()
cfg = load_config(a.config)
data, prov = load_data(a.data, cfg.market.symbols, cfg.market.timeframe)
for df in data.values():
    require_clean(df, cfg.market.timeframe)
st = get_strategy(a.strategy or cfg.strategy.name, cfg.strategy.params or None, cfg.market.timeframe, cfg.market.htf,
                  a.htf or cfg.strategy.use_htf_confirmation)
import pandas as pd
r = run_backtest(data, cfg, st, start=pd.Timestamp(a.start, tz="UTC") if a.start else None,
                 end=pd.Timestamp(a.end, tz="UTC") if a.end else None)
m = compute_metrics(r.trades, r.equity, cfg.account.initial_capital, cfg.market.timeframe)
print("DATA:", prov, "\nSTRATEGY:", st.name, st.params, "\n")
print(rp.metrics_md(m), "\n\nMonthly:\n", rp.monthly_md(m))
reg = {s: by_label(r.trades[r.trades.symbol == s], label_blocks(df)) for s, df in data.items()}
print("\nBy market regime (ex-post labels, reporting only):")
for s, d in reg.items():
    for k, v in sorted(d.items()):
        print(f"  {s} {k:22s} trades={v['trades']:4d} net={v['net_pnl']:10.0f} PF={v['profit_factor']:.2f}")
f = assess_target(r.equity, cfg.account.initial_capital, cfg, m["max_drawdown_pct"])
print("\nTarget feasibility:", f["verdict"])
from pathlib import Path
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
rp.dump_json({"provenance": prov, "strategy": st.name, "params": st.params, "metrics": m, "regimes": reg, "feasibility": f}, a.out + ".json")
body = (f"<h1>Backtest: {st.name}</h1><p>{prov}</p>" + rp.equity_svg(r.equity, cfg.account.initial_capital)
        + f"<pre style='white-space:pre-wrap'>{rp.metrics_md(m)}\n\n{rp.monthly_md(m)}</pre><p><b>{f['verdict']}</b></p>")
Path(a.out + ".html").write_text(rp.html_page("Backtest", body))
print(f"\nwrote {a.out}.json / .html")
