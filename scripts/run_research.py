#!/usr/bin/env python3
"""Statistical screen of indicator conditions/combinations (development data only) with BH-FDR control."""
import argparse

import pandas as pd

from cryptoalgo.config import load_config
from cryptoalgo.data.loader import load_data
from cryptoalgo.research.study import run_study

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--horizon", type=int, default=12)
ap.add_argument("--dev-fraction", type=float, default=0.70, help="only the first fraction of history is used")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
cfg = load_config(a.config)
data, prov = load_data(a.data, cfg.market.symbols, cfg.market.timeframe)
cut = int(min(len(d) for d in data.values()) * a.dev_fraction)
data = {s: d.iloc[:cut] for s, d in data.items()}
c = cfg.costs
t = run_study(data, a.horizon, 2 * (c.taker_fee_bps + c.half_spread_bps + c.slippage_bps))
print("DATA:", prov, f"(development slice: {cut} bars)")
print(f"{len(t)} hypotheses tested; {int(t['significant'].sum())} significant at FDR 10% (net of costs)\n")
pd.set_option("display.width", 200)
print(t.head(a.top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
