#!/usr/bin/env python3
"""Write SYNTHETIC candles to CSV (testing machinery only; NOT real market data)."""
import argparse

from cryptoalgo.data.loader import save_csv
from cryptoalgo.data.synthetic import generate

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="structured", choices=["null", "structured", "strong_trend"])
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--days", type=int, default=1460)
ap.add_argument("--timeframe", default="1h")
ap.add_argument("--out", default="data/synthetic")
a = ap.parse_args()
data, reg = generate(days=a.days, timeframe=a.timeframe, seed=a.seed, world=a.world)
for s, df in data.items():
    save_csv(df, f"{a.out}/{s}_{a.timeframe}.csv")
    print("wrote", f"{a.out}/{s}_{a.timeframe}.csv", len(df))
