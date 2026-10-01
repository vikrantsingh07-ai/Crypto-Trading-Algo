#!/usr/bin/env python3
"""Full validation protocol -> reports/<label>.{json,md,html}. Paper trading requires reports/validation.json to PASS.

  python scripts/run_validation.py --data data/            # real CSVs from fetch_data.py
  python scripts/run_validation.py --data synthetic:null:1:1460 --label null_control
"""
import argparse

from cryptoalgo.backtest.pipeline import run_validation
from cryptoalgo.config import load_config
from cryptoalgo.data.loader import load_data

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--data", required=True, help="CSV directory or synthetic:<world>:<seed>:<days>")
ap.add_argument("--label", default="validation")
ap.add_argument("--out", default="reports")
ap.add_argument("--htf", action="store_true", help="enable higher-timeframe confirmation")
ap.add_argument("--no-research", action="store_true")
ap.add_argument("--strategies", nargs="*", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
data, prov = load_data(a.data, cfg.market.symbols, cfg.market.timeframe)
print("DATA:", prov)
rep = run_validation(data, cfg, a.strategies, a.htf, a.out, a.label, do_research=not a.no_research)
print(f"\nRESULT: {'PASSED' if rep['passed'] else 'FAILED'} gates  |  chosen={rep['chosen_strategy']}  params={rep['params']}")
for g in rep["gates"]:
    print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['gate']}: {g['value']} (need {g['threshold']})")
print("\nTarget feasibility:", rep["feasibility_holdout"]["verdict"])
print(f"Reports: {a.out}/{a.label}.md  .json  .html   (data provenance: {prov})")
