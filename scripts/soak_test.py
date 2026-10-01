#!/usr/bin/env python3
"""Accelerated long-running stability run with random faults and restarts (PAPER ONLY, simulated feed).

  python scripts/soak_test.py --bars 20000 --seed 1
Prints invariants at the end; exit code 1 if any is violated. For a *wall-clock* soak, run scripts/run_paper.py
for days against the live feed and watch the dashboard / healthcheck.
"""
import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

from cryptoalgo.config import load_config
from cryptoalgo.data.synthetic import generate
from cryptoalgo.paper.feeds import SimulatedFeed
from cryptoalgo.paper.runner import PaperTrader
from cryptoalgo.storage.db import connect
from cryptoalgo.strategies.library import get_strategy

ap = argparse.ArgumentParser()
ap.add_argument("--bars", type=int, default=20000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--strategy", default="regime_ensemble")
a = ap.parse_args()

cfg = load_config()
tmp = Path(tempfile.mkdtemp(prefix="soak_"))
cfg.paper.db_path, cfg.paper.health_file = str(tmp / "soak.db"), str(tmp / "health.json")
cfg.paper.history_bars = 1500
days = int(a.bars / 24) + 120
data, _ = generate(tuple(cfg.market.symbols), days=days, world="structured", seed=a.seed)
n = min(len(d) for d in data.values())
feed = SimulatedFeed(data, cfg.market.timeframe, n - a.bars)
st = get_strategy(a.strategy, None, cfg.market.timeframe, cfg.market.htf)


def new_trader():
    return PaperTrader(cfg, feed, st, cfg.paper.db_path, clock=lambda: feed.now.timestamp(), sleeper=lambda s: None,
                       info={"label": "SOAK (simulated)"}, health_path=cfg.paper.health_file)


t = new_trader(); t.start()
rnd = random.Random(a.seed)
done = restarts = errors = 0
t0 = time.time()
while done < a.bars:
    k = min(rnd.choice([1, 4, 12]), a.bars - done)
    feed.release(k); done += k
    feed.overlap = rnd.choice([0, 0, 5])
    if rnd.random() < 0.05:
        feed.fail_next(rnd.randint(1, 3))
    if rnd.random() < 0.01:
        t.conn.close(); t = new_trader(); t.start(); restarts += 1
    for _ in range(50):
        try:
            t.poll_once(); break
        except Exception as e:
            errors += 1; t.handle_error(e)
    if done % 2000 < k:
        print(f"  {done}/{a.bars} bars  equity {t.engine.equity:,.0f}  errors {errors} restarts {restarts}  {time.time()-t0:.0f}s", flush=True)

c = connect(cfg.paper.db_path, read_only=True)
q = lambda s: c.execute(s).fetchone()[0]
checks = {
    "no duplicate order ids": q("SELECT COUNT(*)-COUNT(DISTINCT order_id) FROM orders") == 0,
    "no duplicate trade ids": q("SELECT COUNT(*)-COUNT(DISTINCT trade_id) FROM trades") == 0,
    "orders == 2*trades + open positions": q("SELECT COUNT(*) FROM orders") == 2 * q("SELECT COUNT(*) FROM trades") + len(t.engine.pf.positions),
    "equity rows unique per bar": q("SELECT COUNT(*)-COUNT(DISTINCT ts_ms) FROM equity") == 0,
    "cash identity": abs(t.engine.pf.cash - (cfg.account.initial_capital + q("SELECT COALESCE(SUM(net_pnl),0) FROM trades")
                         - sum(p.entry_fee_inr + p.funding_inr for p in t.engine.pf.positions.values()))) < 1e-4,
}
print(f"\nbars={a.bars} trades={q('SELECT COUNT(*) FROM trades')} errors_handled={errors} restarts={restarts} equity={t.engine.equity:,.0f}")
for k, v in checks.items():
    print(("OK   " if v else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
