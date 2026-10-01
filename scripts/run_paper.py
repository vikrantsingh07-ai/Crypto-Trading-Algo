#!/usr/bin/env python3
"""PAPER TRADING runner (virtual INR 1,00,000). Never places real orders.

  Live market data (public Binance klines), 24/7:
      python scripts/run_paper.py --dashboard
  Simulated feed (no network) for demos / soak tests:
      python scripts/run_paper.py --simulate synthetic:structured:7:600 --sim-bars 3000 --allow-unvalidated --dashboard

Refuses to start unless reports/validation.json says every robustness gate passed (override: --allow-unvalidated,
which labels the run UNVALIDATED everywhere).
"""
import argparse
import logging
import signal
import sys
import time

from cryptoalgo.config import load_config
from cryptoalgo.dashboard.server import serve
from cryptoalgo.data.loader import load_data
from cryptoalgo.data.models import tf_seconds
from cryptoalgo.monitoring import setup_logging
from cryptoalgo.paper.feeds import BinanceFeed, SimulatedFeed
from cryptoalgo.paper.runner import PaperTrader, ValidationRequired, load_validated_strategy
from cryptoalgo.safety import PAPER_BANNER, SafetyError, assert_paper_only

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default=None)
ap.add_argument("--db", default=None)
ap.add_argument("--allow-unvalidated", action="store_true")
ap.add_argument("--dashboard", action="store_true", help="also serve the dashboard from this process")
ap.add_argument("--simulate", default=None, help="synthetic:<world>:<seed>:<days> -> use a simulated feed")
ap.add_argument("--sim-bars", type=int, default=2000, help="bars to replay live in simulation")
ap.add_argument("--speed", type=float, default=0, help="simulated bars per second (0 = as fast as possible)")
ap.add_argument("--log-file", default="runtime/paper.log")
a = ap.parse_args()

cfg = load_config(a.config)
log = setup_logging(a.log_file)
try:
    assert_paper_only(cfg)
    strategy, info = load_validated_strategy(cfg, a.allow_unvalidated)
except (ValidationRequired, SafetyError) as e:
    print(f"\n{e}\n", file=sys.stderr)
    sys.exit(2)
print(PAPER_BANNER, "|", info["label"])

if a.simulate:
    data, prov = load_data(a.simulate, cfg.market.symbols, cfg.market.timeframe)
    print("SIMULATED FEED:", prov)
    info = {**info, "label": f"{info['label']} | {prov}"}
    n = min(len(d) for d in data.values())
    feed = SimulatedFeed(data, cfg.market.timeframe, start_pos=n - a.sim_bars)
    trader = PaperTrader(cfg, feed, strategy, a.db, clock=lambda: feed.now.timestamp(), sleeper=lambda s: None, info=info)
else:
    feed = BinanceFeed()
    trader = PaperTrader(cfg, feed, strategy, a.db, info=info)

if a.dashboard:
    serve(a.db or cfg.paper.db_path, cfg.dashboard.host, cfg.dashboard.port, cfg.account.initial_capital,
          cfg.paper.health_file, background=True)
    print(f"dashboard: http://{cfg.dashboard.host}:{cfg.dashboard.port}")

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: trader.stop())

if a.simulate:
    trader.start()
    while not feed.exhausted and not trader.stop_event.is_set():
        feed.release(1)
        try:
            trader.poll_once()
        except Exception as e:
            trader.handle_error(e)
        if a.speed:
            time.sleep(1 / a.speed)
    print(f"simulation finished: {trader.bars_processed} bars, equity INR {trader.engine.equity:,.0f}")
    if a.dashboard:
        print("dashboard still serving; Ctrl-C to exit")
        trader.stop_event.wait()
else:
    trader.run_forever()
