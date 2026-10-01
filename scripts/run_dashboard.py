#!/usr/bin/env python3
"""Serve the read-only PAPER TRADING dashboard for a paper DB."""
import argparse

from cryptoalgo.config import load_config
from cryptoalgo.dashboard.server import serve

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--db", default=None)
ap.add_argument("--host", default=None)
ap.add_argument("--port", type=int, default=None)
a = ap.parse_args()
cfg = load_config(a.config)
host, port = a.host or cfg.dashboard.host, a.port or cfg.dashboard.port
print(f"PAPER TRADING dashboard on http://{host}:{port}  (db={a.db or cfg.paper.db_path})")
serve(a.db or cfg.paper.db_path, host, port, cfg.account.initial_capital, cfg.paper.health_file)
