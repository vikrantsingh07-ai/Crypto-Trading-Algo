#!/usr/bin/env python3
"""Exit 0 if the paper trader's heartbeat is fresh, else 1. For systemd/Docker/cron watchdogs."""
import argparse
import sys

from cryptoalgo.config import load_config
from cryptoalgo.monitoring import check_health

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--max-age", type=float, default=300.0)
a = ap.parse_args()
ok, msg = check_health(load_config(a.config).paper.health_file, a.max_age)
print(("HEALTHY: " if ok else "UNHEALTHY: ") + msg)
sys.exit(0 if ok else 1)
