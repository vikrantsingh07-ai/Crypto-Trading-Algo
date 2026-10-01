"""Logging setup, health file, and a health-check usable by systemd/Docker/cron."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path

from .safety import PAPER_BANNER


def setup_logging(log_file: str | None = "runtime/paper.log", level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("cryptoalgo")
    if root.handlers:
        return root
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    return root


def write_health(path: str, payload: dict) -> None:
    """Atomic write so a reader never sees a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"banner": PAPER_BANNER, **payload}, default=str, indent=2))
    os.replace(tmp, p)


def check_health(path: str, max_poll_age_s: float = 300.0, now: float | None = None) -> tuple[bool, str]:
    """(healthy, message). Healthy = recent heartbeat and status not FAILED."""
    now = now if now is not None else time.time()
    p = Path(path)
    if not p.exists():
        return False, "no health file"
    try:
        h = json.loads(p.read_text())
    except Exception as e:
        return False, f"unreadable health file: {e}"
    age = now - float(h.get("last_poll_epoch", 0))
    if age > max_poll_age_s:
        return False, f"stale heartbeat ({age:.0f}s)"
    if h.get("status") == "FAILED":
        return False, f"status FAILED: {h.get('last_error')}"
    return True, f"{h.get('status')} (heartbeat {age:.0f}s ago)"
