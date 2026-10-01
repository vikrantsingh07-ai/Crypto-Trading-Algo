"""Paper-only safety guard.

This project contains NO code path that can place a real order:
  * the only broker is `execution.PaperBroker` (pure in-memory simulation),
  * the only exchange client is `data.binance.BinancePublicClient`, which calls
    unauthenticated public market-data endpoints and has no way to accept keys,
  * no signing/HMAC code exists.

Real trading would require a deliberate, separate implementation (see
docs/RISK_AND_SAFETY.md). This module makes accidental enabling loud.
"""
from __future__ import annotations

import os

PAPER_BANNER = "PAPER TRADING / DEMO - VIRTUAL FUNDS ONLY - NO REAL ORDERS"

# If any of these are present in the environment the process refuses to start:
# credentials have no business being near a paper-only system.
_FORBIDDEN_ENV = ("BINANCE_API_KEY", "BINANCE_API_SECRET", "EXCHANGE_API_KEY",
                   "EXCHANGE_API_SECRET", "LIVE_TRADING", "ENABLE_LIVE_TRADING")


class SafetyError(RuntimeError):
    pass


def assert_paper_only(cfg) -> None:
    if getattr(cfg, "mode", None) != "paper":
        raise SafetyError(f"refusing to start: mode={getattr(cfg, 'mode', None)!r}. {PAPER_BANNER}")
    present = [k for k in _FORBIDDEN_ENV if os.environ.get(k)]
    if present:
        raise SafetyError(
            "refusing to start: exchange credentials / live-trading flags found in environment "
            f"({', '.join(present)}). Remove them; this system is paper-only.")
