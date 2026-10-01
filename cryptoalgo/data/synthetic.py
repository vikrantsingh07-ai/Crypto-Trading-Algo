"""SYNTHETIC market generator -- for testing machinery only.

Nothing generated here is evidence about real-market profitability. Two worlds:

* ``world="null"``: efficient market. Student-t returns with GARCH-like volatility
  clustering, zero drift, zero autocorrelation. A sound validation pipeline MUST
  reject every strategy here (false-positive control).
* ``world="strong_trend"``: like "structured" but with large, persistent trend drifts (+/-250% annualised
  at ~45% vol, regimes lasting 60-200 days, 70% of the time in bull/bear, strong momentum autocorrelation). A deliberately EASY market that contains a real, tradable
  edge; used to prove the validator can say PASS when an edge truly exists (power check).
* ``world="structured"``: regimes with drift (bull/bear), mean-reversion
  (sideways), vol shocks (high_vol) and calm (low_vol), plus mild return
  autocorrelation inside trends. A pipeline that cannot detect edge here is
  too insensitive. Because the structure is injected by construction, success
  here proves detection power only.

BTC and ETH share regime segments and correlated shocks (rho ~ 0.8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import tf_seconds

REGIMES = ["bull", "bear", "sideways", "high_vol", "low_vol"]
# annual drift, annual vol, ar1 phi, OU strength per bar
_PARAMS = {
    "bull":     (0.90, 0.55, +0.06, 0.0),
    "bear":     (-0.80, 0.65, +0.06, 0.0),
    "sideways": (0.00, 0.40, -0.05, 0.015),
    "high_vol": (0.00, 1.40, 0.00, 0.0),
    "low_vol":  (0.05, 0.22, 0.00, 0.0),
}


_STRONG = {
    "bull":     (2.5, 0.45, +0.12, 0.0),
    "bear":     (-2.2, 0.50, +0.12, 0.0),
    "sideways": (0.00, 0.30, -0.05, 0.015),
    "high_vol": (0.00, 1.00, 0.00, 0.0),
    "low_vol":  (0.10, 0.20, 0.00, 0.0),
}


_STRONG_WEIGHTS = [0.40, 0.30, 0.20, 0.0, 0.10]       # bull, bear, sideways, high_vol, low_vol


def _segments(n_bars: int, bars_per_day: float, rng: np.random.Generator, lo_days: int = 20, hi_days: int = 90,
              weights=None) -> np.ndarray:
    """Array of regime ids per bar, random segment lengths of lo_days..hi_days."""
    out = np.empty(n_bars, dtype=np.int8)
    i = 0
    prev = -1
    p = None if weights is None else np.asarray(weights) / np.sum(weights)
    while i < n_bars:
        r = int(rng.choice(len(REGIMES), p=p))
        while r == prev:
            r = int(rng.choice(len(REGIMES), p=p))
        prev = r
        length = int(rng.integers(lo_days, hi_days + 1) * bars_per_day)
        out[i:i + length] = r
        i += length
    return out


def generate(symbols=("BTCUSDT", "ETHUSDT"), start="2021-01-01", days=1095, timeframe="1h",
             seed=7, world="structured", start_prices=None, sub_steps=8, edge_scale=1.0):
    """Return (dict[symbol -> OHLCV DataFrame], regime Series of names).

    ``edge_scale`` multiplies the regime drifts (power checks only; >1 plants an unrealistically strong edge).
    """
    if world not in ("null", "structured", "strong_trend"):
        raise ValueError("world must be 'null', 'structured' or 'strong_trend'")
    table = _STRONG if world == "strong_trend" else _PARAMS
    rng = np.random.default_rng(seed)
    spb = tf_seconds(timeframe)
    bpd = 86400 / spb
    n = int(days * bpd)
    idx = pd.date_range(start=start, periods=n, freq=f"{spb}s", tz="UTC")
    year_bars = 365 * bpd

    if world != "null":
        seg = _segments(n, bpd, rng, *((60, 200) if world == "strong_trend" else (20, 90)),
                        weights=_STRONG_WEIGHTS if world == "strong_trend" else None)
    else:
        seg = np.full(n, REGIMES.index("sideways"), dtype=np.int8)  # label only; dynamics are null
    regime_names = pd.Series(np.array(REGIMES)[seg] if world != "null" else "null", index=idx, name="regime")

    start_prices = start_prices or {"BTCUSDT": 30000.0, "ETHUSDT": 1800.0}
    vol_scale = {"BTCUSDT": 1.0, "ETHUSDT": 1.25}
    rho = 0.8
    shared = rng.standard_t(4, n) / np.sqrt(2.0)   # unit variance (t4 var = 2)
    out = {}
    for k, sym in enumerate(symbols):
        own = rng.standard_t(4, n) / np.sqrt(2.0)
        z = shared if k == 0 else rho * shared + np.sqrt(1 - rho ** 2) * own
        base_vol = 0.60 if world == "null" else None
        sigma = np.empty(n)
        mu = np.zeros(n)
        phi = np.zeros(n)
        ou = np.zeros(n)
        for r_i, name in enumerate(REGIMES):
            m = seg == r_i
            d, v, p, o = table[name]
            if world == "null":
                continue
            sigma[m] = v * vol_scale.get(sym, 1.1) / np.sqrt(year_bars)
            mu[m] = d * edge_scale / year_bars
            phi[m] = p
            ou[m] = o
        if world == "null":
            sigma[:] = base_vol * vol_scale.get(sym, 1.1) / np.sqrt(year_bars)
        # GARCH-like clustering: multiplicative AR(1) log-vol factor
        lv = np.zeros(n)
        eps = rng.normal(0, 0.06, n)
        for i in range(1, n):
            lv[i] = 0.97 * lv[i - 1] + eps[i]
        sigma = sigma * np.exp(lv)
        ret = np.zeros(n)
        logp = np.log(start_prices.get(sym, 100.0))
        anchor = logp
        logp_path = np.empty(n)
        prev_r = 0.0
        for i in range(n):
            if i > 0 and seg[i] != seg[i - 1]:
                anchor = logp          # sideways anchor resets at segment start
            r = mu[i] + phi[i] * prev_r + sigma[i] * z[i] - ou[i] * (logp - anchor)
            r -= 0.5 * sigma[i] ** 2
            logp += r
            logp_path[i] = logp
            ret[i] = r
            prev_r = r
        close = np.exp(logp_path)
        openp = np.concatenate([[start_prices.get(sym, 100.0)], close[:-1]])
        # intrabar path for realistic high/low
        sub = rng.normal(0, 1, (n, sub_steps)) * (sigma[:, None] / np.sqrt(sub_steps))
        cum = np.cumsum(sub, axis=1)
        cum = cum - cum[:, -1:] * (np.arange(1, sub_steps + 1)[None, :] / sub_steps)  # bridge to 0 at end
        path = np.log(openp)[:, None] + cum + (np.log(close) - np.log(openp))[:, None] * \
            (np.arange(1, sub_steps + 1)[None, :] / sub_steps)
        high = np.maximum(np.exp(path.max(axis=1)), np.maximum(openp, close))
        low = np.minimum(np.exp(path.min(axis=1)), np.minimum(openp, close))
        hours = (idx.hour.values + idx.minute.values / 60)
        season = 1 + 0.35 * np.sin((hours - 8) / 24 * 2 * np.pi)       # intraday volume seasonality
        volume = 100 * season * (1 + 1.5 * np.abs(z)) * rng.lognormal(0, 0.35, n)
        out[sym] = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close,
                                 "volume": volume}, index=idx)
    return out, regime_names
