"""Monte Carlo on realised trade P&L (INR, additive on initial capital).

Two resampling schemes:
  * 'bootstrap': trades drawn with replacement (tests dependence on the particular sample)
  * 'shuffle'  : same trades, random order (tests path dependence: drawdown / ruin)
"""
from __future__ import annotations

import numpy as np


def monte_carlo(pnls, initial: float, n_sims: int = 5000, seed: int = 11, mode: str = "bootstrap",
                block: int = 1, dd_limit_pct: float = 25.0) -> dict:
    x = np.asarray(pnls, dtype=float)
    n = len(x)
    if n < 5:
        return {"n_trades": n, "insufficient": True}
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        idx = np.argsort(rng.random((n_sims, n)), axis=1)
    elif block > 1:
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, (n_sims, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_sims, -1)[:, :n]
    else:
        idx = rng.integers(0, n, (n_sims, n))
    paths = initial + np.cumsum(x[idx], axis=1)
    paths = np.concatenate([np.full((n_sims, 1), initial), paths], axis=1)
    peak = np.maximum.accumulate(paths, axis=1)
    dd = ((peak - paths) / peak * 100).max(axis=1)
    final = paths[:, -1]
    q = lambda a, p: float(np.percentile(a, p))
    return {"mode": mode, "block": block, "n_trades": n, "n_sims": n_sims,
            "final_equity": {"p5": q(final, 5), "p50": q(final, 50), "p95": q(final, 95)},
            "max_dd_pct": {"p50": q(dd, 50), "p95": q(dd, 95), "p99": q(dd, 99)},
            "prob_loss": float((final < initial).mean()),
            "prob_dd_exceeds_limit": float((dd > dd_limit_pct).mean()), "dd_limit_pct": dd_limit_pct,
            "prob_ruin_50pct": float((paths.min(axis=1) < initial * 0.5).mean())}
