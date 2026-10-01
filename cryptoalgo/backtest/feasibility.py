"""Is the INR daily-profit target achievable under the tested risk constraints?

We never tune anything toward the target. We measure what the (already
validated, untouched) strategy delivered, quantify the uncertainty, then ask
what scaling the target would require and whether the risk limits permit it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import Config
from .metrics import daily_pnl_series


def stationary_bootstrap_mean(x: np.ndarray, n_boot: int = 5000, mean_block: int = 7, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        rnd = rng.integers(0, n, n)
        for i in range(1, n):
            idx[i] = rnd[i] if jumps[i] else (idx[i - 1] + 1) % n
        out[b] = x[idx].mean()
    return out


def assess_target(equity: pd.Series, initial: float, cfg: Config, max_dd_pct_observed: float,
                  target: float | None = None, n_boot: int = 2000) -> dict:
    target = target or cfg.gates.target_daily_pnl
    d = daily_pnl_series(equity, initial)
    out: dict = {"target_daily_pnl": target, "target_pct_of_capital": target / initial * 100,
                 "required_annual_return_simple_pct": target * 365 / initial * 100,
                 "required_annual_return_compounded_pct": ((1 + target / initial) ** 365 - 1) * 100,
                 "days": int(len(d))}
    if len(d) < 30:
        out.update(verdict="INSUFFICIENT DATA", achievable=False)
        return out
    x = d.to_numpy()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    boot = stationary_bootstrap_mean(x, n_boot)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    out.update(mean_daily_pnl=mean, std_daily_pnl=sd, mean_ci95=[lo, hi],
               prob_mean_ge_target=float((boot >= target).mean()),
               fraction_days_ge_target=float((x >= target).mean()),
               median_daily_pnl=float(np.median(x)),
               sharpe_needed_for_target=(target / sd * math.sqrt(365)) if sd > 0 else None,
               sharpe_achieved=(mean / sd * math.sqrt(365)) if sd > 0 else 0.0)
    r = cfg.risk
    if mean > 0:
        k = target / mean
        out["scale_needed"] = k
        out["risk_per_trade_needed_pct"] = r.risk_per_trade_pct * k
        out["implied_max_drawdown_pct"] = max_dd_pct_observed * k
        out["capital_needed_for_target_at_current_return"] = initial * k
        out["breaches_risk_limits"] = bool(r.risk_per_trade_pct * k > 5.0 or max_dd_pct_observed * k > r.max_drawdown_pct)
    else:
        out.update(scale_needed=None, breaches_risk_limits=True)
    ok = (mean >= target and lo > 0 and out["prob_mean_ge_target"] >= 0.8 and max_dd_pct_observed <= r.max_drawdown_pct)
    out["achievable"] = bool(ok)
    if ok:
        out["verdict"] = "TARGET MET in this sample (still no guarantee; see limitations)"
    elif mean <= 0 or lo <= 0:
        out["verdict"] = ("NOT ACHIEVABLE: mean daily P&L is not distinguishable from (or is below) zero, "
                          "so no scaling of this strategy can reach the target")
    else:
        out["verdict"] = (f"NOT ACHIEVABLE within risk limits: mean daily P&L is INR {mean:,.0f}; reaching INR {target:,.0f} "
                          f"needs ~{out['scale_needed']:.1f}x the risk, implying ~{out['implied_max_drawdown_pct']:.0f}% "
                          f"drawdown vs a {r.max_drawdown_pct:.0f}% limit")
    out["after_india_tax_note"] = ("India taxes crypto gains at a flat 30% with no loss set-off, plus 1% TDS on sells; "
                                   "net-of-tax daily profit would be roughly "
                                   f"INR {max(mean, 0) * 0.7:,.0f} before TDS/liquidity effects (not modelled).")
    return out
