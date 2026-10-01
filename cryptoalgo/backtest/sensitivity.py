"""Parameter-sensitivity test: perturb each numeric parameter by about -30% / +30%.

A real edge should degrade gracefully; an overfit one collapses when parameters move a little.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..strategies.library import get_strategy
from .metrics import compute_metrics
from .runner import precompute, run_backtest


def sensitivity(data: dict, cfg: Config, strategy_name: str, params: dict, start=None, end=None,
                use_htf: bool = False) -> dict:
    tf = cfg.market.timeframe
    base = get_strategy(strategy_name, params, tf, cfg.market.htf, use_htf)
    variants = [("base", {})] + [(f"{k}={v}", {k: v}) for d in base.neighbours() for k, v in d.items()]
    rows = []
    for label, kw in variants:
        st = base.with_params(**kw)
        r = run_backtest(data, cfg, st, tf, start, end, precompute(data, st))
        m = compute_metrics(r.trades, r.equity, cfg.account.initial_capital, tf)
        rows.append({"variant": label, "trades": m["trades"], "net_pnl": m["net_pnl"],
                     "profit_factor": m["profit_factor"], "max_dd_pct": m["max_drawdown_pct"], "sharpe": m["sharpe"]})
    nb = [r for r in rows if r["variant"] != "base"]
    prof = [r for r in nb if r["net_pnl"] > 0 and r["profit_factor"] > 1.0]
    return {"rows": rows, "n_neighbours": len(nb), "fraction_profitable": (len(prof) / len(nb)) if nb else 0.0,
            "median_pf": float(np.median([r["profit_factor"] for r in nb])) if nb else 0.0,
            "worst_net_pnl": float(min(r["net_pnl"] for r in nb)) if nb else 0.0}
