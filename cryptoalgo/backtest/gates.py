"""Pre-declared robustness gates. Paper trading is refused unless all pass."""
from __future__ import annotations

from ..config import Gates


def evaluate_gates(g: Gates, holdout: dict, wf_summary: dict, wf_oos: dict, mc: dict, sens: dict,
                   stressed: dict, per_asset_net: dict) -> tuple[bool, list[dict]]:
    """All inputs are plain dicts of metrics produced by the validation pipeline."""
    rows = []

    def add(name, value, threshold, ok, note=""):
        rows.append({"gate": name, "value": value, "threshold": threshold, "pass": bool(ok), "note": note})

    add("holdout profit factor", holdout.get("profit_factor"), f">= {g.min_profit_factor}",
        holdout.get("profit_factor", 0) >= g.min_profit_factor and holdout.get("net_pnl", 0) > 0)
    add("holdout trades", holdout.get("trades"), f">= {g.min_trades}", holdout.get("trades", 0) >= g.min_trades)
    add("walk-forward OOS trades", wf_oos.get("trades"), f">= {g.min_trades}", wf_oos.get("trades", 0) >= g.min_trades)
    add("walk-forward OOS profit factor", wf_oos.get("profit_factor"), f">= {g.min_profit_factor}",
        wf_oos.get("profit_factor", 0) >= g.min_profit_factor and wf_oos.get("net_pnl", 0) > 0)
    add("walk-forward positive windows", round(wf_summary.get("positive_window_fraction", 0), 3),
        f">= {g.min_wf_positive_windows}", wf_summary.get("positive_window_fraction", 0) >= g.min_wf_positive_windows,
        "flat windows count as not positive")
    add("holdout max drawdown %", round(holdout.get("max_drawdown_pct", 100), 2), f"<= {g.max_drawdown_pct}",
        holdout.get("max_drawdown_pct", 100) <= g.max_drawdown_pct)
    add("holdout Sharpe", round(holdout.get("sharpe", 0), 2), f">= {g.min_sharpe}", holdout.get("sharpe", 0) >= g.min_sharpe)
    mc_dd = (mc.get("max_dd_pct") or {}).get("p95", 100.0)
    mc_f5 = (mc.get("final_equity") or {}).get("p5", 0.0)
    add("Monte-Carlo 95th pct drawdown %", round(mc_dd, 2), f"<= {g.mc_dd95_limit_pct}",
        (not mc.get("insufficient")) and mc_dd <= g.mc_dd95_limit_pct)
    add("Monte-Carlo 5th pct final equity", round(mc_f5, 0), "> initial capital",
        (not mc.get("insufficient")) and mc_f5 > holdout.get("initial_capital", 0))
    add("parameter sensitivity (neighbours profitable)", round(sens.get("fraction_profitable", 0), 3),
        f">= {g.min_sensitivity_positive}", sens.get("fraction_profitable", 0) >= g.min_sensitivity_positive)
    add(f"holdout net profit at {g.cost_stress_multiplier}x costs", round(stressed.get("net_pnl", 0), 0), "> 0",
        stressed.get("net_pnl", 0) > 0)
    n_prof = sum(1 for v in per_asset_net.values() if v > 0)
    add("assets profitable (holdout)", n_prof, f">= {g.min_assets_profitable}", n_prof >= g.min_assets_profitable)
    return all(r["pass"] for r in rows), rows
