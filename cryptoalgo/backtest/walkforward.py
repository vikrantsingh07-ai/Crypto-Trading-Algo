"""Walk-forward optimisation with a pre-declared parameter grid.

For each window: choose parameters using ONLY the training slice, then trade the
following unseen test slice with them. If nothing is profitable in training the
rule is "stay flat" (no trading) for the test slice -- a strategy with no
in-sample edge does not get to trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import Config
from ..strategies.library import get_strategy
from .metrics import compute_metrics
from .runner import common_index, precompute, run_backtest


@dataclass
class WFWindow:
    train: tuple
    test: tuple
    params: dict | None
    train_metrics: dict
    test_metrics: dict
    test_trades: pd.DataFrame
    test_equity: pd.Series


@dataclass
class WFResult:
    windows: list = field(default_factory=list)
    oos_trades: pd.DataFrame = None
    oos_equity: pd.Series = None
    summary: dict = field(default_factory=dict)


def objective(m: dict, min_trades: int = 20) -> float:
    """Risk-adjusted in-sample score: net profit per unit of drawdown, needs a minimum sample."""
    if m["trades"] < min_trades or m["net_pnl"] <= 0 or m["profit_factor"] <= 1.0:
        return float("-inf")
    return m["net_pnl"] / max(m["max_drawdown_pct"], 1.0)


def walk_forward(data: dict, cfg: Config, strategy_name: str, grid: list[dict] | None = None,
                 train_days: int = 365, test_days: int = 90, step_days: int | None = None,
                 use_htf: bool = False, min_train_trades: int = 20, start=None, end=None) -> WFResult:
    tf = cfg.market.timeframe
    idx = common_index(data)
    t0 = pd.Timestamp(start) if start is not None else idx[0]
    t_end = pd.Timestamp(end) if end is not None else idx[-1] + pd.Timedelta(seconds=1)
    step = pd.Timedelta(days=step_days or test_days)
    base = get_strategy(strategy_name, None, tf, cfg.market.htf, use_htf)
    grid = grid or type(base).param_grid()
    cache: dict[int, tuple] = {}
    for gi, params in enumerate(grid):
        st = get_strategy(strategy_name, params, tf, cfg.market.htf, use_htf)
        cache[gi] = (st, precompute(data, st))
    res = WFResult()
    cur = t0
    running = cfg.account.initial_capital
    eq_parts, tr_parts = [], []
    while cur + pd.Timedelta(days=train_days + test_days) <= t_end:
        tr0, tr1 = cur, cur + pd.Timedelta(days=train_days)
        te0, te1 = tr1, tr1 + pd.Timedelta(days=test_days)
        best, best_score, best_m = None, float("-inf"), None
        for gi, (st, rows) in cache.items():
            r = run_backtest(data, cfg, st, tf, tr0, tr1, rows)
            m = compute_metrics(r.trades, r.equity, cfg.account.initial_capital, tf)
            sc = objective(m, min_train_trades)
            if sc > best_score:
                best, best_score, best_m = gi, sc, m
        wcfg = cfg.copy()
        wcfg.account.initial_capital = running
        if best is None:
            tm = compute_metrics(pd.DataFrame(columns=["net_pnl"]), pd.Series(dtype=float), running, tf)
            win = WFWindow((tr0, tr1), (te0, te1), None, best_m or {}, tm, pd.DataFrame(), pd.Series(dtype=float))
        else:
            st, rows = cache[best]
            r = run_backtest(data, wcfg, st, tf, te0, te1, rows)
            tm = compute_metrics(r.trades, r.equity, running, tf)
            win = WFWindow((tr0, tr1), (te0, te1), dict(st.params), best_m, tm, r.trades, r.equity)
            if len(r.equity):
                running = float(r.equity.iloc[-1])
                eq_parts.append(r.equity)
            tr_parts.append(r.trades)
        res.windows.append(win)
        cur += step
    res.oos_trades = pd.concat(tr_parts, ignore_index=True) if tr_parts else pd.DataFrame()
    res.oos_equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    traded = [w for w in res.windows if w.params is not None]
    pos = [w for w in traded if w.test_metrics["net_pnl"] > 0]
    res.summary = {"windows": len(res.windows), "traded_windows": len(traded), "flat_windows": len(res.windows) - len(traded),
                   "positive_windows": len(pos),
                   "positive_window_fraction": (len(pos) / len(res.windows)) if res.windows else 0.0,
                   "positive_fraction_of_traded": (len(pos) / len(traded)) if traded else 0.0,
                   "oos_net_pnl": running - cfg.account.initial_capital}
    return res
