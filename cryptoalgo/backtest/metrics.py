"""Performance metrics for a BacktestResult (or any trades + equity pair)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PF_CAP = 99.0


def _pf(pnl: pd.Series) -> float:
    w, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if l <= 0:
        return PF_CAP if w > 0 else 0.0
    return min(w / l, PF_CAP)


def _streaks(pnl: pd.Series) -> tuple[int, int]:
    best_w = best_l = cw = cl = 0
    for v in pnl.to_numpy():
        if v > 0:
            cw, cl = cw + 1, 0
        elif v < 0:
            cl, cw = cl + 1, 0
        else:
            cw = cl = 0
        best_w, best_l = max(best_w, cw), max(best_l, cl)
    return best_w, best_l


def daily_equity(equity: pd.Series, initial: float) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    d = equity.resample("1D").last().ffill()
    return d


def drawdown_series(equity: pd.Series, initial: float) -> pd.Series:
    e = pd.concat([pd.Series([initial]), equity.reset_index(drop=True)], ignore_index=True)
    return (e.cummax() - e) / e.cummax() * 100


def side_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "net_pnl": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "avg_trade": 0.0}
    return {"trades": int(len(t)), "net_pnl": float(t["net_pnl"].sum()),
            "win_rate_pct": float((t["net_pnl"] > 0).mean() * 100),
            "profit_factor": float(_pf(t["net_pnl"])), "avg_trade": float(t["net_pnl"].mean())}


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, initial: float, timeframe: str = "1h") -> dict:
    m: dict = {"initial_capital": initial}
    final = float(equity.iloc[-1]) if len(equity) else initial
    net = final - initial
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400 + 1, 1) if len(equity) > 1 else 1
    m.update(final_balance=final, net_pnl=net, total_return_pct=net / initial * 100, calendar_days=float(days),
             daily_avg_pnl=net / days)
    pnl = trades["net_pnl"] if len(trades) else pd.Series(dtype=float)
    n = len(trades)
    m["trades"] = n
    if n:
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        m.update(win_rate_pct=len(wins) / n * 100, loss_rate_pct=len(losses) / n * 100,
                 avg_win=float(wins.mean()) if len(wins) else 0.0, avg_loss=float(losses.mean()) if len(losses) else 0.0,
                 risk_reward=float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else 0.0,
                 profit_factor=float(_pf(pnl)), avg_trade=float(pnl.mean()), expectancy_r=float(trades["r_multiple"].mean()),
                 best_trade=float(pnl.max()), worst_trade=float(pnl.min()), total_fees=float(trades["fees"].sum()),
                 total_funding=float(trades["funding"].sum()), avg_bars_held=float(trades["bars_held"].mean()),
                 trades_per_day=n / days)
        m["max_consecutive_wins"], m["max_consecutive_losses"] = _streaks(pnl)
        m["long"] = side_stats(trades[trades["side"] == "long"])
        m["short"] = side_stats(trades[trades["side"] == "short"])
        m["by_symbol"] = {s: side_stats(g) for s, g in trades.groupby("symbol")}
        m["by_exit_reason"] = {s: {"trades": int(len(g)), "net_pnl": float(g["net_pnl"].sum())}
                               for s, g in trades.groupby("exit_reason")}
        m["by_source"] = {s: side_stats(g) for s, g in trades.groupby("source")}
    else:
        m.update(win_rate_pct=0.0, loss_rate_pct=0.0, avg_win=0.0, avg_loss=0.0, risk_reward=0.0, profit_factor=0.0,
                 avg_trade=0.0, expectancy_r=0.0, best_trade=0.0, worst_trade=0.0, total_fees=0.0, total_funding=0.0,
                 avg_bars_held=0.0, trades_per_day=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                 long=side_stats(trades), short=side_stats(trades), by_symbol={}, by_exit_reason={}, by_source={})
    # equity-curve based
    if len(equity) > 1:
        dd = drawdown_series(equity, initial)
        m["max_drawdown_pct"] = float(dd.max())
        m["max_drawdown_inr"] = float(((pd.concat([pd.Series([initial]), equity.reset_index(drop=True)]).cummax()
                                        - pd.concat([pd.Series([initial]), equity.reset_index(drop=True)])).max()))
        d = daily_equity(equity, initial)
        base = pd.concat([pd.Series([initial], index=[d.index[0] - pd.Timedelta(days=1)]), d])
        dr = base.pct_change().dropna()
        dpnl = base.diff().dropna()
        ann = math.sqrt(365)
        sd = dr.std(ddof=1) if len(dr) > 1 else 0.0
        dn = dr[dr < 0]
        dsd = math.sqrt((dn ** 2).sum() / len(dr)) if len(dr) else 0.0
        m["sharpe"] = float(dr.mean() / sd * ann) if sd > 0 else 0.0
        m["sortino"] = float(dr.mean() / dsd * ann) if dsd > 0 else 0.0
        m["daily_pnl_mean"], m["daily_pnl_std"] = float(dpnl.mean()), float(dpnl.std(ddof=1)) if len(dpnl) > 1 else 0.0
        m["pct_days_ge_target"] = None
        m["best_day"], m["worst_day"] = float(dpnl.max()), float(dpnl.min())
        m["positive_days_pct"] = float((dpnl > 0).mean() * 100)
        mo = equity.resample("MS").last()
        prev = pd.concat([pd.Series([initial]), mo.iloc[:-1]]).to_numpy()
        m["monthly"] = [{"month": k.strftime("%Y-%m"), "pnl": float(v - p), "return_pct": float((v / p - 1) * 100)}
                        for k, v, p in zip(mo.index, mo.to_numpy(), prev)]
        m["positive_months_pct"] = float(np.mean([x["pnl"] > 0 for x in m["monthly"]]) * 100) if m["monthly"] else 0.0
    else:
        m.update(max_drawdown_pct=0.0, max_drawdown_inr=0.0, sharpe=0.0, sortino=0.0, daily_pnl_mean=0.0,
                 daily_pnl_std=0.0, best_day=0.0, worst_day=0.0, positive_days_pct=0.0, monthly=[],
                 positive_months_pct=0.0)
    return m


def daily_pnl_series(equity: pd.Series, initial: float) -> pd.Series:
    d = daily_equity(equity, initial)
    base = pd.concat([pd.Series([initial], index=[d.index[0] - pd.Timedelta(days=1)]), d])
    return base.diff().dropna()


def by_label(trades: pd.DataFrame, labels: pd.Series) -> dict:
    """Group trades by an (ex-post) regime label evaluated at the trade's entry bar. Reporting only."""
    if trades.empty:
        return {}
    ts = pd.to_datetime(trades["entry_ts_ms"], unit="ms", utc=True)
    lab = labels.reindex(ts, method="ffill").to_numpy()
    out = {}
    for k, g in trades.assign(_l=lab).groupby("_l"):
        out[str(k)] = side_stats(g)
    return out
