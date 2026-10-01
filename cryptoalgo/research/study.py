"""Which indicator conditions (and combinations) carry a statistically meaningful edge?

Method
  * Each condition is a boolean STATE per bar. Every bar where it holds is a candidate sample (decision at close t).
  * Entry is next bar's open; payoff is the signed return over the next H bars, minus round-trip costs.
  * Candidates closer than H bars to the previously accepted sample are dropped (non-overlapping payoffs),
    so a plain t-test on payoffs is approximately valid.
  * Every condition is tested in BOTH directions (long and short): e.g. "RSI<30" is tested as a
    mean-reversion long AND as a momentum short. Pairs of conditions (AND) are tested too.
  * ALL hypotheses are pooled and p-values corrected with Benjamini-Hochberg FDR.
Only DEVELOPMENT data (never the hold-out) is fed in by the pipeline.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .. import indicators as ind


def conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["close"]
    atr = ind.atr(df, 14)
    adx, pdi, mdi = ind.adx(df, 14)
    rsi = ind.rsi(c, 14)
    ef, es, et = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    _, _, mh = ind.macd(c)
    mid, up, lo = ind.bollinger(c, 20, 2.0)
    hi48, lo48 = ind.donchian_prior(df, 48)
    vr = ind.volume_ratio(df, 20)
    vs = ind.vol_state(atr / c)
    return {
        "close>EMA200": c > et, "close<EMA200": c < et,
        "EMA20>EMA50": ef > es, "EMA20<EMA50": ef < es,
        "MACDhist>0": mh > 0, "MACDhist<0": mh < 0,
        "RSI<30": rsi < 30, "RSI>70": rsi > 70,
        "RSI<45": rsi < 45, "RSI>55": rsi > 55,
        "close<BBlow": c < lo, "close>BBup": c > up,
        "Donchian48 up": c > hi48, "Donchian48 down": c < lo48,
        "ADX>25": adx > 25, "ADX<20": adx < 20,
        "VolSpike>1.5": vr > 1.5,
        "HighVol>1.5": vs > 1.5, "LowVol<0.8": vs < 0.8,
    }


def _events(state: pd.Series) -> np.ndarray:
    """Candidate decision bars: every bar where the condition holds (thinned later to non-overlapping)."""
    return state.fillna(False).to_numpy(dtype=bool)


def _payoffs(ev: np.ndarray, direction: int, opn: np.ndarray, cls: np.ndarray, H: int, cost: float, warm: int):
    """Non-overlapping signed returns (after cost) for events; entry at next open, exit at close H bars later."""
    out, last = [], -10**9
    n = len(ev)
    for t in np.flatnonzero(ev):
        if t < warm or t + 1 + H >= n or t - last < H:
            continue
        entry, exit_ = opn[t + 1], cls[t + H]
        out.append(direction * (exit_ / entry - 1) - cost)
        last = t
    return np.asarray(out)


def _p_two_sided(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def run_study(data: dict[str, pd.DataFrame], horizon: int = 12, round_trip_cost_bps: float = 34.0,
              min_events: int = 30, fdr: float = 0.10, warm: int = 300, pairs: bool = True) -> pd.DataFrame:
    """Return a table of all hypotheses with mean net payoff (bps), t-stat, p, FDR q-value."""
    cost = round_trip_cost_bps / 1e4
    per_sym = {}
    for s, df in data.items():
        per_sym[s] = (conditions(df), df["open"].to_numpy(), df["close"].to_numpy())
    names = list(next(iter(per_sym.values()))[0])
    combos = [(n,) for n in names]
    if pairs:
        combos += list(itertools.combinations(names, 2))
    rows = []
    for combo in combos:
        for direction in (+1, -1):
            pay = []
            for s, (conds, opn, cls) in per_sym.items():
                state = conds[combo[0]]
                for extra in combo[1:]:
                    state = state & conds[extra]
                pay.append(_payoffs(_events(state), direction, opn, cls, horizon, cost, warm))
            x = np.concatenate(pay) if pay else np.array([])
            if len(x) < min_events or x.std(ddof=1) == 0:
                continue
            t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
            rows.append({"condition": " AND ".join(combo), "direction": "long" if direction > 0 else "short",
                         "n_events": len(x), "mean_net_bps": x.mean() * 1e4, "hit_rate": float((x > 0).mean()),
                         "t_stat": t, "p_value": _p_two_sided(t), "n_components": len(combo)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr_q"] = bh_fdr(df["p_value"].to_numpy())
    df["significant"] = (df["fdr_q"] <= fdr) & (df["mean_net_bps"] > 0)
    return df.sort_values("p_value").reset_index(drop=True)
