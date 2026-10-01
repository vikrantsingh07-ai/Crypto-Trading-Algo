"""Candidate strategies. Few parameters each, on purpose (anti-overfitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .. import regime as reg
from .base import Strategy, empty_signals


def _common(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["close"] = df["close"]
    f["atr"] = ind.atr(df, 14)
    f["adx"], f["pdi"], f["mdi"] = ind.adx(df, 14)
    f["rsi"] = ind.rsi(df["close"], 14)
    return f


class TrendBreakout(Strategy):
    """Donchian breakout in the direction of the EMA trend, confirmed by ADX and volume."""
    name = "trend_breakout"
    defaults = dict(donch_n=48, exit_n=24, ema_trend=200, adx_min=20.0, vol_min=1.0,
                    stop_atr=2.5, trail_atr=3.0, allow_short=True)
    grid = dict(donch_n=[30, 48, 72], adx_min=[15.0, 20.0, 25.0], stop_atr=[2.0, 3.0])

    def signals(self, df):
        p = self.params
        f = _common(df, p)
        hi, lo = ind.donchian_prior(df, int(p["donch_n"]))
        xhi, xlo = ind.donchian_prior(df, int(p["exit_n"]))
        et = ind.ema(df["close"], int(p["ema_trend"]))
        vr = ind.volume_ratio(df, 20)
        ok = f["adx"] > p["adx_min"]
        if p["vol_min"] > 0:
            ok &= vr > p["vol_min"]
        long_ = (df["close"] > hi) & (df["close"] > et) & ok
        short_ = (df["close"] < lo) & (df["close"] < et) & ok & bool(p["allow_short"])
        if self.use_htf:
            h = ind.htf_trend(df, self.base_tf, self.htf)
            long_ &= h == 1
            short_ &= h == -1
        out = empty_signals(df.index)
        out["signal"] = np.where(long_, 1, np.where(short_, -1, 0))
        on = out["signal"] != 0
        out.loc[on, "stop_dist"] = (p["stop_atr"] * f["atr"])[on]
        out.loc[on, "trail_dist"] = (p["trail_atr"] * f["atr"])[on]
        out["exit_long"] = (df["close"] < xlo).fillna(False)
        out["exit_short"] = (df["close"] > xhi).fillna(False)
        out.loc[on, "regime"] = "trend_breakout"
        return out


class TrendPullback(Strategy):
    """Buy the RSI recovery inside an established EMA/MACD uptrend (mirror for shorts)."""
    name = "trend_pullback"
    defaults = dict(ema_fast=20, ema_slow=50, ema_trend=200, rsi_long=45.0, rsi_short=55.0,
                    adx_min=18.0, stop_atr=2.0, trail_atr=3.0, allow_short=True)
    grid = dict(rsi_long=[40.0, 45.0, 50.0], adx_min=[15.0, 20.0], stop_atr=[2.0, 3.0])

    def signals(self, df):
        p = self.params
        f = _common(df, p)
        ef, es, et = (ind.ema(df["close"], int(p[k])) for k in ("ema_fast", "ema_slow", "ema_trend"))
        _, _, mh = ind.macd(df["close"])
        up = (ef > es) & (es > et) & (mh > 0) & (f["adx"] > p["adx_min"])
        dn = (ef < es) & (es < et) & (mh < 0) & (f["adx"] > p["adx_min"])
        r, r1 = f["rsi"], f["rsi"].shift(1)
        long_ = up & (r1 < p["rsi_long"]) & (r >= p["rsi_long"])
        short_ = dn & (r1 > p["rsi_short"]) & (r <= p["rsi_short"]) & bool(p["allow_short"])
        if self.use_htf:
            h = ind.htf_trend(df, self.base_tf, self.htf)
            long_ &= h == 1
            short_ &= h == -1
        out = empty_signals(df.index)
        out["signal"] = np.where(long_, 1, np.where(short_, -1, 0))
        on = out["signal"] != 0
        out.loc[on, "stop_dist"] = (p["stop_atr"] * f["atr"])[on]
        out.loc[on, "trail_dist"] = (p["trail_atr"] * f["atr"])[on]
        out["exit_long"] = ((ef < es)).fillna(False)
        out["exit_short"] = ((ef > es)).fillna(False)
        out.loc[on, "regime"] = "trend_pullback"
        return out


class MeanReversion(Strategy):
    """Fade Bollinger/RSI extremes only when ADX says the market is ranging; target the mean."""
    name = "mean_reversion"
    defaults = dict(bb_n=20, bb_k=2.0, rsi_lo=30.0, rsi_hi=70.0, adx_max=20.0, stop_atr=2.0,
                    max_hold=24, allow_short=True)
    grid = dict(bb_k=[1.8, 2.0, 2.5], rsi_lo=[25.0, 30.0], stop_atr=[1.5, 2.0, 3.0])

    def signals(self, df):
        p = self.params
        f = _common(df, p)
        mid, up, lo = ind.bollinger(df["close"], int(p["bb_n"]), p["bb_k"])
        quiet = f["adx"] < p["adx_max"]
        long_ = quiet & (df["close"] < lo) & (f["rsi"] < p["rsi_lo"])
        short_ = quiet & (df["close"] > up) & (f["rsi"] > p["rsi_hi"]) & bool(p["allow_short"])
        out = empty_signals(df.index)
        out["signal"] = np.where(long_, 1, np.where(short_, -1, 0))
        on = out["signal"] != 0
        out.loc[on, "stop_dist"] = (p["stop_atr"] * f["atr"])[on]
        out.loc[on, "tp_dist"] = (df["close"] - mid).abs()[on]
        out.loc[on, "max_hold"] = p["max_hold"]
        out.loc[on, "regime"] = "mean_reversion"
        return out


class RegimeEnsemble(Strategy):
    """Route by causal regime: trend strategies in trends, mean reversion in ranges, flat in chaos."""
    name = "regime_ensemble"
    defaults = dict(adx_trend=25.0, adx_range=20.0, high_vol=2.0, use_pullback=True,
                    donch_n=48, stop_atr=2.5, bb_k=2.0)
    grid = dict(donch_n=[30, 48, 72], stop_atr=[2.0, 3.0], bb_k=[2.0, 2.5])

    def _subs(self):
        p = self.params
        kw = dict(base_tf=self.base_tf, htf=self.htf, use_htf_confirmation=self.use_htf)
        tb = TrendBreakout({"donch_n": int(p["donch_n"]), "stop_atr": p["stop_atr"],
                            "adx_min": p["adx_trend"] - 5}, **kw)
        tp = TrendPullback({"stop_atr": min(p["stop_atr"], 2.5), "adx_min": p["adx_trend"] - 7}, **kw)
        mr = MeanReversion({"bb_k": p["bb_k"], "adx_max": p["adx_range"]}, **kw)
        return tb, tp, mr

    def signals(self, df):
        p = self.params
        r = reg.live_regime(df, p["adx_trend"], p["adx_range"], p["high_vol"])
        tb, tp, mr = self._subs()
        s_tb, s_tp, s_mr = tb.signals(df), tp.signals(df), mr.signals(df)
        trend = r.isin(["trend_up", "trend_down"])
        rng = r == "range"
        out = empty_signals(df.index)
        pick_tb = trend & (s_tb["signal"] != 0)
        pick_tp = trend & ~pick_tb & (s_tp["signal"] != 0) & bool(p["use_pullback"])
        pick_mr = rng & (s_mr["signal"] != 0)
        for mask, src in ((pick_tb, s_tb), (pick_tp, s_tp), (pick_mr, s_mr)):
            cols = ["signal", "stop_dist", "tp_dist", "trail_dist", "max_hold", "regime"]
            out.loc[mask, cols] = src.loc[mask, cols]
        out["signal"] = out["signal"].astype(int)
        # exits: trend-follower exits apply only to positions opened by that sub-strategy; since the
        # engine tags positions with their source regime we expose all exit flags and let the
        # engine filter by `entry_regime` (see engine._exit_signal_applies).
        out["exit_long"] = s_tb["exit_long"] | s_tp["exit_long"]
        out["exit_short"] = s_tb["exit_short"] | s_tp["exit_short"]
        out["live_regime"] = r
        return out

    @property
    def min_bars(self) -> int:
        return 600


REGISTRY = {c.name: c for c in (TrendBreakout, TrendPullback, MeanReversion, RegimeEnsemble)}


def get_strategy(name: str, params: dict | None = None, base_tf="1h", htf="4h", use_htf=False) -> Strategy:
    try:
        return REGISTRY[name](params, base_tf, htf, use_htf)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}") from None
