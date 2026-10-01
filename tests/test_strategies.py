"""Strategy tests on constructed and synthetic series."""
import numpy as np
import pandas as pd
import pytest

from cryptoalgo.strategies.base import SIGNAL_COLS
from cryptoalgo.strategies.library import REGISTRY, TrendBreakout, get_strategy


def ohlc(close, vol=100.0, spread=0.002):
    idx = pd.date_range("2023-01-01", periods=len(close), freq="1h", tz="UTC")
    c = pd.Series(close, index=idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
                         "close": c, "volume": vol}, index=idx)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_signal_frame_contract(synth, name):
    df = synth[0]["BTCUSDT"]          # full 400 days: trend_pullback is a rare-trigger strategy
    s = get_strategy(name).signals(df)
    assert set(SIGNAL_COLS) <= set(s.columns) and len(s) == len(df) and s.index.equals(df.index)
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    on = s["signal"] != 0
    assert on.sum() > 0
    assert (s.loc[on, "stop_dist"] > 0).all()                      # every entry carries a protective stop
    assert s.loc[~on, "stop_dist"].isna().all()
    assert s["exit_long"].dtype == bool


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_signals_during_indicator_warmup(synth, name):
    df = synth[0]["BTCUSDT"].iloc[:2500]
    s = get_strategy(name).signals(df)
    assert (s["signal"].iloc[:50] == 0).all()


def test_breakout_long_signal_on_clean_uptrend_breakout():
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0.05, 0.4, 700))
    base[-1] = base[-60:].max() + 8                                  # decisive breakout on the last bar
    df = ohlc(base)
    df.loc[df.index[-1], "volume"] = 500.0
    st = TrendBreakout({"adx_min": 0.0, "vol_min": 1.0})
    s = st.signals(df)
    assert s["signal"].iloc[-1] == 1 and s["stop_dist"].iloc[-1] > 0 and s["trail_dist"].iloc[-1] > s["stop_dist"].iloc[-1] * 0.5


def test_breakout_never_goes_long_below_ema200_and_short_only_when_allowed():
    rng = np.random.default_rng(2)
    down = 200 - np.cumsum(np.abs(rng.normal(0.1, 0.3, 700)))
    df = ohlc(down)
    s = TrendBreakout({"adx_min": 0.0, "vol_min": 0}).signals(df)
    assert (s["signal"] == 1).sum() == 0                              # a pure downtrend: no longs
    s2 = TrendBreakout({"adx_min": 0.0, "vol_min": 0, "allow_short": False}).signals(df)
    assert (s2["signal"] == -1).sum() == 0
    assert (s["signal"] == -1).sum() > 0


def test_mean_reversion_only_fires_in_low_adx_and_targets_mean(synth):
    df = synth[0]["BTCUSDT"].iloc[:6000]
    from cryptoalgo import indicators as ind
    s = get_strategy("mean_reversion").signals(df)
    adx = ind.adx(df)[0]
    on = s["signal"] != 0
    assert (adx[on] < 20).all()
    assert (s.loc[on, "tp_dist"] > 0).all() and (s.loc[on, "max_hold"] > 0).all()


def test_ensemble_routes_by_regime_and_stands_aside_in_high_vol(synth):
    df = synth[0]["BTCUSDT"].iloc[:8000]
    s = get_strategy("regime_ensemble").signals(df)
    on = s["signal"] != 0
    assert on.sum() > 0
    assert (s.loc[on, "live_regime"].isin(["trend_up", "trend_down", "range"])).all()
    assert set(s.loc[on, "regime"].unique()) <= {"trend_breakout", "trend_pullback", "mean_reversion"}
    assert (s.loc[s["live_regime"] == "high_vol", "signal"] == 0).all()


def test_unknown_params_and_strategy_rejected():
    with pytest.raises(ValueError):
        get_strategy("trend_breakout", {"not_a_param": 1})
    with pytest.raises(ValueError):
        get_strategy("moon_strategy")


def test_param_grids_are_small_and_neighbours_perturb():
    for name, cls in REGISTRY.items():
        assert 1 <= len(cls.param_grid()) <= 30, name              # pre-declared, small grid (anti-overfitting)
    st = get_strategy("trend_breakout")
    nb = st.neighbours()
    assert nb and all(len(d) == 1 for d in nb)
    assert st.with_params(donch_n=30).params["donch_n"] == 30 and st.params["donch_n"] == 48


def test_htf_confirmation_only_removes_signals(synth):
    df = synth[0]["BTCUSDT"].iloc[:5000]
    a = get_strategy("trend_breakout", {"adx_min": 15.0}, use_htf=False).signals(df)["signal"]
    b = get_strategy("trend_breakout", {"adx_min": 15.0}, use_htf=True).signals(df)["signal"]
    assert ((b != 0) <= (a != 0)).all() and (b != 0).sum() < (a != 0).sum()
