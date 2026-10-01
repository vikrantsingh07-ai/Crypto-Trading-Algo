"""Look-ahead-bias tests: values at bar t must not change if the future is removed."""
import numpy as np
import pandas as pd
import pytest

from cryptoalgo import indicators as ind
from cryptoalgo import regime
from cryptoalgo.research.study import conditions
from cryptoalgo.strategies.library import REGISTRY, get_strategy


@pytest.fixture(scope="module")
def df(synth):
    return synth[0]["BTCUSDT"].iloc[:3000]


def _same(a, b):
    if isinstance(a, pd.DataFrame):
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
        pd.testing.assert_frame_equal(a.astype(object).where(a.notna(), None), b.astype(object).where(b.notna(), None), check_dtype=False)
    else:
        np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True, rtol=1e-9, atol=1e-9)


CUTS = (1200, 1777, 2400)


@pytest.mark.parametrize("name,fn", [
    ("ema", lambda d: ind.ema(d["close"], 50)),
    ("rsi", lambda d: ind.rsi(d["close"], 14)),
    ("atr", lambda d: ind.atr(d, 14)),
    ("adx", lambda d: ind.adx(d, 14)[0]),
    ("macd", lambda d: ind.macd(d["close"])[2]),
    ("boll", lambda d: ind.bollinger(d["close"])[1]),
    ("donch", lambda d: ind.donchian_prior(d, 48)[0]),
    ("volratio", lambda d: ind.volume_ratio(d)),
    ("volstate", lambda d: ind.vol_state(ind.atr(d) / d["close"])),
    ("htf", lambda d: ind.htf_trend(d, "1h", "4h").astype(float)),
])
def test_indicator_has_no_lookahead(df, name, fn):
    full = fn(df)
    for cut in CUTS:
        part = fn(df.iloc[:cut])
        np.testing.assert_allclose(part.to_numpy(dtype=float), full.iloc[:cut].to_numpy(dtype=float), equal_nan=True,
                                   rtol=1e-9, atol=1e-9, err_msg=f"{name} changed when future bars were removed (cut={cut})")


@pytest.mark.parametrize("name", sorted(REGISTRY))
@pytest.mark.parametrize("htf", [False, True])
def test_strategy_signals_have_no_lookahead(df, name, htf):
    st = get_strategy(name, use_htf=htf)
    full = st.signals(df)
    for cut in CUTS:
        part = st.signals(df.iloc[:cut])
        for col in ("signal", "exit_long", "exit_short"):
            assert (part[col].to_numpy() == full[col].iloc[:cut].to_numpy()).all(), f"{name}/{col} cut={cut}"
        for col in ("stop_dist", "tp_dist", "trail_dist"):
            np.testing.assert_allclose(part[col].to_numpy(float), full[col].iloc[:cut].to_numpy(float), equal_nan=True, rtol=1e-9)


def test_live_regime_causal(df):
    full = regime.live_regime(df)
    part = regime.live_regime(df.iloc[:1777])
    assert (full.iloc[:1777].to_numpy() == part.to_numpy()).all()


def test_research_conditions_causal(df):
    full = conditions(df)
    part = conditions(df.iloc[:1777])
    for k in full:
        assert (full[k].iloc[:1777].fillna(False).to_numpy() == part[k].fillna(False).to_numpy()).all(), k


def test_htf_uses_only_closed_higher_timeframe_bars():
    """A 4h bar [00:00,04:00) is known only at 04:00. The 1h bar opening 03:00 (closing 04:00) may use it;
    the 1h bar opening 02:00 (closing 03:00) must still see the PREVIOUS 4h bar."""
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
    d = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    t = ind.htf_trend(d, "1h", "4h")
    mod = d.copy()
    # perturb ONLY the 03:00 bar of some day; trend seen at the 02:00 bar (closing 03:00) must be unaffected
    k = idx.get_loc(pd.Timestamp("2024-01-30 03:00", tz="UTC"))
    mod.iloc[k, mod.columns.get_loc("close")] += 50
    mod.iloc[k, mod.columns.get_loc("high")] += 50
    t2 = ind.htf_trend(mod, "1h", "4h")
    assert (t.iloc[:k].to_numpy() == t2.iloc[:k].to_numpy()).all()
