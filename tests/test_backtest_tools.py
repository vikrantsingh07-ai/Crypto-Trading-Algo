"""Metrics, walk-forward integrity, Monte Carlo, sensitivity, feasibility, gates, research."""
import math

import numpy as np
import pandas as pd
import pytest

from cryptoalgo.backtest.feasibility import assess_target
from cryptoalgo.backtest.gates import evaluate_gates
from cryptoalgo.backtest.metrics import by_label, compute_metrics, drawdown_series
from cryptoalgo.backtest.montecarlo import monte_carlo
from cryptoalgo.backtest.runner import precompute, run_backtest
from cryptoalgo.backtest.sensitivity import sensitivity
from cryptoalgo.backtest.walkforward import walk_forward
from cryptoalgo.config import Gates, load_config
from cryptoalgo.data.synthetic import generate
from cryptoalgo.research.study import bh_fdr, run_study
from cryptoalgo.strategies.library import get_strategy

INIT = 100_000.0


def trades_df(pnls, side="long", start="2024-01-01"):
    n = len(pnls)
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"trade_id": [f"t{i}" for i in range(n)], "symbol": "BTCUSDT",
                         "side": [side if not isinstance(side, list) else side[i] for i in range(n)],
                         "net_pnl": pnls, "fees": 10.0, "funding": 1.0, "r_multiple": np.array(pnls) / 500.0,
                         "bars_held": 5, "entry_ts_ms": [int(t.timestamp() * 1000) for t in ts], "exit_reason": "stop", "source": "x"})


def equity_from(pnls, start="2024-01-01", days_per_trade=1):
    idx = pd.date_range(start, periods=len(pnls) * days_per_trade, freq="1D", tz="UTC")
    eq = INIT + np.cumsum(np.repeat(np.array(pnls) / days_per_trade, days_per_trade))
    return pd.Series(eq, index=idx)


# ------------------------------------------------------------------ metrics
def test_metrics_hand_computed():
    pnls = [100, -50, 200, -50, -50, 300]
    m = compute_metrics(trades_df(pnls, side=["long", "short", "long", "long", "short", "long"]), equity_from(pnls), INIT)
    assert m["trades"] == 6 and m["final_balance"] == pytest.approx(INIT + 450) and m["net_pnl"] == 450
    assert m["total_return_pct"] == pytest.approx(0.45)
    assert m["win_rate_pct"] == pytest.approx(50) and m["loss_rate_pct"] == pytest.approx(50)
    assert m["profit_factor"] == pytest.approx(600 / 150)
    assert m["avg_win"] == pytest.approx(200) and m["avg_loss"] == pytest.approx(-50) and m["risk_reward"] == pytest.approx(4.0)
    assert m["avg_trade"] == pytest.approx(75)
    assert m["max_consecutive_wins"] == 1 and m["max_consecutive_losses"] == 2
    assert m["long"]["trades"] == 4 and m["long"]["net_pnl"] == pytest.approx(550)
    assert m["short"]["trades"] == 2 and m["short"]["net_pnl"] == pytest.approx(-100)
    assert m["daily_avg_pnl"] == pytest.approx(450 / 6)
    assert m["max_drawdown_pct"] == pytest.approx(100 / 100_250 * 100, rel=1e-6)     # peak 100250 -> 100150
    assert m["monthly"][0]["month"] == "2024-01" and m["monthly"][0]["pnl"] == pytest.approx(450)


def test_sharpe_sortino_match_manual_formula():
    rng = np.random.default_rng(4)
    pnl = rng.normal(80, 400, 200)
    eq = equity_from(pnl)
    m = compute_metrics(trades_df(list(pnl)), eq, INIT)
    r = pd.concat([pd.Series([INIT]), eq.reset_index(drop=True)]).pct_change().dropna().to_numpy()
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(365), rel=1e-9)
    dd = math.sqrt((np.minimum(r, 0) ** 2).mean())
    assert m["sortino"] == pytest.approx(r.mean() / dd * math.sqrt(365), rel=1e-9)


def test_drawdown_series_and_edge_cases():
    eq = pd.Series([101, 102, 90, 95, 110], dtype=float) * 1000
    dd = drawdown_series(eq, 100_000)
    assert dd.max() == pytest.approx((102 - 90) / 102 * 100)
    m0 = compute_metrics(pd.DataFrame(columns=["net_pnl"]), pd.Series(dtype=float), INIT)
    assert m0["trades"] == 0 and m0["final_balance"] == INIT and m0["profit_factor"] == 0
    m1 = compute_metrics(trades_df([10, 20]), equity_from([10, 20]), INIT)       # no losing trade -> capped PF, not inf/NaN
    assert math.isfinite(m1["profit_factor"]) and m1["max_consecutive_losses"] == 0


def test_by_label_groups_by_entry_bar_label():
    t = trades_df([10, -5, 7])
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    lab = pd.Series(["bull", "bull", "bear", "bear", "bear"], index=idx)
    g = by_label(t, lab)
    assert g["bull"]["trades"] == 2 and g["bear"]["trades"] == 1 and g["bull"]["net_pnl"] == 5


def test_backtest_result_consistent_with_metrics(synth, base_cfg):
    data, _ = synth
    st = get_strategy("trend_breakout")
    r = run_backtest(data, base_cfg, st)
    m = compute_metrics(r.trades, r.equity, INIT)
    assert not r.trades.empty and r.trades["trade_id"].is_unique
    assert m["final_balance"] == pytest.approx(INIT + r.trades["net_pnl"].sum(), rel=1e-9)    # liquidated at end: exact
    assert (r.trades["exit_ts_ms"] >= r.trades["entry_ts_ms"]).all()
    assert (r.trades["entry_ts_ms"] > 0).all() and m["total_fees"] > 0


def test_backtest_costs_reduce_profit(synth, base_cfg):
    data, _ = synth
    st = get_strategy("trend_breakout")
    rows = precompute(data, st)
    a = run_backtest(data, base_cfg, st, rows=rows)
    cheap = base_cfg.copy(); cheap.costs = base_cfg.costs.scaled(0.0)
    b = run_backtest(data, cheap, st, rows=rows)
    dear = base_cfg.copy(); dear.costs = base_cfg.costs.scaled(2.0)
    c = run_backtest(data, dear, st, rows=rows)
    assert float(b.equity.iloc[-1]) > float(a.equity.iloc[-1]) > float(c.equity.iloc[-1])


def test_backtest_window_slicing_is_independent_of_other_data(synth, base_cfg):
    """Trading a window must only depend on data up to the window end (no peeking at later bars)."""
    data, _ = synth
    st = get_strategy("trend_breakout")
    idx = data["BTCUSDT"].index
    cut = idx[6000]
    full = run_backtest(data, base_cfg, st, start=idx[1000], end=cut)
    trunc = run_backtest({s: d[d.index < cut] for s, d in data.items()}, base_cfg, st, start=idx[1000], end=cut)
    assert full.trades["trade_id"].tolist() == trunc.trades["trade_id"].tolist()
    assert full.trades["net_pnl"].tolist() == pytest.approx(trunc.trades["net_pnl"].tolist())


# ------------------------------------------------------------------ walk-forward
def test_walk_forward_structure_and_no_peeking(synth, base_cfg):
    data, _ = synth
    grid = [{"donch_n": 30}, {"donch_n": 48}, {"donch_n": 72}]
    wf = walk_forward(data, base_cfg, "trend_breakout", grid, train_days=150, test_days=50)
    assert len(wf.windows) >= 4
    for w in wf.windows:
        assert w.test[0] == w.train[1] and w.test[1] > w.test[0]
    for a, b in zip(wf.windows, wf.windows[1:]):
        assert b.test[0] >= a.test[1]                               # OOS windows never overlap
    # Perturb ALL prices after window-0's training end: window-0's chosen params must not change.
    w0 = wf.windows[0]
    mod = {}
    rng = np.random.default_rng(0)
    for s, d in data.items():
        d2 = d.copy()
        m = d2.index >= w0.test[0]
        d2.loc[m, ["open", "high", "low", "close"]] = d2.loc[m, ["open", "high", "low", "close"]].to_numpy()[::-1]
        mod[s] = d2
    wf2 = walk_forward(mod, base_cfg, "trend_breakout", grid, train_days=150, test_days=50)
    assert wf2.windows[0].params == w0.params


def test_walk_forward_flat_when_no_in_sample_edge(base_cfg):
    d, _ = generate(days=420, world="null", seed=77)
    wf = walk_forward(d, base_cfg, "trend_breakout", [{"donch_n": 48}], train_days=150, test_days=60)
    flats = [w for w in wf.windows if w.params is None]
    for w in flats:
        assert w.test_trades.empty                                  # no profitable training param -> do not trade


# ------------------------------------------------------------------ Monte Carlo
def test_monte_carlo_properties():
    win = monte_carlo([100.0] * 50, INIT)
    assert win["prob_loss"] == 0 and win["final_equity"]["p5"] == pytest.approx(INIT + 5000)
    rng = np.random.default_rng(0)
    pn = rng.normal(0, 300, 200)
    sh = monte_carlo(pn, INIT, mode="shuffle")
    assert sh["final_equity"]["p5"] == pytest.approx(sh["final_equity"]["p95"])       # shuffling can't change the total
    bs = monte_carlo(pn, INIT)
    assert bs["final_equity"]["p5"] < bs["final_equity"]["p50"] < bs["final_equity"]["p95"]
    assert bs["max_dd_pct"]["p95"] >= bs["max_dd_pct"]["p50"]
    assert monte_carlo(pn, INIT, seed=3) == monte_carlo(pn, INIT, seed=3)
    assert monte_carlo([1, 2], INIT)["insufficient"]
    loser = monte_carlo([-200.0] * 30 + [50.0] * 10, INIT)
    assert loser["prob_loss"] == 1.0


# ------------------------------------------------------------------ sensitivity
def test_sensitivity_output(synth, base_cfg):
    data, _ = synth
    s = sensitivity(data, base_cfg, "trend_breakout", {}, data["BTCUSDT"].index[0], data["BTCUSDT"].index[7000])
    assert s["n_neighbours"] >= 6 and 0 <= s["fraction_profitable"] <= 1
    assert s["rows"][0]["variant"] == "base"


# ------------------------------------------------------------------ feasibility
def test_target_not_achievable_for_losing_or_flat_strategy(base_cfg):
    eq = equity_from(list(np.random.default_rng(1).normal(-30, 400, 365)))
    f = assess_target(eq, INIT, base_cfg, 8.0)
    assert not f["achievable"] and "NOT ACHIEVABLE" in f["verdict"]


def test_target_scaling_math_and_risk_limit(base_cfg):
    rng = np.random.default_rng(2)
    eq = equity_from(list(rng.normal(250, 300, 365)))
    mdd = float(drawdown_series(eq, INIT).max())
    f = assess_target(eq, INIT, base_cfg, mdd)
    assert not f["achievable"]
    assert f["scale_needed"] == pytest.approx(2000 / f["mean_daily_pnl"])
    assert f["implied_max_drawdown_pct"] == pytest.approx(mdd * f["scale_needed"])
    assert f["required_annual_return_simple_pct"] == pytest.approx(730.0)
    assert f["target_pct_of_capital"] == pytest.approx(2.0)
    assert "NOT ACHIEVABLE" in f["verdict"]


def test_target_can_be_reported_met_when_data_really_supports_it(base_cfg):
    rng = np.random.default_rng(3)
    eq = equity_from(list(rng.normal(2600, 700, 400)))
    f = assess_target(eq, INIT, base_cfg, 3.0)
    assert f["achievable"] and f["prob_mean_ge_target"] >= 0.8


def test_target_insufficient_data(base_cfg):
    f = assess_target(equity_from([100] * 10), INIT, base_cfg, 1.0)
    assert f["verdict"] == "INSUFFICIENT DATA" and not f["achievable"]


# ------------------------------------------------------------------ gates
def good_inputs():
    ho = {"profit_factor": 1.5, "net_pnl": 5000, "trades": 150, "max_drawdown_pct": 8, "sharpe": 1.2, "initial_capital": INIT}
    wfs = {"positive_window_fraction": 0.8}
    wfo = {"trades": 200, "profit_factor": 1.4, "net_pnl": 4000}
    mc = {"max_dd_pct": {"p95": 12}, "final_equity": {"p5": INIT + 100}}
    sens = {"fraction_profitable": 0.9}
    stress = {"net_pnl": 1000}
    return ho, wfs, wfo, mc, sens, stress, {"BTCUSDT": 1, "ETHUSDT": 1}


def test_gates_all_pass_and_each_gate_can_fail():
    ok, rows = evaluate_gates(Gates(), *good_inputs())
    assert ok and len(rows) == 12 and all(r["pass"] for r in rows)
    mutations = [
        lambda a: a[0].update(profit_factor=1.0), lambda a: a[0].update(trades=10), lambda a: a[2].update(trades=10),
        lambda a: a[2].update(profit_factor=1.0), lambda a: a[1].update(positive_window_fraction=0.2),
        lambda a: a[0].update(max_drawdown_pct=30), lambda a: a[0].update(sharpe=0.1),
        lambda a: a[3]["max_dd_pct"].update(p95=60), lambda a: a[3]["final_equity"].update(p5=INIT - 1),
        lambda a: a[4].update(fraction_profitable=0.2), lambda a: a[5].update(net_pnl=-1),
        lambda a: a[6].update(ETHUSDT=-5)]
    for i, mut in enumerate(mutations):
        args = good_inputs()
        mut(args)
        ok, rows = evaluate_gates(Gates(), *args)
        assert not ok and sum(not r["pass"] for r in rows) == 1, i


# ------------------------------------------------------------------ research / multiple testing
def test_bh_fdr_properties():
    p = np.array([0.001, 0.04, 0.03, 0.8, 0.02])
    q = bh_fdr(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()
    assert q[np.argsort(p)].tolist() == sorted(q.tolist())          # monotone in p
    assert bh_fdr(np.array([0.5]))[0] == 0.5


def test_research_study_controls_false_discoveries_on_pure_noise():
    d, _ = generate(days=700, world="null", seed=123)
    t = run_study(d, horizon=12, pairs=False)
    assert len(t) >= 20
    assert int(t["significant"].sum()) <= max(2, int(0.1 * len(t)))        # no edge exists: FDR control must keep this tiny
    assert (t["fdr_q"] >= t["p_value"] - 1e-12).all()


def _planted_series(n, plant, seed=5, horizon=6, boost=0.0015):
    """Random-walk prices; if `plant`, bars following an RSI(14)<30 reading get +boost/bar for `horizon` bars
    (generated sequentially, so the effect is causal: it depends only on information available at the time)."""
    rng = np.random.default_rng(seed)
    close = [100.0]
    up = dn = None
    left = 0
    rsi_prev = 50.0
    for i in range(1, n):
        r = rng.normal(0, 0.006) + (boost if left > 0 else 0.0)
        left = max(left - 1, 0)
        c = close[-1] * (1 + r)
        d = c - close[-1]
        g, l = max(d, 0.0), max(-d, 0.0)
        up = g if up is None else (up * 13 + g) / 14
        dn = l if dn is None else (dn * 13 + l) / 14
        rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
        close.append(c)
        if plant and i > 30 and rsi < 30 and left == 0:
            left = horizon
        rsi_prev = rsi
    c = pd.Series(close, index=pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC"))
    o = c.shift(1).fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": 100.0}, index=c.index)


def test_research_study_detects_a_planted_effect_and_not_without_it():
    planted = {"BTCUSDT": _planted_series(40000, True, 1), "ETHUSDT": _planted_series(40000, True, 2)}
    t = run_study(planted, horizon=6, round_trip_cost_bps=34, pairs=False)
    hit = t[(t["condition"] == "RSI<30") & (t["direction"] == "long")].iloc[0]
    assert hit["significant"] and hit["mean_net_bps"] > 20
    plain = {"BTCUSDT": _planted_series(40000, False, 1), "ETHUSDT": _planted_series(40000, False, 2)}
    t0 = run_study(plain, horizon=6, round_trip_cost_bps=34, pairs=False)
    assert int(t0["significant"].sum()) == 0
