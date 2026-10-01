"""Backtest/paper core: fills, stops, costs, accounting. Hand-checkable numbers (zero-cost config, fx=1)."""
import pytest

from cryptoalgo.config import Costs
from cryptoalgo.engine import TradingEngine
from cryptoalgo.events import MemorySink
from conftest import H, T0, bar, sig

S = "BTCUSDT"


def mk(cfg):
    sink = MemorySink()
    return TradingEngine(cfg, "t", sink), sink


def step(eng, i, b, s=None):
    return eng.process_bar(T0 + i * H, {S: b}, {S: s} if s else {})


def test_signal_fills_at_next_open_not_same_bar(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100, c=100), sig())              # decision at close of bar 0
    assert not eng.pf.positions                          # nothing executed on the signal bar
    step(eng, 1, bar(105, h=106, l=104, c=105))          # fills at bar 1 OPEN
    p = eng.pf.positions[S]
    assert p.entry_price == 105 and p.entry_ts_ms == T0 + H and p.signal_ts_ms == T0


def test_stop_loss_exact_risk(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10))
    step(eng, 1, bar(100, h=101, l=99, c=100))
    assert eng.pf.positions[S].stop == 90
    step(eng, 2, bar(99, h=100, l=85, c=86))
    t = sink.trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == 90
    assert t["net_pnl"] == pytest.approx(-1000.0)       # 1% of 100,000 risked
    assert t["r_multiple"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10))
    step(eng, 1, bar(100))
    step(eng, 2, bar(80, h=82, l=78, c=79))              # gaps below the 90 stop
    assert sink.trades[0]["exit_price"] == 80
    assert sink.trades[0]["net_pnl"] == pytest.approx(-2000.0)


def test_stop_and_target_same_bar_stop_wins(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10, tp=20))
    step(eng, 1, bar(100))
    step(eng, 2, bar(100, h=130, l=80, c=100))           # both levels inside the bar
    assert sink.trades[0]["exit_reason"] == "stop"


def test_take_profit(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10, tp=20))
    step(eng, 1, bar(100))
    step(eng, 2, bar(110, h=125, l=109, c=120))
    t = sink.trades[0]
    assert t["exit_reason"] == "tp" and t["exit_price"] == 120 and t["net_pnl"] == pytest.approx(2000.0)


def test_trailing_stop_ratchets_up_only(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10, trail=5))
    step(eng, 1, bar(100, h=100, l=99, c=100))
    step(eng, 2, bar(100, h=110, l=100, c=108))          # best=110 -> stop 105
    assert eng.pf.positions[S].stop == 105
    step(eng, 3, bar(108, h=108, l=106, c=107))          # no new high: stop must not fall
    assert eng.pf.positions[S].stop == 105
    step(eng, 4, bar(106, h=106, l=104, c=104))
    t = sink.trades[0]
    assert t["exit_reason"] == "trail_stop" and t["exit_price"] == 105


def test_short_side_pnl_and_stop(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(side=-1, stop=10))
    step(eng, 1, bar(100))
    assert eng.pf.positions[S].stop == 110 and eng.pf.positions[S].side == -1
    step(eng, 2, bar(105, h=112, l=104, c=111))
    t = sink.trades[0]
    assert t["side"] == "short" and t["exit_price"] == 110 and t["net_pnl"] == pytest.approx(-1000.0)


def test_fees_and_slippage_applied_both_sides(exact_cfg):
    exact_cfg.costs = Costs(taker_fee_bps=10, half_spread_bps=2, slippage_bps=5, stop_extra_slippage_bps=0, funding_bps_per_8h=0)
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10))
    step(eng, 1, bar(100))
    p = eng.pf.positions[S]
    assert p.entry_price == pytest.approx(100 * (1 + 7e-4))   # buy pays spread+slippage
    assert p.entry_fee_inr == pytest.approx(p.entry_price * p.qty * 1e-3)
    eng.liquidate()
    t = sink.trades[0]
    assert t["exit_price"] == pytest.approx(100 * (1 - 7e-4))
    assert t["fees"] == pytest.approx((t["entry_price"] + t["exit_price"]) * t["qty"] * 1e-3)
    assert t["net_pnl"] < 0                                    # round-trip costs alone lose money


def test_funding_sign_long_pays_short_receives(exact_cfg):
    exact_cfg.costs = Costs(0, 0, 0, 0, funding_bps_per_8h=10)
    for side, sign in ((1, -1), (-1, +1)):
        eng, sink = mk(exact_cfg)
        eng.bar_hours = 8
        step(eng, 0, bar(100), sig(side=side, stop=50))
        step(eng, 1, bar(100))
        step(eng, 2, bar(100))
        eng.liquidate()
        assert sign * sink.trades[0]["net_pnl"] > 0


def test_time_stop(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=50, hold=3))
    for i in range(1, 6):
        step(eng, i, bar(100))
    assert sink.trades and sink.trades[0]["exit_reason"] == "time_stop"


def test_accounting_identity(exact_cfg):
    eng, sink = mk(exact_cfg)
    exact_cfg.costs = Costs(10, 2, 5, 5, 1)
    eng, sink = mk(exact_cfg)
    import random
    rnd = random.Random(3)
    px = 100.0
    for i in range(300):
        o = px
        c = o * (1 + rnd.gauss(0, 0.01))
        h, l = max(o, c) * 1.003, min(o, c) * 0.997
        s = sig(side=rnd.choice([1, -1]), stop=px * 0.02, tp=px * 0.03 if i % 2 else None, trail=px * 0.02) if i % 7 == 0 else None
        step(eng, i, (o, h, l, c, 1.0), s)
        px = c
    eng.liquidate()
    assert not eng.pf.positions
    assert eng.pf.cash == pytest.approx(exact_cfg.account.initial_capital + sum(t["net_pnl"] for t in sink.trades), rel=1e-9)
    assert len(sink.trades) > 5


def test_replayed_bar_is_ignored(exact_cfg):
    eng, sink = mk(exact_cfg)
    assert step(eng, 0, bar(100), sig()) is True
    n = len(sink.orders)
    assert step(eng, 0, bar(100), sig()) is False
    assert eng.bar_idx == 1 and len(sink.orders) == n


def test_exit_signal_closes_trend_position_at_next_open(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=50, regime="trend_breakout"))
    step(eng, 1, bar(100))
    step(eng, 2, bar(101), sig(side=0, exit_long=True, stop=float("nan")))
    assert not sink.trades
    step(eng, 3, bar(103))
    assert sink.trades[0]["exit_reason"] == "signal_exit" and sink.trades[0]["exit_price"] == 103


def test_mean_reversion_position_ignores_trend_exit_flags(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=50, tp=30, regime="mean_reversion"))
    step(eng, 1, bar(100))
    step(eng, 2, bar(100), sig(side=0, exit_long=True, stop=float("nan")))
    step(eng, 3, bar(100))
    assert not sink.trades


def test_state_roundtrip(exact_cfg):
    eng, sink = mk(exact_cfg)
    step(eng, 0, bar(100), sig(stop=10, trail=5))
    step(eng, 1, bar(100))
    d = eng.state_dict()
    import json
    d = json.loads(json.dumps(d))
    eng2, _ = mk(exact_cfg)
    eng2.load_state(d)
    assert eng2.pf.positions[S].stop == eng.pf.positions[S].stop
    assert eng2.last_ts_ms == eng.last_ts_ms and eng2.pf.cash == eng.pf.cash
