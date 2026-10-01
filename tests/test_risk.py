"""Risk-management tests: sizing, caps, halts, resets, cooldowns."""
import pytest

from cryptoalgo.config import Config, Costs, Risk, load_config
from cryptoalgo.engine import TradingEngine
from cryptoalgo.events import MemorySink
from cryptoalgo.risk import RiskManager
from conftest import H, T0, bar, sig

S = "BTCUSDT"
DAY = 24 * H


def rm(**kw):
    r = Risk(**kw)
    events = []
    m = RiskManager(r, Costs(0, 0, 0, 0, 0), 100_000, 1.0, emit=lambda ts, k, d: events.append((ts, k, d)))
    return m, events


def test_position_size_risks_configured_fraction():
    m, _ = rm(risk_per_trade_pct=0.5)
    sz = m.size(100_000, 50_000, 1_000, 0, 0)
    assert sz.risk_inr == pytest.approx(500) and sz.qty * 1_000 == pytest.approx(500)


def test_size_includes_round_trip_cost_allowance():
    r = Risk(risk_per_trade_pct=0.5)
    m0 = RiskManager(r, Costs(0, 0, 0, 0, 0), 100_000, 1.0)
    m1 = RiskManager(r, Costs(10, 2, 5, 5, 0), 100_000, 1.0)
    assert m1.size(100_000, 100, 2, 0, 0).qty < m0.size(100_000, 100, 2, 0, 0).qty


def test_leverage_cap_limits_notional():
    m, _ = rm(risk_per_trade_pct=2.0, max_leverage=1.0, max_position_pct=1000, max_total_open_risk_pct=10)
    sz = m.size(100_000, 100, 0.5, 0, 0)       # tiny stop => huge size, must be capped to 1x equity
    assert sz.notional_inr == pytest.approx(100_000)
    assert m.size(100_000, 100, 0.5, 100_000, 0).qty == 0   # already fully invested


def test_max_position_pct_cap():
    m, _ = rm(max_position_pct=25, risk_per_trade_pct=2.0, max_leverage=5, max_total_open_risk_pct=10)
    assert m.size(100_000, 100, 0.5, 0, 0).notional_inr == pytest.approx(25_000)


def test_min_notional_rejected():
    m, _ = rm(min_notional_inr=10_000, risk_per_trade_pct=0.1)
    assert m.size(100_000, 100, 50, 0, 0).qty == 0


def test_max_open_positions_and_total_risk():
    m, _ = rm(max_open_positions=2, max_total_open_risk_pct=1.0)
    ok, _w = m.can_enter(10, "A", 1, 0, 400, 100_000)
    assert ok
    assert m.can_enter(10, "B", 2, 0, 400, 100_000) == (False, "max_open_positions")
    assert m.can_enter(10, "B", 1, 1, 400, 100_000) == (False, "max_open_positions")   # pending counts
    assert m.can_enter(10, "B", 1, 0, 1000, 100_000) == (False, "max_total_open_risk")


def test_daily_loss_halt_then_reset_next_utc_day():
    m, ev = rm(max_daily_loss_pct=2.0)
    m.start_bar(T0, 1, 100_000)
    m.update_equity(T0, 1, 98_500)
    assert m.can_enter(2, "A", 0, 0, 0, 98_500)[0]
    m.update_equity(T0 + H, 2, 97_900)
    assert m.daily_halted and m.can_enter(3, "A", 0, 0, 0, 97_900) == (False, "daily_loss_halt")
    m.start_bar(T0 + 5 * H, 6, 97_900)               # same day: still halted
    assert m.daily_halted
    m.start_bar(T0 + DAY, 25, 97_900)                # next UTC day: reset
    assert not m.daily_halted and m.day_start_equity == 97_900
    kinds = [k for _, k, _ in ev]
    assert "daily_loss_halt" in kinds and "daily_loss_reset" in kinds


def test_drawdown_halt_cooldown_reset_rebaselines():
    m, ev = rm(max_drawdown_pct=10, drawdown_cooldown_bars=5, drawdown_reset_mode="cooldown", hard_stop_drawdown_pct=50)
    m.start_bar(T0, 1, 100_000)
    assert m.update_equity(T0, 1, 89_000) is True       # newly halted (11% dd)
    assert m.dd_halted and m.can_enter(2, "A", 0, 0, 0, 89_000) == (False, "drawdown_halt")
    for b in range(2, 6):
        m.update_equity(T0 + b * H, b, 89_000)
    assert m.dd_halted                                   # cooldown not elapsed
    m.update_equity(T0 + 6 * H, 6, 89_000)
    assert not m.dd_halted and m.peak == 89_000
    assert m.true_peak == 100_000                        # true peak never re-baselined
    assert "drawdown_reset" in [k for _, k, _ in ev]


def test_drawdown_manual_mode_never_auto_resets():
    m, _ = rm(max_drawdown_pct=10, drawdown_reset_mode="manual", drawdown_cooldown_bars=1, hard_stop_drawdown_pct=50)
    m.start_bar(T0, 1, 100_000)
    m.update_equity(T0, 1, 85_000)
    for b in range(2, 500):
        m.update_equity(T0 + b * H, b, 85_000)
    assert m.dd_halted
    m.manual_reset(T0 + 600 * H, 85_000)
    assert not m.dd_halted and m.can_enter(600, "A", 0, 0, 0, 85_000)[0]


def test_hard_stop_from_all_time_peak_needs_manual_reset():
    m, ev = rm(max_drawdown_pct=10, hard_stop_drawdown_pct=20, drawdown_cooldown_bars=2)
    m.start_bar(T0, 1, 100_000)
    m.update_equity(T0, 1, 89_000)                       # soft halt
    for b in range(2, 5):
        m.update_equity(T0 + b * H, b, 89_000)           # cooldown re-baselines to 89k
    assert not m.dd_halted
    m.update_equity(T0 + 6 * H, 6, 79_000)               # 21% below ALL-TIME peak -> hard stop
    assert m.hard_halted
    assert m.can_enter(7, "A", 0, 0, 0, 79_000) == (False, "hard_stop")
    for b in range(7, 400):
        m.update_equity(T0 + b * H, b, 79_000)
    assert m.hard_halted                                 # cooldown must NOT clear a hard stop
    m.manual_reset(T0 + 500 * H, 79_000)
    assert not m.hard_halted


def test_consecutive_loss_cooldown():
    m, ev = rm(max_consecutive_losses=3, loss_cooldown_bars=10, symbol_cooldown_bars=0)
    for i in range(3):
        m.on_trade_closed(T0, 5 + i, "A", -100)
    assert m.can_enter(8, "A", 0, 0, 0, 1e5) == (False, "loss_streak_cooldown")
    assert m.can_enter(14, "A", 0, 0, 0, 1e5) == (False, "loss_streak_cooldown")
    assert m.can_enter(18, "A", 0, 0, 0, 1e5)[0]
    assert "loss_streak_cooldown" in [k for _, k, _ in ev]


def test_win_resets_loss_streak():
    m, _ = rm(max_consecutive_losses=3, symbol_cooldown_bars=0)
    m.on_trade_closed(T0, 1, "A", -1)
    m.on_trade_closed(T0, 2, "A", -1)
    m.on_trade_closed(T0, 3, "A", +5)
    m.on_trade_closed(T0, 4, "A", -1)
    assert m.can_enter(5, "A", 0, 0, 0, 1e5)[0]


def test_symbol_cooldown_and_max_trades_per_day():
    m, _ = rm(symbol_cooldown_bars=4, max_trades_per_day=2)
    m.on_trade_closed(T0, 10, "A", 1)
    assert m.can_enter(12, "A", 0, 0, 0, 1e5) == (False, "symbol_cooldown")
    assert m.can_enter(12, "B", 0, 0, 0, 1e5)[0]
    m.trades_today = 2
    assert m.can_enter(20, "B", 0, 0, 0, 1e5) == (False, "max_trades_per_day")


# ---- engine-level behaviour -------------------------------------------------
def engine_cfg(**risk):
    c = load_config()
    c.costs = Costs(0, 0, 0, 0, 0)
    c.account.usdt_inr = 1.0
    c.market.symbols = [S]
    for k, v in risk.items():
        setattr(c.risk, k, v)
    return c


def test_daily_loss_blocks_new_entries_in_engine():
    c = engine_cfg(risk_per_trade_pct=1.0, max_daily_loss_pct=1.5, max_trades_per_day=100, symbol_cooldown_bars=0,
                   max_consecutive_losses=99, max_leverage=10, max_position_pct=1000)
    sink = MemorySink()
    eng = TradingEngine(c, "t", sink)
    i = 0
    losses = 0
    while losses < 2:   # two stopped-out trades = -2% > 1.5% limit
        eng.process_bar(T0 + i * H, {S: bar(100)}, {S: sig(stop=10)}); i += 1
        eng.process_bar(T0 + i * H, {S: bar(100)}, {}); i += 1
        eng.process_bar(T0 + i * H, {S: bar(95, h=96, l=80, c=85)}, {}); i += 1
        losses = len(sink.trades)
    n = len(sink.trades)
    eng.process_bar(T0 + i * H, {S: bar(100)}, {S: sig(stop=10)})
    eng.process_bar(T0 + (i + 1) * H, {S: bar(100)}, {})
    assert not eng.pf.positions and len(sink.trades) == n
    assert any(e["kind"] == "daily_loss_halt" for e in sink.risk_events)
    assert any(d.get("reason") == "daily_loss_halt" for d in sink.decisions)


def test_drawdown_halt_flattens_open_positions_and_blocks_entries():
    c = engine_cfg(risk_per_trade_pct=5.0, max_drawdown_pct=3.0, max_daily_loss_pct=100, hard_stop_drawdown_pct=50,
                   max_leverage=10, max_position_pct=1000, max_total_open_risk_pct=20, flatten_on_drawdown_halt=True)
    sink = MemorySink()
    eng = TradingEngine(c, "t", sink)
    eng.process_bar(T0, {S: bar(100)}, {S: sig(stop=50)})
    eng.process_bar(T0 + H, {S: bar(100)}, {})
    assert S in eng.pf.positions
    eng.process_bar(T0 + 2 * H, {S: bar(99, h=99, l=65, c=68)}, {})      # 100 units x -32 = -3.2% of equity (stop at 50 not hit)
    assert any(e["kind"] == "drawdown_halt" for e in sink.risk_events)
    eng.process_bar(T0 + 3 * H, {S: bar(68)}, {S: sig(stop=5)})          # flatten at this open; new entry refused
    assert not eng.pf.positions
    assert sink.trades[-1]["exit_reason"] == "drawdown_halt_flatten"
    eng.process_bar(T0 + 4 * H, {S: bar(68)}, {})
    assert not eng.pf.positions


def test_config_rejects_bad_risk_values():
    from cryptoalgo.config import ConfigError
    with pytest.raises(ConfigError):
        load_config(overrides={"risk": {"risk_per_trade_pct": 50}})
    with pytest.raises(ConfigError):
        load_config(overrides={"risk": {"hard_stop_drawdown_pct": 5.0}})
    with pytest.raises(ConfigError):
        load_config(overrides={"risk": {"drawdown_reset_mode": "yolo"}})
