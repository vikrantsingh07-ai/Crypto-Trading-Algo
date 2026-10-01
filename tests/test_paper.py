"""Paper execution, duplicate prevention, failure/reconnection, recovery, backtest equivalence."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from cryptoalgo.backtest.runner import run_backtest
from cryptoalgo.config import Costs, load_config
from cryptoalgo.data.synthetic import generate
from cryptoalgo.engine import TradingEngine
from cryptoalgo.events import MemorySink
from cryptoalgo.execution import DuplicateOrderError, Order, PaperBroker, make_client_order_id
from cryptoalgo.monitoring import check_health
from cryptoalgo.paper.feeds import FeedError, SimulatedFeed
from cryptoalgo.paper.runner import PaperTrader, ValidationRequired, load_validated_strategy
from cryptoalgo.safety import SafetyError
from cryptoalgo.storage.db import connect, init_db
from cryptoalgo.strategies.library import get_strategy
from conftest import H, T0, bar, sig
from paper_helpers import db_scalar, db_trades, drive, make_cfg, make_trader


@pytest.fixture(scope="module")
def data():
    d, _ = generate(days=170, world="structured", seed=21)
    return d


# ------------------------------------------------------------------ paper order execution
def test_broker_market_stop_tp_prices():
    b = PaperBroker(Costs(10, 2, 5, 5, 0))
    buy = b.fill_market(Order("a", 0, "X", "buy", "market", 2.0, "entry"), 100.0)
    assert buy.price == pytest.approx(100.07) and buy.fee_usdt == pytest.approx(100.07 * 2 * 0.001)
    sell = b.fill_market(Order("b", 0, "X", "sell", "market", 2.0, "exit"), 100.0)
    assert sell.price == pytest.approx(99.93)
    stop = b.fill_stop(Order("c", 0, "X", "sell", "stop", 1.0, "stop"), 90.0, 100.0)
    assert stop.price == pytest.approx(90 * (1 - 12e-4))                 # worse than the stop level
    gap = b.fill_stop(Order("d", 0, "X", "sell", "stop", 1.0, "stop"), 90.0, 80.0)
    assert gap.price == pytest.approx(80 * (1 - 12e-4))                  # gapped: fills at the open, not at 90
    tp = b.fill_take_profit(Order("e", 0, "X", "sell", "take_profit", 1.0, "tp"), 120.0, 118.0)
    assert tp.price == 120.0
    tp_gap = b.fill_take_profit(Order("f", 0, "X", "sell", "take_profit", 1.0, "tp"), 120.0, 125.0)
    assert tp_gap.price == 125.0


def test_every_fill_is_simulated_and_recorded(tmp_path, data):
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(tmp_path, data, n - 1500)
    t.start()
    drive(t, feed, 1500)
    orders = db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM orders")
    trades = db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM trades")
    open_pos = len(t.engine.pf.positions)
    assert trades > 3 and orders == 2 * trades + open_pos
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM fees") == orders
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM trades WHERE trade_id IS NULL OR exit_ts_ms IS NULL") == 0
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM signals") > 0
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM equity") >= 1500 - 5


# ------------------------------------------------------------------ duplicate-order prevention
def test_client_order_id_deterministic_and_distinct():
    a = make_client_order_id("s", "BTCUSDT", 1000, "entry:1")
    assert a == make_client_order_id("s", "BTCUSDT", 1000, "entry:1")
    assert len({a, make_client_order_id("s", "BTCUSDT", 1000, "entry:-1"), make_client_order_id("s", "ETHUSDT", 1000, "entry:1"),
                make_client_order_id("s", "BTCUSDT", 2000, "entry:1")}) == 4


def test_broker_rejects_second_fill_of_same_order_id():
    b = PaperBroker(Costs(0, 0, 0, 0, 0))
    o = Order("dup", 0, "X", "buy", "market", 1, "entry")
    b.fill_market(o, 100)
    with pytest.raises(DuplicateOrderError):
        b.fill_market(o, 100)


def test_engine_blocks_entry_whose_order_id_already_executed(exact_cfg):
    cid = make_client_order_id("t", "BTCUSDT", T0, "entry:1")
    sink = MemorySink()
    eng = TradingEngine(exact_cfg, "t", sink, PaperBroker(exact_cfg.costs, {cid}))
    eng.process_bar(T0, {"BTCUSDT": bar(100)}, {"BTCUSDT": sig(stop=10)})
    eng.process_bar(T0 + H, {"BTCUSDT": bar(100)}, {})
    assert not eng.pf.positions and not sink.orders
    assert any(e["kind"] == "duplicate_order_blocked" for e in sink.risk_events)


def test_db_unique_order_id_constraint():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    row = ("id1", 1, "X", "buy", "market", "entry", 1.0, 1.0, 0.0, 0.0, "filled", "", "p")
    conn.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)


# ------------------------------------------------------------------ equivalence: paper path == backtest
def test_paper_replay_matches_backtest_trade_for_trade(tmp_path, data):
    n = 2200
    sub = {s: d.iloc[:n] for s, d in data.items()}
    cfg = make_cfg(tmp_path, history_bars=10000)    # window must cover all history so warm-up matches the backtest
    st = get_strategy("trend_breakout", None, cfg.market.timeframe, cfg.market.htf)
    bt = run_backtest(sub, cfg, st, liquidate_at_end=False)
    feed = SimulatedFeed(sub, cfg.market.timeframe, start_pos=500)
    t = PaperTrader(cfg, feed, st, cfg.paper.db_path, clock=lambda: feed.now.timestamp(), sleeper=lambda s: None,
                    process_history=True, health_path=None)
    t.start()
    drive(t, feed, n - 500, per_poll=23)
    got = db_trades(cfg.paper.db_path)
    exp = bt.trades.sort_values(["exit_ts_ms", "trade_id"]).reset_index(drop=True)
    assert len(got) == len(exp) > 10
    for g, e in zip(got, exp.to_dict("records")):
        assert g["trade_id"] == e["trade_id"] and g["exit_reason"] == e["exit_reason"]
        assert g["entry_price"] == pytest.approx(e["entry_price"]) and g["exit_price"] == pytest.approx(e["exit_price"])
        assert g["net_pnl"] == pytest.approx(e["net_pnl"])
    assert t.engine.equity == pytest.approx(float(bt.equity.iloc[-1]), rel=1e-9)


# ------------------------------------------------------------------ failure / reconnection
def _reference(tmp_path, data, bars=1200, per_poll=5):
    d = tmp_path / "ref"
    d.mkdir()
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(d, data, n - bars)
    t.start()
    drive(t, feed, bars, per_poll)
    return db_trades(cfg.paper.db_path), t.engine.equity


def _same(a, b):
    assert [x["trade_id"] for x in a] == [x["trade_id"] for x in b]
    assert [round(x["net_pnl"], 6) for x in a] == [round(x["net_pnl"], 6) for x in b]


def test_recovers_from_feed_failures_without_duplicates(tmp_path, data):
    ref, ref_eq = _reference(tmp_path, data)
    d = tmp_path / "faulty"
    d.mkdir()
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(d, data, n - 1200)
    t.start()
    errs = drive(t, feed, 1200, 5, fault=lambda i: feed.fail_next(2 if i % 9 == 0 else 0))
    assert errs >= 20
    got = db_trades(cfg.paper.db_path)
    _same(ref, got)
    assert t.engine.equity == pytest.approx(ref_eq)
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM errors") >= errs
    assert t.consecutive_errors == 0                       # recovered
    ok, msg = check_health(cfg.paper.health_file, now=t.clock())
    assert ok, msg


def test_overlapping_candles_after_reconnect_are_harmless(tmp_path, data):
    ref, _ = _reference(tmp_path, data)
    d = tmp_path / "ov"
    d.mkdir()
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(d, data, n - 1200)
    feed.overlap = 12                                       # exchange re-sends 12 already-processed bars each time
    t.start()
    drive(t, feed, 1200, 5)
    _same(ref, db_trades(cfg.paper.db_path))


def test_backoff_grows_exponentially_with_cap_and_resets(tmp_path, data):
    t, feed, cfg = make_trader(tmp_path, data, 1000)
    t.start()
    delays = [t.handle_error(FeedError("x")) for _ in range(12)]
    assert delays[1] > delays[0] and delays[3] > delays[2]
    assert max(delays) <= cfg.paper.backoff_max_s * 1.2 + 1e-9
    feed.release(3)
    t.poll_once()
    assert t.consecutive_errors == 0


def test_status_degrades_then_fails_then_recovers(tmp_path, data):
    t, feed, cfg = make_trader(tmp_path, data, 1000)
    t.start()
    feed.release(2)
    feed.fail_next(100)
    for _ in range(3):
        try:
            t.poll_once()
        except FeedError as e:
            t.handle_error(e)
    h = json.loads(open(cfg.paper.health_file).read())
    assert h["status"] == "DEGRADED" and h["consecutive_errors"] == 3 and "PAPER" in h["banner"]
    for _ in range(20):
        t.handle_error(FeedError("still down"))
    assert json.loads(open(cfg.paper.health_file).read())["status"] == "FAILED"
    assert not check_health(cfg.paper.health_file, now=t.clock())[0]
    feed._fail = 0
    t.poll_once()
    assert json.loads(open(cfg.paper.health_file).read())["status"] == "RUNNING"


def test_stale_heartbeat_detected(tmp_path, data):
    t, feed, cfg = make_trader(tmp_path, data, 1000)
    t.start()
    t.poll_once()
    assert check_health(cfg.paper.health_file, now=t.last_poll_epoch + 10)[0]
    assert not check_health(cfg.paper.health_file, now=t.last_poll_epoch + 1000)[0]
    assert not check_health(str(tmp_path / "nope.json"))[0]


def test_commit_failure_rolls_back_whole_bar_and_retries(tmp_path, data):
    ref, ref_eq = _reference(tmp_path, data, bars=900)
    d = tmp_path / "crashy"
    d.mkdir()
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(d, data, n - 900)
    t.start()
    real = t.sink.end_bar
    state = {"calls": 0}

    def flaky(ts, eng):
        state["calls"] += 1
        if state["calls"] in (40, 41, 200):
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        return real(ts, eng)
    t.sink.end_bar = flaky
    errs = drive(t, feed, 900, 5)
    assert errs >= 3
    got = db_trades(cfg.paper.db_path)
    _same(ref, got)
    assert t.engine.equity == pytest.approx(ref_eq)
    ids = [r[0] for r in connect(cfg.paper.db_path).execute("SELECT order_id FROM orders")]
    assert len(ids) == len(set(ids))


def test_one_symbol_lagging_blocks_processing_until_both_have_the_bar(tmp_path, data):
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(tmp_path, data, n - 600)
    t.start()
    feed.hold("ETHUSDT")                                    # ETH is one bar behind BTC
    feed.release(5)
    assert t.poll_once() == 4                               # only the 4 bars BOTH symbols have; no guessing the 5th
    last = t.engine.last_ts_ms
    assert t.poll_once() == 0                               # nothing new until ETH catches up
    feed.unhold("ETHUSDT")
    assert t.poll_once() == 1 and t.engine.last_ts_ms > last


# ------------------------------------------------------------------ restart / recovery
def test_restart_resumes_exactly_and_processes_missed_bars(tmp_path, data):
    ref, ref_eq = _reference(tmp_path, data, bars=1000)
    d = tmp_path / "restart"
    d.mkdir()
    n = len(next(iter(data.values())))
    t1, feed, cfg = make_trader(d, data, n - 1000)
    t1.start()
    drive(t1, feed, 400, 5)
    t1.conn.close()                                          # process dies
    feed.release(60)                                         # 60 bars arrive while the bot is down
    t2 = PaperTrader(cfg, feed, t1.strategy, cfg.paper.db_path, clock=lambda: feed.now.timestamp(),
                     sleeper=lambda s: None, info={"label": "x"}, health_path=None)
    t2.start()
    assert t2.engine.last_ts_ms == t1.engine.last_ts_ms     # resumed from committed state, not from "now"
    drive(t2, feed, 1000 - 460, 5)
    _same(ref, db_trades(cfg.paper.db_path))
    assert t2.engine.equity == pytest.approx(ref_eq)


def test_fresh_start_does_not_trade_history(tmp_path, data):
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(tmp_path, data, n - 1000)
    t.start()
    assert t.poll_once() == 0
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM orders") == 0


# ------------------------------------------------------------------ validation gate & safety
def test_paper_refused_without_passing_validation(tmp_path):
    cfg = load_config()
    cfg.paper.validation_report = str(tmp_path / "none.json")
    with pytest.raises(ValidationRequired):
        load_validated_strategy(cfg)
    st, info = load_validated_strategy(cfg, allow_unvalidated=True)
    assert not info["validated"] and "UNVALIDATED" in info["label"]
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"passed": False, "chosen_strategy": "trend_breakout", "params": {}, "timeframe": "1h",
                               "symbols": cfg.market.symbols}))
    cfg.paper.validation_report = str(rep)
    with pytest.raises(ValidationRequired):
        load_validated_strategy(cfg)
    rep.write_text(json.dumps({"passed": True, "chosen_strategy": "trend_breakout", "params": {"donch_n": 30},
                               "timeframe": "1h", "symbols": cfg.market.symbols}))
    st, info = load_validated_strategy(cfg)
    assert info["validated"] and st.params["donch_n"] == 30
    rep.write_text(json.dumps({"passed": True, "chosen_strategy": "trend_breakout", "params": {}, "timeframe": "15m",
                               "symbols": cfg.market.symbols}))
    with pytest.raises(ValidationRequired):                  # report for a different timeframe must not unlock paper
        load_validated_strategy(cfg)


def test_trader_refuses_non_paper_mode(tmp_path, data):
    cfg = make_cfg(tmp_path)
    cfg.mode = "live"
    with pytest.raises(SafetyError):
        PaperTrader(cfg, SimulatedFeed(data, "1h", 100), get_strategy("trend_breakout"), cfg.paper.db_path)


def test_startup_survives_feed_outage_and_retries(tmp_path, data):
    """Regression: a restart while the feed is down must back off and retry, not crash."""
    n = len(next(iter(data.values())))
    delays = []
    cfg = make_cfg(tmp_path)
    feed = SimulatedFeed(data, cfg.market.timeframe, start_pos=n - 500)
    feed.fail_next(4)
    st = get_strategy("trend_breakout", None, cfg.market.timeframe, cfg.market.htf)
    t = PaperTrader(cfg, feed, st, cfg.paper.db_path, clock=lambda: feed.now.timestamp(), sleeper=delays.append,
                    info={"label": "x"}, health_path=cfg.paper.health_file)
    t.start()                                              # would have raised FeedError before the fix
    assert len(delays) == 4 and delays[1] > delays[0]      # exponential back-off was applied
    assert t.engine.last_ts_ms > 0 and t.consecutive_errors == 4
    feed.release(3)
    assert t.poll_once() == 3 and t.consecutive_errors == 0
