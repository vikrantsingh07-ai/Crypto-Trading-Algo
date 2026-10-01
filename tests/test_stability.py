"""Long-running stability (accelerated): thousands of bars, random faults, restarts; invariants must hold."""
import random
import time
import tracemalloc

import pytest

from cryptoalgo.data.synthetic import generate
from cryptoalgo.paper.runner import PaperTrader
from cryptoalgo.storage.db import connect
from paper_helpers import db_scalar, db_trades, drive, make_cfg, make_trader

pytestmark = pytest.mark.slow


def test_long_run_with_random_faults_and_restarts_matches_clean_run(tmp_path):
    data, _ = generate(days=300, world="structured", seed=31)
    n = len(next(iter(data.values())))
    BARS = 3200

    clean_dir = tmp_path / "clean"; clean_dir.mkdir()
    t0, f0, c0 = make_trader(clean_dir, data, n - BARS)
    t0.start()
    drive(t0, f0, BARS, 8)
    ref = db_trades(c0.paper.db_path)
    assert len(ref) > 40

    d = tmp_path / "chaos"; d.mkdir()
    t, feed, cfg = make_trader(d, data, n - BARS)
    t.start()
    rnd = random.Random(99)
    done, restarts, errors, polls = 0, 0, 0, 0
    durations = []
    tracemalloc.start()
    mem_mid = None
    while done < BARS:
        step = min(rnd.choice([1, 3, 8, 12]), BARS - done)
        feed.release(step)
        done += step
        feed.overlap = rnd.choice([0, 0, 3, 20])
        if rnd.random() < 0.08:
            feed.fail_next(rnd.randint(1, 3))
        if rnd.random() < 0.03:                              # process crash + restart on the same DB
            t.conn.close()
            t = PaperTrader(cfg, feed, t.strategy, cfg.paper.db_path, clock=lambda: feed.now.timestamp(),
                            sleeper=lambda s: None, info={"label": "x"}, health_path=cfg.paper.health_file)
            t.start()
            restarts += 1
        for _ in range(50):
            s0 = time.perf_counter()
            try:
                t.poll_once()
                durations.append(time.perf_counter() - s0)
                break
            except Exception as e:
                errors += 1
                t.handle_error(e)
        else:
            raise AssertionError("never recovered")
        polls += 1
        if mem_mid is None and done > BARS // 2:
            mem_mid = tracemalloc.get_traced_memory()[0]
    mem_end = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    got = db_trades(cfg.paper.db_path)
    assert restarts >= 2 and errors >= 5 and polls > 100
    assert [x["trade_id"] for x in got] == [x["trade_id"] for x in ref], "faults/restarts changed the trade history"
    assert [round(x["net_pnl"], 6) for x in got] == [round(x["net_pnl"], 6) for x in ref]
    assert t.engine.equity == pytest.approx(t0.engine.equity)

    # DB integrity
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders") == 0
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) - COUNT(DISTINCT trade_id) FROM trades") == 0
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM equity") == db_scalar(cfg.paper.db_path, "SELECT COUNT(DISTINCT ts_ms) FROM equity")
    assert db_scalar(cfg.paper.db_path, "SELECT COUNT(*) FROM orders") == 2 * len(got) + len(t.engine.pf.positions)
    # no resource creep: memory after the second half within +25 MB of mid-run; poll time not growing >3x
    assert (mem_end - mem_mid) < 25e6
    k = max(10, len(durations) // 10)
    assert sorted(durations[-k:])[k // 2] < 3 * sorted(durations[:k])[k // 2] + 0.05
