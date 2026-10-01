import json

from cryptoalgo.config import load_config
from cryptoalgo.paper.feeds import SimulatedFeed
from cryptoalgo.paper.runner import PaperTrader
from cryptoalgo.storage.db import connect
from cryptoalgo.strategies.library import get_strategy


def make_cfg(tmp_path, history_bars=1500):
    cfg = load_config()
    cfg.paper.db_path = str(tmp_path / "paper.db")
    cfg.paper.health_file = str(tmp_path / "health.json")
    cfg.paper.history_bars = history_bars
    return cfg


def make_trader(tmp_path, data, start_pos, strategy="trend_breakout", process_history=False, cfg=None, feed=None, params=None):
    cfg = cfg or make_cfg(tmp_path)
    feed = feed or SimulatedFeed(data, cfg.market.timeframe, start_pos=start_pos)
    st = get_strategy(strategy, params, cfg.market.timeframe, cfg.market.htf)
    t = PaperTrader(cfg, feed, st, cfg.paper.db_path, clock=lambda: feed.now.timestamp(), sleeper=lambda s: None,
                    info={"validated": False, "label": "UNVALIDATED (test)"}, process_history=process_history,
                    health_path=cfg.paper.health_file)
    return t, feed, cfg


def drive(trader, feed, bars, per_poll=5, fault=None):
    """Release bars in chunks and poll; `fault(i)` may inject failures before poll i. Returns #errors handled."""
    errors, i, done = 0, 0, 0
    while done < bars and not feed.exhausted:
        n = feed.release(min(per_poll, bars - done))
        done += n
        if fault:
            fault(i)
        i += 1
        for _ in range(50):          # retry loop == what run_forever does
            try:
                trader.poll_once()
                break
            except Exception as e:
                errors += 1
                trader.handle_error(e)
        else:
            raise AssertionError("trader never recovered")
    return errors


def db_trades(path):
    c = connect(path, read_only=True)
    try:
        return [dict(r) for r in c.execute("SELECT * FROM trades ORDER BY exit_ts_ms, trade_id")]
    finally:
        c.close()


def db_scalar(path, sql):
    c = connect(path, read_only=True)
    try:
        return c.execute(sql).fetchone()[0]
    finally:
        c.close()
