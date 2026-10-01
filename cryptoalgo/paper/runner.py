"""24/7 paper-trading runner (PAPER ONLY).

Loop: poll closed candles -> (dedupe, gap-check) -> signals -> shared TradingEngine -> atomic SQLite commit.
Failure handling: FeedError / unexpected exceptions are logged to the DB, health goes DEGRADED, the loop backs off
exponentially (with jitter) and retries. Because bars are processed once (state in DB) and orders carry
deterministic ids with a UNIQUE constraint, retries and restarts cannot duplicate trades.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

from ..config import Config
from ..data.models import from_ms, tf_timedelta, to_ms
from ..data.store import CandleStore
from ..engine import TradingEngine, rows_from_signals
from ..execution import PaperBroker
from .. import regime as regime_mod
from ..monitoring import write_health
from ..safety import PAPER_BANNER, assert_paper_only
from ..storage.db import SQLiteSink, connect, init_db, load_engine_state, load_order_ids
from ..strategies.library import get_strategy
from .feeds import FeedError

log = logging.getLogger("cryptoalgo.paper")


class ValidationRequired(RuntimeError):
    pass


def load_validated_strategy(cfg: Config, allow_unvalidated: bool = False):
    """Return (strategy, info). Refuses to run unless the latest validation report passed all gates."""
    path = Path(cfg.paper.validation_report)
    info = {"validated": False, "label": "UNVALIDATED (plumbing/demo run)", "report": str(path)}
    if path.exists():
        rep = json.loads(path.read_text())
        ok = bool(rep.get("passed"))
        match = rep.get("timeframe") == cfg.market.timeframe and set(rep.get("symbols", [])) == set(cfg.market.symbols)
        if ok and match:
            st = get_strategy(rep["chosen_strategy"], rep.get("params") or None, cfg.market.timeframe, cfg.market.htf,
                              bool(rep.get("use_htf")))
            info.update(validated=True, label="VALIDATED (all robustness gates passed)", created=rep.get("created_utc"),
                        data_fingerprint=rep.get("data_fingerprint"))
            return st, info
        why = "report did not pass all gates" if not ok else "report symbols/timeframe do not match config"
    else:
        why = f"no validation report at {path}"
    if not allow_unvalidated:
        raise ValidationRequired(
            f"Paper trading refused: {why}. Run scripts/run_validation.py on real data first, or pass "
            "--allow-unvalidated for a clearly-labelled plumbing demo.")
    st = get_strategy(cfg.strategy.name, cfg.strategy.params or None, cfg.market.timeframe, cfg.market.htf,
                      cfg.strategy.use_htf_confirmation)
    info["reason"] = why
    return st, info


class PaperTrader:
    def __init__(self, cfg: Config, feed, strategy, db_path: str | None = None, clock=time.time, sleeper=time.sleep,
                 info: dict | None = None, process_history: bool = False, health_path: str | None = None,
                 rng: random.Random | None = None):
        assert_paper_only(cfg)
        self.cfg, self.feed, self.strategy = cfg, feed, strategy
        self.tf = cfg.market.timeframe
        self.clock, self.sleeper = clock, sleeper
        self.info = info or {"validated": False, "label": "UNVALIDATED"}
        self.db_path = db_path or cfg.paper.db_path
        self.health_path = health_path if health_path is not None else cfg.paper.health_file
        self.process_history = process_history
        self.rng = rng or random.Random(1)
        self.conn = connect(self.db_path)
        init_db(self.conn)
        self.store = CandleStore(self.conn)
        self.sink = SQLiteSink(self.conn)
        self.engine: TradingEngine | None = None
        self.consecutive_errors = 0
        self.last_error: str | None = None
        self.last_poll_epoch = 0.0
        self.last_ok_epoch = 0.0
        self.bars_processed = 0
        self.started_epoch = clock()
        self.stop_event = threading.Event()
        self.regimes: dict[str, str] = {}
        self._build_engine()

    # ------------------------------------------------------------------ setup / recovery
    def _build_engine(self) -> None:
        """(Re)build the engine from the last COMMITTED state. Used at startup and after any failed bar."""
        broker = PaperBroker(self.cfg.costs, load_order_ids(self.conn))
        eng = TradingEngine(self.cfg, self.strategy.name, self.sink, broker,
                            bar_hours=tf_timedelta(self.tf).total_seconds() / 3600)
        st = load_engine_state(self.conn)
        if st:
            eng.load_state(st)
        self.engine = eng

    def start(self) -> None:
        log.info("%s | strategy=%s params=%s | %s", PAPER_BANNER, self.strategy.name, self.strategy.params, self.info["label"])
        self.sink.set_status("validation", self.info)
        self.sink.set_status("strategy", {"name": self.strategy.name, "params": self.strategy.params,
                                          "symbols": self.cfg.market.symbols, "timeframe": self.tf})
        self.sink.set_status("config", {"initial_capital": self.cfg.account.initial_capital, "risk": vars(self.cfg.risk),
                                        "costs": vars(self.cfg.costs), "usdt_inr": self.cfg.account.usdt_inr})
        # Startup must survive a feed outage too (e.g. a restart during a network blip): retry with back-off.
        while not self.stop_event.is_set():
            try:
                self.backfill()
                break
            except Exception as e:
                self.sleeper(self.handle_error(e))
        if self.stop_event.is_set() and not self.store.load(self.cfg.market.symbols[0], self.tf, 1).shape[0]:
            return
        if self.engine.last_ts_ms < 0 and not self.process_history:
            # fresh start: warm up on history, trade only bars that close AFTER now
            last = min(self.store.load(s, self.tf, 1).index[-1] for s in self.cfg.market.symbols)
            self.engine.last_ts_ms = to_ms(last)
            self.sink.conn.execute("INSERT OR REPLACE INTO engine_state VALUES(1,?,?)",
                                   (self.engine.last_ts_ms, json.dumps(self.engine.state_dict())))
            self.sink.conn.commit()
        self._write_health("RUNNING")

    def backfill(self) -> None:
        for s in self.cfg.market.symbols:
            df = self.feed.fetch_closed(s, self.tf, limit=min(1000, self.cfg.paper.history_bars))
            if not df.empty:
                self.store.upsert(s, self.tf, df)

    # ------------------------------------------------------------------ one polling cycle
    def poll_once(self) -> int:
        """Fetch -> process every new closed bar once. Returns number of bars processed."""
        self.last_poll_epoch = self.clock()
        eng = self.engine
        since = from_ms(eng.last_ts_ms) if eng.last_ts_ms > 0 else None
        for s in self.cfg.market.symbols:
            df = self.feed.fetch_closed(s, self.tf, limit=1000, since=since)
            if not df.empty:
                self.store.upsert(s, self.tf, df)
        hist = {s: self.store.load(s, self.tf, limit=self.cfg.paper.history_bars + 1000) for s in self.cfg.market.symbols}
        common = None
        for s, d in hist.items():
            ts = {to_ms(t) for t in d.index if to_ms(t) > eng.last_ts_ms}
            common = ts if common is None else common & ts
        new_ts = sorted(common or ())
        n = 0
        if new_ts:
            sig_rows, windows = {}, {}
            for s, d in hist.items():
                if len(d) >= self.strategy.min_bars:
                    mb = self.strategy.min_bars      # same warm-up rule as the backtester
                    rows = rows_from_signals(self.strategy.signals(d))
                    sig_rows[s] = dict(zip((to_ms(t) for t in d.index[mb:]), rows[mb:]))
                    self.regimes[s] = str(regime_mod.live_regime(d).iloc[-1])
                windows[s] = d
            step = tf_timedelta(self.tf)
            for ts in new_ts:
                if eng.last_ts_ms > 0 and ts - eng.last_ts_ms > step.total_seconds() * 1000 * 1.5:
                    self.sink.log_error("data", f"gap before bar {from_ms(ts)}: {from_ms(eng.last_ts_ms)} -> {from_ms(ts)}")
                    log.warning("gap in candles before %s", from_ms(ts))
                t = from_ms(ts)
                bars = {s: tuple(windows[s].loc[t, ["open", "high", "low", "close", "volume"]]) for s in windows}
                sigs = {s: sig_rows[s].get(ts) for s in sig_rows}
                try:
                    done = eng.process_bar(ts, bars, sigs, market_regime=self.regimes.get(self.cfg.market.symbols[0], ""))
                except Exception:
                    # roll back in-memory state to the last committed bar; the bar is retried next cycle
                    self._build_engine()
                    raise
                if done:
                    n += 1
                    self.bars_processed += 1
                eng = self.engine
        self.consecutive_errors, self.last_ok_epoch = 0, self.clock()
        self._publish("RUNNING")
        return n

    # ------------------------------------------------------------------ resilience
    def handle_error(self, exc: Exception) -> float:
        self.consecutive_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.sink.log_error("poll", self.last_error, traceback.format_exc() if not isinstance(exc, FeedError) else "")
        log.warning("poll failed (%d in a row): %s", self.consecutive_errors, self.last_error)
        status = "FAILED" if self.consecutive_errors >= 20 else "DEGRADED"
        self._publish(status)
        c = self.cfg.paper
        delay = min(c.backoff_max_s, c.backoff_initial_s * 2 ** (self.consecutive_errors - 1))
        return delay * (1 + 0.2 * self.rng.random())

    def run_forever(self, max_cycles: int | None = None) -> None:
        self.start()
        cycles = 0
        while not self.stop_event.is_set():
            try:
                self.poll_once()
                delay = self.cfg.paper.poll_seconds
            except Exception as e:           # network, bad payload, DB hiccup, bug: never die, never fabricate trades
                delay = self.handle_error(e)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self.sleeper(delay)
        self._publish("STOPPED")

    def stop(self) -> None:
        self.stop_event.set()

    # ------------------------------------------------------------------ status
    def _risk_status(self) -> str:
        r = self.engine.risk
        if r.hard_halted:
            return "HARD_STOP"
        if r.dd_halted:
            return "DRAWDOWN_HALT"
        if r.daily_halted:
            return "DAILY_LOSS_HALT"
        if self.engine.bar_idx < r.loss_pause_until:
            return "LOSS_STREAK_COOLDOWN"
        return "OK"

    def _publish(self, status: str) -> None:
        e = self.engine
        last_bar = from_ms(e.last_ts_ms) if e.last_ts_ms > 0 else None
        age_bars = None
        if last_bar is not None:
            age_bars = (pd.Timestamp(self.clock(), unit="s", tz="UTC") - (last_bar + tf_timedelta(self.tf))) / tf_timedelta(self.tf)
            if status == "RUNNING" and age_bars > self.cfg.paper.max_data_age_bars:
                status = "DEGRADED"
                self.last_error = f"stale data: last bar {last_bar}"
        payload = {"status": status, "mode": "PAPER", "last_poll_epoch": self.last_poll_epoch,
                   "last_ok_epoch": self.last_ok_epoch, "last_error": self.last_error,
                   "consecutive_errors": self.consecutive_errors, "last_bar": str(last_bar),
                   "data_age_bars": age_bars, "bars_processed": self.bars_processed,
                   "uptime_s": self.clock() - self.started_epoch, "equity": e.equity, "risk_status": self._risk_status(),
                   "open_positions": len(e.pf.positions), "regimes": self.regimes, "validation": self.info["label"]}
        self.sink.set_status("bot", payload)
        self.sink.set_status("risk", {"status": self._risk_status(), "peak": e.risk.peak, "true_peak": e.risk.true_peak,
                                      "day_start_equity": e.risk.day_start_equity, "loss_streak": e.risk.loss_streak,
                                      "trades_today": e.risk.trades_today, "dd_halted": e.risk.dd_halted,
                                      "daily_halted": e.risk.daily_halted, "hard_halted": e.risk.hard_halted})
        if self.health_path:
            write_health(self.health_path, payload)

    def _write_health(self, status: str) -> None:
        self._publish(status)
