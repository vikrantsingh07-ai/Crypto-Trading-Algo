"""SQLite persistence for the paper-trading engine.

Every signal, decision, order, fill fee, position event, trade, risk event and
error is stored. Each processed bar is committed in ONE transaction together
with the engine state snapshot, so a crash can never leave half a bar behind and
a restart resumes exactly after the last committed bar.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ..events import Sink

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT, signal INTEGER, stop_dist REAL, tp_dist REAL,
  regime TEXT, close REAL);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT, action TEXT, reason TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS orders(
  order_id TEXT PRIMARY KEY, ts_ms INTEGER, symbol TEXT, side TEXT, kind TEXT, purpose TEXT, qty REAL, price REAL,
  fee_usdt REAL, fee_inr REAL, status TEXT, reason TEXT, position_id TEXT);
CREATE TABLE IF NOT EXISTS fees(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, order_id TEXT UNIQUE, symbol TEXT, fee_usdt REAL, fee_inr REAL);
CREATE TABLE IF NOT EXISTS positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, position_id TEXT, symbol TEXT, action TEXT, side TEXT,
  qty REAL, price REAL, stop REAL, tp REAL);
CREATE TABLE IF NOT EXISTS trades(
  trade_id TEXT PRIMARY KEY, position_id TEXT, symbol TEXT, side TEXT, qty REAL, entry_ts_ms INTEGER, entry_price REAL,
  exit_ts_ms INTEGER, exit_price REAL, gross_pnl REAL, fees REAL, funding REAL, net_pnl REAL, r_multiple REAL,
  exit_reason TEXT, bars_held INTEGER, source TEXT, notional_inr REAL, strategy TEXT);
CREATE TABLE IF NOT EXISTS risk_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS errors(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, component TEXT, message TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS equity(
  ts_ms INTEGER PRIMARY KEY, equity REAL, cash REAL, unrealized REAL, n_open INTEGER, drawdown_pct REAL, regime TEXT);
CREATE TABLE IF NOT EXISTS engine_state(
  id INTEGER PRIMARY KEY CHECK(id=1), ts_ms INTEGER, state TEXT);
CREATE TABLE IF NOT EXISTS processed_bars(
  symbol TEXT, ts_ms INTEGER, PRIMARY KEY(symbol, ts_ms));
CREATE TABLE IF NOT EXISTS status(
  key TEXT PRIMARY KEY, value TEXT, updated_ms INTEGER);
CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(exit_ts_ms);
"""


def connect(path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=10)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


class SQLiteSink(Sink):
    """Buffers one bar's events and writes them atomically in end_bar()."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.lock = threading.Lock()
        self._buf: list[tuple[str, tuple]] = []

    # -- buffering ---------------------------------------------------------
    def _q(self, sql: str, args: tuple) -> None:
        self._buf.append((sql, args))

    def begin_bar(self, ts_ms):
        self._buf.clear()

    def signal(self, r):
        self._q("INSERT INTO signals(ts_ms,symbol,signal,stop_dist,tp_dist,regime,close) VALUES(?,?,?,?,?,?,?)",
                (r["ts_ms"], r["symbol"], r["signal"], r.get("stop_dist"), r.get("tp_dist"), r.get("regime"), r.get("close")))

    def decision(self, r):
        extra = {k: v for k, v in r.items() if k not in ("ts_ms", "symbol", "action", "reason")}
        self._q("INSERT INTO decisions(ts_ms,symbol,action,reason,detail) VALUES(?,?,?,?,?)",
                (r["ts_ms"], r["symbol"], r["action"], r.get("reason"), json.dumps(extra, default=float)))

    def order(self, r):
        # plain INSERT: the PRIMARY KEY makes a duplicate client_order_id fail loudly and roll the bar back
        self._q("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["order_id"], r["ts_ms"], r["symbol"], r["side"], r["kind"], r["purpose"], r["qty"], r["price"],
                 r["fee_usdt"], r["fee_inr"], r["status"], r["reason"], r["position_id"]))
        self._q("INSERT INTO fees(ts_ms,order_id,symbol,fee_usdt,fee_inr) VALUES(?,?,?,?,?)",
                (r["ts_ms"], r["order_id"], r["symbol"], r["fee_usdt"], r["fee_inr"]))

    def position(self, r):
        self._q("INSERT INTO positions(ts_ms,position_id,symbol,action,side,qty,price,stop,tp) VALUES(?,?,?,?,?,?,?,?,?)",
                (r["ts_ms"], r["position_id"], r["symbol"], r["action"], r.get("side"), r.get("qty"), r.get("price"),
                 r.get("stop"), r.get("tp")))

    def trade(self, r):
        cols = ["trade_id", "position_id", "symbol", "side", "qty", "entry_ts_ms", "entry_price", "exit_ts_ms",
                "exit_price", "gross_pnl", "fees", "funding", "net_pnl", "r_multiple", "exit_reason", "bars_held",
                "source", "notional_inr", "strategy"]
        self._q(f"INSERT INTO trades({','.join(cols)}) VALUES({','.join('?' * len(cols))})", tuple(r[c] for c in cols))

    def risk_event(self, r):
        self._q("INSERT INTO risk_events(ts_ms,kind,detail) VALUES(?,?,?)", (r["ts_ms"], r["kind"], json.dumps(r["detail"], default=float)))

    def equity(self, r):
        self._q("INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?,?,?)",
                (r["ts_ms"], r["equity"], r["cash"], r["unrealized"], r["n_open"], r["drawdown_pct"], r.get("regime", "")))

    def end_bar(self, ts_ms, engine):
        with self.lock:
            try:
                cur = self.conn.cursor()
                cur.execute("BEGIN")
                for sql, args in self._buf:
                    cur.execute(sql, args)
                cur.execute("INSERT OR REPLACE INTO engine_state VALUES(1,?,?)", (ts_ms, json.dumps(engine.state_dict())))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                self._buf.clear()

    # -- out-of-band records (not part of a bar) ---------------------------
    def log_error(self, component: str, message: str, detail: str = "") -> None:
        with self.lock:
            self.conn.execute("INSERT INTO errors(ts_ms,component,message,detail) VALUES(?,?,?,?)",
                              (int(time.time() * 1000), component, message[:500], detail[:2000]))
            self.conn.commit()

    def set_status(self, key: str, value) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO status VALUES(?,?,?)",
                              (key, json.dumps(value, default=str), int(time.time() * 1000)))
            self.conn.commit()

    def mark_processed(self, symbol: str, ts_ms: int) -> None:
        self.conn.execute("INSERT OR IGNORE INTO processed_bars VALUES(?,?)", (symbol, ts_ms))


def load_engine_state(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT state FROM engine_state WHERE id=1").fetchone()
    return json.loads(row["state"]) if row else None


def load_order_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["order_id"] for r in conn.execute("SELECT order_id FROM orders")}
