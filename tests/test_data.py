"""Data-integrity tests: validator, Binance public client (mocked HTTP), CSV, store."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from cryptoalgo.data.binance import BinancePublicClient, MarketDataError
from cryptoalgo.data.integrity import DataIntegrityError, check_candles, clean_frame, require_clean
from cryptoalgo.data.loader import load_csv, save_csv
from cryptoalgo.data.models import closed_only, from_ms, to_ms
from cryptoalgo.data.store import CandleStore
from cryptoalgo.data.synthetic import generate


def frame(n=50, tf="1h"):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = 100 + np.arange(n) * 0.1
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c + 0.2, "volume": 10.0}, index=idx)


def test_clean_frame_passes():
    assert check_candles(frame(), "1h").ok


def test_detects_duplicates_nan_gaps_bad_ohlc_negative():
    f = frame()
    assert not check_candles(pd.concat([f, f.iloc[[3]]]).sort_index(), "1h").ok
    g = f.copy(); g.iloc[5, 0] = np.nan
    assert not check_candles(g, "1h").ok
    gap = f.drop(f.index[10:13])
    rep = check_candles(gap, "1h")
    assert rep.ok and rep.gaps and rep.gaps[0][2] == 3
    with pytest.raises(DataIntegrityError):
        require_clean(gap, "1h")
    require_clean(gap, "1h", allow_gaps=True)
    b = f.copy(); b.iloc[7, 1] = b.iloc[7, 2] - 5            # high below low
    assert not check_candles(b, "1h").ok
    n = f.copy(); n.iloc[2, 3] = -1
    assert not check_candles(n, "1h").ok
    v = f.copy(); v.iloc[2, 4] = -1
    assert not check_candles(v, "1h").ok
    spike = f.copy(); spike.iloc[9, 3] = spike.iloc[8, 3] * 3
    assert not check_candles(spike, "1h").ok
    assert not check_candles(f.tz_localize(None), "1h").ok


def test_wrong_timeframe_detected():
    rep = check_candles(frame(), "4h")           # 1h data declared as 4h: intervals are shorter than the timeframe
    assert not rep.ok and any("shorter than timeframe" in i for i in rep.issues)


def test_clean_frame_dedups_and_sorts():
    f = frame(10)
    messy = pd.concat([f.iloc[5:], f.iloc[:6]])
    out = clean_frame(messy)
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates and len(out) == 10


def test_synthetic_data_is_clean_and_regimes_labelled():
    data, reg = generate(days=60, world="structured", seed=1)
    for df in data.values():
        assert check_candles(df, "1h").ok
    assert len(reg) == len(next(iter(data.values())))


def test_synthetic_is_deterministic_by_seed():
    a, _ = generate(days=30, seed=5)
    b, _ = generate(days=30, seed=5)
    c, _ = generate(days=30, seed=6)
    assert a["BTCUSDT"].equals(b["BTCUSDT"]) and not a["BTCUSDT"].equals(c["BTCUSDT"])


def test_csv_roundtrip(tmp_path):
    f = frame(20)
    save_csv(f, tmp_path / "x.csv")
    g = load_csv(tmp_path / "x.csv")
    assert np.allclose(f.to_numpy(), g.to_numpy()) and (f.index == g.index).all()


def test_candle_store_upsert_idempotent():
    st = CandleStore(sqlite3.connect(":memory:"))
    f = frame(10)
    st.upsert("BTCUSDT", "1h", f)
    st.upsert("BTCUSDT", "1h", f)                    # same rows again: no duplicates
    g = f.iloc[-2:].copy(); g["close"] = 999.0
    st.upsert("BTCUSDT", "1h", g)                    # revised candle overwrites
    out = st.load("BTCUSDT", "1h")
    assert len(out) == 10 and out["close"].iloc[-1] == 999.0 and out.index.equals(f.index)


def test_closed_only_drops_forming_candle():
    f = frame(5)
    now = f.index[-1] + pd.Timedelta(minutes=30)     # last candle still forming
    assert len(closed_only(f, "1h", now)) == 4


# ---- Binance public client with a fake HTTP session -----------------------
class Resp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        return self._p


def kline_rows(start_ms, n, step_ms=3_600_000):
    return [[start_ms + i * step_ms, "100", "101", "99", "100.5", "12.5", start_ms + (i + 1) * step_ms - 1] for i in range(n)]


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_client_parses_and_drops_unclosed():
    start = to_ms(pd.Timestamp("2024-01-01", tz="UTC"))
    now = (start + 2.5 * 3_600_000) / 1000
    c = BinancePublicClient(FakeSession([Resp(payload=kline_rows(start, 3))]), now_fn=lambda: now)
    df = c.klines("BTCUSDT", "1h")
    assert len(df) == 2 and df.index[0] == from_ms(start) and df["volume"].iloc[0] == 12.5


def test_client_falls_back_to_second_host_and_raises_when_all_fail():
    start = to_ms(pd.Timestamp("2024-01-01", tz="UTC"))
    s = FakeSession([ConnectionError("boom"), Resp(payload=kline_rows(start, 2))])
    c = BinancePublicClient(s, now_fn=lambda: start / 1000 + 10 * 3600)
    assert len(c.klines("BTCUSDT", "1h")) == 2
    assert s.calls[0][0].startswith("https://api.binance.com") and "data-api.binance.vision" in s.calls[1][0]
    bad = FakeSession([Resp(429, text="slow down"), Resp(500, text="x")])
    with pytest.raises(MarketDataError):
        BinancePublicClient(bad).klines("BTCUSDT", "1h")
    with pytest.raises(MarketDataError):
        BinancePublicClient(FakeSession([Resp(200, payload={"msg": "err"}), Resp(200, payload={"msg": "err"})])).klines("BTCUSDT", "1h")


def test_client_history_paginates_without_duplicates():
    start = to_ms(pd.Timestamp("2024-01-01", tz="UTC"))
    p1, p2 = kline_rows(start, 1000), kline_rows(start + 1000 * 3_600_000, 500)
    s = FakeSession([Resp(payload=p1), Resp(payload=p2), Resp(payload=[])])
    c = BinancePublicClient(s, now_fn=lambda: start / 1000 + 5000 * 3600)
    df = c.history("BTCUSDT", "1h", from_ms(start), from_ms(start + 1600 * 3_600_000), sleep_s=0)
    assert len(df) == 1500 and not df.index.has_duplicates and check_candles(df, "1h").ok
    assert s.calls[1][1]["startTime"] == start + 1000 * 3_600_000


def test_client_request_has_no_credentials_or_signature():
    start = to_ms(pd.Timestamp("2024-01-01", tz="UTC"))
    s = FakeSession([Resp(payload=kline_rows(start, 1))])
    BinancePublicClient(s, now_fn=lambda: start / 1000 + 99999).klines("BTCUSDT", "1h")
    url, params = s.calls[0]
    assert url.endswith("/api/v3/klines") and set(params) <= {"symbol", "interval", "limit", "startTime", "endTime"}
