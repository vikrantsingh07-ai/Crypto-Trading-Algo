"""Dashboard + monitoring (read-only view of the paper DB)."""
import json
import urllib.request

import pytest

from cryptoalgo.dashboard.server import serve
from cryptoalgo.dashboard.state import build_state
from cryptoalgo.data.synthetic import generate
from cryptoalgo.storage.db import connect
from paper_helpers import drive, make_trader


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    data, _ = generate(days=170, world="structured", seed=21)
    n = len(next(iter(data.values())))
    t, feed, cfg = make_trader(tmp, data, n - 1500)
    t.start()
    drive(t, feed, 1500, 10)
    return t, cfg


def test_state_has_every_required_panel(populated):
    t, cfg = populated
    conn = connect(cfg.paper.db_path, read_only=True)
    s = build_state(conn, 100_000.0)
    conn.close()
    for key in ("banner", "balance", "equity", "today_pnl", "total_pnl", "positions", "recent_trades", "metrics",
                "equity_curve", "daily_pnl", "monthly", "regimes", "bot", "risk", "errors", "risk_events", "validation"):
        assert key in s, key
    assert "PAPER TRADING" in s["banner"] and "UNVALIDATED" in s["validation"]["label"]
    m = s["metrics"]
    for k in ("win_rate_pct", "profit_factor", "max_drawdown_pct", "sharpe"):
        assert k in m
    assert s["equity"] == pytest.approx(t.engine.equity)
    assert s["total_pnl"] == pytest.approx(t.engine.equity - 100_000)
    assert len(s["equity_curve"]) > 100 and len(s["recent_trades"]) > 0
    assert s["bot"]["status"] == "RUNNING" and s["bot"]["mode"] == "PAPER"
    assert set(s["regimes"]) == set(cfg.market.symbols)


def test_http_server_serves_banner_state_and_health(populated):
    t, cfg = populated
    srv = serve(cfg.paper.db_path, "127.0.0.1", 0, 100_000.0, cfg.paper.health_file, background=True)
    port = srv.server_address[1]
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "PAPER TRADING / DEMO" in html
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state").read())
        assert st["equity"] > 0 and "banner" in st
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
        assert e.value.code == 404
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
            body = json.loads(r.read())
        except urllib.error.HTTPError as e2:      # stale heartbeat in a replayed sim is acceptable: must be a clean 503
            assert e2.code == 503
            body = json.loads(e2.read())
        assert "healthy" in body
    finally:
        srv.shutdown()


def test_dashboard_is_read_only(populated):
    t, cfg = populated
    conn = connect(cfg.paper.db_path, read_only=True)
    with pytest.raises(Exception):
        conn.execute("DELETE FROM trades")
    conn.close()


def test_dashboard_survives_empty_database(tmp_path):
    from cryptoalgo.storage.db import init_db
    conn = connect(tmp_path / "e.db")
    init_db(conn)
    conn.close()
    ro = connect(tmp_path / "e.db", read_only=True)
    s = build_state(ro, 100_000.0)
    assert s["equity"] == 100_000.0 and s["positions"] == [] and s["equity_curve"] == []
