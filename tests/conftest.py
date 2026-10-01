import pytest

from cryptoalgo.config import Config, Costs, load_config
from cryptoalgo.data.synthetic import generate


@pytest.fixture(scope="session")
def base_cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def synth():
    """~400 days of synthetic 1h BTC/ETH (testing only)."""
    data, regimes = generate(days=400, world="structured", seed=11)
    return data, regimes


@pytest.fixture
def exact_cfg() -> Config:
    """Zero costs, fx=1, 1% risk: makes hand arithmetic exact in engine tests."""
    c = load_config()
    c.costs = Costs(0, 0, 0, 0, 0)
    c.account.usdt_inr = 1.0
    c.risk.risk_per_trade_pct = 1.0
    c.risk.max_leverage = 10.0
    c.risk.max_position_pct = 1000.0
    c.risk.max_total_open_risk_pct = 5.0
    c.risk.symbol_cooldown_bars = 0
    c.risk.max_trades_per_day = 1000
    c.risk.max_daily_loss_pct = 100
    c.risk.max_drawdown_pct = 100
    c.risk.hard_stop_drawdown_pct = 100
    c.market.symbols = ["BTCUSDT"]
    return c


H = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % (24 * H))  # a UTC midnight


def bar(o, h=None, l=None, c=None, v=1000.0):
    h = max(o, c if c is not None else o) if h is None else h
    l = min(o, c if c is not None else o) if l is None else l
    return (o, h, l, o if c is None else c, v)


def sig(side=1, stop=10.0, tp=None, trail=None, hold=None, regime="t", **kw):
    d = {"signal": side, "stop_dist": stop, "tp_dist": tp if tp is not None else float("nan"),
         "trail_dist": trail if trail is not None else float("nan"),
         "exit_long": False, "exit_short": False,
         "max_hold": hold if hold is not None else float("nan"), "regime": regime}
    d.update(kw)
    return d
