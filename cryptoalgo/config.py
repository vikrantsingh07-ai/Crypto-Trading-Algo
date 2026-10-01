"""Typed configuration loaded from TOML. All risk limits are configurable here."""
from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.toml"


class ConfigError(ValueError):
    pass


@dataclass
class Account:
    initial_capital: float = 100_000.0
    currency: str = "INR"
    usdt_inr: float = 88.0


@dataclass
class Market:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframe: str = "1h"
    htf: str = "4h"
    warmup_bars: int = 300


@dataclass
class Costs:
    taker_fee_bps: float = 10.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0
    stop_extra_slippage_bps: float = 5.0
    funding_bps_per_8h: float = 1.0

    def scaled(self, k: float) -> "Costs":
        """Cost-stress copy: fees, spread, slippage multiplied by k."""
        return Costs(self.taker_fee_bps * k, self.half_spread_bps * k, self.slippage_bps * k,
                     self.stop_extra_slippage_bps * k, self.funding_bps_per_8h * k)


@dataclass
class Risk:
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 2
    max_total_open_risk_pct: float = 1.0
    max_leverage: float = 1.0
    max_position_pct: float = 60.0
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    flatten_on_drawdown_halt: bool = True
    drawdown_reset_mode: str = "cooldown"
    drawdown_cooldown_bars: int = 72
    hard_stop_drawdown_pct: float = 20.0   # from the ALL-TIME peak; only a manual reset clears it
    max_consecutive_losses: int = 3
    loss_cooldown_bars: int = 24
    symbol_cooldown_bars: int = 2
    max_trades_per_day: int = 6
    min_notional_inr: float = 500.0


@dataclass
class StrategyCfg:
    name: str = "regime_ensemble"
    use_htf_confirmation: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gates:
    min_profit_factor: float = 1.25
    min_trades: int = 100
    min_wf_positive_windows: float = 0.60
    max_drawdown_pct: float = 15.0
    mc_dd95_limit_pct: float = 25.0
    min_sharpe: float = 0.8
    min_sensitivity_positive: float = 0.70
    min_assets_profitable: int = 2
    cost_stress_multiplier: float = 2.0
    holdout_fraction: float = 0.30
    target_daily_pnl: float = 2000.0


@dataclass
class PaperCfg:
    db_path: str = "runtime/paper.db"
    poll_seconds: float = 30.0
    history_bars: int = 1500
    validation_report: str = "reports/validation.json"
    health_file: str = "runtime/health.json"
    max_data_age_bars: int = 3
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 300.0


@dataclass
class DashboardCfg:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class Config:
    mode: str = "paper"
    account: Account = field(default_factory=Account)
    market: Market = field(default_factory=Market)
    costs: Costs = field(default_factory=Costs)
    risk: Risk = field(default_factory=Risk)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    gates: Gates = field(default_factory=Gates)
    paper: PaperCfg = field(default_factory=PaperCfg)
    dashboard: DashboardCfg = field(default_factory=DashboardCfg)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def validate(self) -> None:
        r = self.risk
        if self.mode != "paper":
            raise ConfigError(f"mode must be 'paper' (got {self.mode!r}); real trading is not supported")
        if self.account.initial_capital <= 0:
            raise ConfigError("initial_capital must be > 0")
        if not 0 < r.risk_per_trade_pct <= 5:
            raise ConfigError("risk_per_trade_pct must be in (0, 5]")
        if r.max_open_positions < 1:
            raise ConfigError("max_open_positions must be >= 1")
        if not 0 < r.max_leverage <= 10:
            raise ConfigError("max_leverage must be in (0, 10]")
        if not 0 < r.max_daily_loss_pct <= 100 or not 0 < r.max_drawdown_pct <= 100:
            raise ConfigError("loss limits must be in (0, 100]")
        if r.hard_stop_drawdown_pct < r.max_drawdown_pct:
            raise ConfigError("hard_stop_drawdown_pct must be >= max_drawdown_pct")
        if r.drawdown_reset_mode not in ("cooldown", "manual"):
            raise ConfigError("drawdown_reset_mode must be 'cooldown' or 'manual'")
        if not self.market.symbols:
            raise ConfigError("at least one symbol required")
        for name in ("taker_fee_bps", "half_spread_bps", "slippage_bps", "stop_extra_slippage_bps"):
            if getattr(self.costs, name) < 0:
                raise ConfigError(f"{name} must be >= 0")


def _merge(dc, data: dict, path: str = "") -> None:
    known = {f.name: f for f in fields(dc)}
    for k, v in data.items():
        if k not in known:
            raise ConfigError(f"unknown config key: {path}{k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__"):
            if not isinstance(v, dict):
                raise ConfigError(f"[{path}{k}] must be a table")
            _merge(cur, v, f"{path}{k}.")
        else:
            setattr(dc, k, v)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        with open(p, "rb") as fh:
            _merge(cfg, tomllib.load(fh))
    elif path:
        raise ConfigError(f"config file not found: {p}")
    if overrides:
        _merge(cfg, overrides)
    cfg.validate()
    return cfg
