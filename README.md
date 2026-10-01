# Crypto Paper-Trading Algo System

> **PAPER TRADING / DEMO ONLY.** Virtual capital ₹1,00,000. No real orders, no exchange API keys,
> no withdrawal permissions: the code to do so does not exist (and a test enforces that).
> The ₹2,000/day figure is a **hypothesis the system tests**, not a promise. Read
> [`docs/RESULTS.md`](docs/RESULTS.md) for what the evidence actually says.

A research-to-paper-trading pipeline for 24/7 crypto markets:

```
data  ->  research screen  ->  backtest  ->  walk-forward / hold-out / Monte Carlo  ->  gates
                                                                                          |
          dashboard  <-  SQLite  <-  24/7 paper trader (same engine as the backtest)  <---+
```

## Honest status

* The system is complete and tested (see [Tests](#tests)).
* The development sandbox could not reach any crypto exchange, so **no real historical or live market
  data has been processed yet.** All backtests in `reports/` use *clearly labelled synthetic data* and
  demonstrate that the machinery works and that the validator rejects strategies without edge and accepts
  ones with planted edge. **They say nothing about real-market profitability.**
* Paper trading is gated: it refuses to start until `scripts/run_validation.py` has passed every gate on
  data you supply. Run that on real data first.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # numpy, pandas, requests, pytest
pytest                                  # full suite (add -m "not slow" to skip the soak test)
```

### 1. Get real data (needs internet access to api.binance.com or data-api.binance.vision)

```bash
python scripts/fetch_data.py --symbols BTCUSDT ETHUSDT --timeframe 4h --start 2019-01-01 --out data/
```
Only unauthenticated public market-data endpoints are used.
(No exchange access? You can supply your own CSVs named `data/BTCUSDT_4h.csv` with columns
`open_time,open,high,low,close,volume`.)

### 2. Research, backtest, validate

```bash
python scripts/run_research.py   --config config/4h.toml --data data/            # which conditions have a real edge?
python scripts/run_backtest.py   --config config/4h.toml --data data/ --strategy trend_breakout
python scripts/run_validation.py --config config/4h.toml --data data/            # writes reports/validation.{json,md,html}
```
`validation.json` has `"passed": true` only if **all 12 gates** pass (see [`docs/DESIGN.md`](docs/DESIGN.md) §7).
Whatever the outcome, the report states plainly whether ₹2,000/day was achievable under your risk limits.

### 3. Paper trade 24/7

```bash
python scripts/run_paper.py --config config/4h.toml --dashboard      # http://127.0.0.1:8080
```
* Needs a passing `reports/validation.json` that matches the configured symbols/timeframe.
  `--allow-unvalidated` runs a plumbing demo and labels every screen **UNVALIDATED**.
* Offline demo with a simulated feed:
  `python scripts/run_paper.py --simulate synthetic:structured:7:600 --sim-bars 3000 --allow-unvalidated --dashboard`
* VPS: `deploy/paper-trader.service` (+ health timer), or `docker build -t paper-trader .`.
  Dashboard has **no authentication**: bind to localhost / put it behind a reverse proxy with auth.
* Watchdog: `python scripts/healthcheck.py` exits non-zero if the heartbeat is stale.

## Configuration

Everything lives in `config/default.toml` (override file: `--config`). Unknown keys are rejected. Highlights:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `risk` | `risk_per_trade_pct` | 0.5 | % of equity risked to the stop (sizing includes round-trip cost allowance) |
| | `max_open_positions` / `max_total_open_risk_pct` | 2 / 1.0 | concurrency caps |
| | `max_leverage` / `max_position_pct` | 1.0 / 60 | notional caps |
| | `max_daily_loss_pct` | 2.0 | no new entries until next UTC day |
| | `max_drawdown_pct`, `drawdown_reset_mode`, `drawdown_cooldown_bars` | 10, cooldown, 72 | soft halt, flatten, reset rule |
| | `hard_stop_drawdown_pct` | 20 | kill-switch vs ALL-TIME peak; manual reset only |
| | `max_consecutive_losses`, `loss_cooldown_bars` | 3, 24 | pause after a losing streak |
| | `symbol_cooldown_bars`, `max_trades_per_day` | 2, 6 | anti-overtrading |
| `costs` | `taker_fee_bps`, `half_spread_bps`, `slippage_bps`, `stop_extra_slippage_bps`, `funding_bps_per_8h` | 10, 2, 5, 5, 1 | conservative: all fills are taker fills |
| `gates` | thresholds | see file | robustness gates declared before testing |
| `paper` | `poll_seconds`, `backoff_*`, `max_data_age_bars` | 30, 2..300 s, 3 | polling/reconnect/health |

Details: [`docs/RISK_AND_SAFETY.md`](docs/RISK_AND_SAFETY.md). Strategy description: [`docs/STRATEGY.md`](docs/STRATEGY.md).

## Layout

```
cryptoalgo/
  config.py safety.py indicators.py regime.py risk.py execution.py portfolio.py engine.py events.py monitoring.py
  data/        models, binance (public klines), loader (CSV), store (SQLite), integrity, synthetic (TEST data)
  strategies/  base, library (4 strategies)
  backtest/    runner, metrics, walkforward, montecarlo, sensitivity, feasibility, gates, pipeline, report
  research/    study (statistical screen, BH-FDR)
  paper/       feeds (live public + simulated w/ fault injection), runner (24/7 loop, recovery)
  storage/     db (SQLite schema + atomic per-bar sink)
  dashboard/   state, server, index.html
scripts/       fetch_data, run_research, run_backtest, run_validation, run_paper, run_dashboard, healthcheck, ...
tests/         unit / strategy / backtest / data / risk / execution / failure / duplicate / stability
```

## Tests

`pytest` runs ~150 tests across: unit & accounting (hand-computed fills), strategy contracts, **look-ahead
(causality) tests for every indicator and strategy**, backtest/metrics (formulas verified independently),
data integrity, risk management (every limit and reset), paper execution, **failure/reconnection**,
**duplicate-order prevention**, crash-restart recovery, **paper-vs-backtest equivalence**, dashboard, a
source-scan proving no real-trading code exists, and an accelerated long-running stability test with
random faults and restarts. See [`docs/RESULTS.md`](docs/RESULTS.md) for the recorded run.
