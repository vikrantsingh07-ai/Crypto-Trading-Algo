# Design: Paper-Trading Crypto Algo System

> **PAPER / DEMO ONLY.** No real orders, no exchange API keys, no withdrawal
> permission. Virtual capital: ₹1,00,000.
> The ₹2,000/day figure is a *hypothesis to be tested*, never a promise.

## 1. Starting point (inspection)

The repository was empty (no commits, no files). Everything here is new.
The sandbox this was built in blocks every crypto data host, so real
exchange history could not be downloaded during development. Consequences:

* Real-data loaders (Binance public REST, CSV import) are implemented and unit
  tested against mocked HTTP, but have **not** been run against the live API here.
* A synthetic market generator (regime-switching, stochastic volatility,
  correlated BTC/ETH) is included to exercise the engine, the validation
  pipeline and its false-positive control. **Synthetic results say nothing about
  whether the strategy makes money on real markets.**
* The real verdict requires running `scripts/fetch_data.py` then
  `scripts/run_validation.py` on a machine with internet access.

## 2. Architecture

```
            ┌────────────┐   candles   ┌──────────────┐  signals  ┌────────────┐
 Binance ──▶│ data/feeds │────────────▶│ strategies/  │──────────▶│            │
 (public    │  + store   │             │ (causal,     │           │  engine/   │
 REST only) └────────────┘             │  vectorised) │           │  TradingEngine
 CSV/synthetic ─────────────────────────┘              │           │  (one core)│
                                                       │  risk/ ──▶│            │
                                                       │  RiskManager           │
                                                       │  execution/ PaperBroker│
                                                       └─────┬──────┴────┬──────┘
                                       same core code        │           │
                            ┌────────────────────────────────┘           └────────────┐
                            ▼                                                         ▼
                    backtest/ (offline replay, metrics,                paper/ (24/7 runner: poll closed
                    walk-forward, Monte Carlo, sensitivity,            candles, idempotent bar processing,
                    feasibility, validation gates)                     reconnect w/ backoff, health, DB)
                                                                                      │
                                                          storage/ SQLite ◀───────────┤
                                                          dashboard/ (read-only) ◀────┘
```

Key decision: **one trading core** (`engine.TradingEngine.process_bar`) is used
by both the backtester and the paper runner. A consistency test asserts that
replaying history through the paper path reproduces the backtest trade-for-trade,
so "it worked in backtest" and "it ran in paper" mean the same code ran.

| Module | Responsibility |
|---|---|
| `config.py`, `config/default.toml` | typed, validated config; everything risk-related is configurable |
| `safety.py` | paper-only guard; refuses to start in any other mode |
| `data/` | candle model, Binance public REST client, CSV loader, synthetic generator, SQLite candle store, integrity checks |
| `indicators.py` | causal EMA/SMA/RSI/MACD/ATR/ADX/Bollinger/Donchian/volume features, multi-timeframe alignment |
| `strategies/` | building blocks and 4 candidate strategies |
| `regime.py` | trend / range / high-vol / low-vol classification (causal) |
| `risk.py` | position sizing, per-trade risk, caps, daily-loss and drawdown halts, loss-streak cooldown |
| `execution.py` | fee/slippage/spread model; `PaperBroker` with idempotent client order IDs |
| `portfolio.py` | cash, positions, equity, realised/unrealised P&L |
| `engine.py` | the shared bar-by-bar core |
| `backtest/` | runner, metrics, walk-forward, Monte Carlo, sensitivity, feasibility, gates |
| `research/` | statistical study of building blocks and combinations (FDR-controlled) |
| `paper/` | live runner, health, reconnection, state recovery |
| `storage/` | SQLite schema + repository (signals, orders, trades, positions, fees, decisions, errors, risk events) |
| `dashboard/` | stdlib HTTP server + single-page UI, labelled PAPER TRADING / DEMO |

## 3. Dependencies

`numpy`, `pandas` (computation), `pytest` (tests). Optional: `requests` (live
Binance public data). Everything else is the Python 3.11 standard library
(sqlite3, http.server, tomllib, logging, threading). No exchange SDK, no
signing library: nothing capable of placing an order is installed.

## 4. Trading methodology

Crypto trades 24/7 with fat tails and volatility clustering; markets alternate
between trending and ranging regimes. Plan:

1. **Study building blocks first** (`research/`): for each condition (EMA trend,
   Donchian breakout, ADX filter, RSI extremes, MACD sign, volume spike,
   Bollinger excursion, volatility state ...) measure forward returns net of
   costs with HAC (Newey-West) t-statistics, then control the false-discovery
   rate (Benjamini-Hochberg) across *all* tested conditions/combinations. Only
   survivors may be combined. Filter combos are accepted only if they improve
   results beyond what adding a parameter would by chance (ablation, trade-level
   bootstrap).
2. **Four candidate strategies** with few parameters each (anti-overfitting):
   * `trend_breakout`: Donchian breakout + EMA200 side + ADX + volume, ATR stop, chandelier trail.
   * `trend_pullback`: EMA stack + MACD sign, entry on RSI pullback recovery.
   * `mean_reversion`: Bollinger + RSI extremes only when ADX is low; target = mean.
   * `regime_ensemble`: routes to trend or reversion by regime, stands aside in extreme volatility.
3. Optional higher-timeframe confirmation (strictly last *closed* HTF bar).
4. Long and short. Shorts are modelled as perpetual-style synthetic shorts at
   leverage ≤ configured cap (default 1×), with funding cost.

## 5. Backtesting methodology

* Signal computed on bar *t* close; order fills at bar *t+1* open (never same-bar).
* Fills pay taker fee + half-spread + slippage; stop fills pay extra slippage;
  gaps fill at the open, not at the stop price.
* If stop and target both lie inside one bar's range, the **stop wins** (pessimistic).
* Trailing stops update from closed bars only.
* Risk rules are applied identically in backtest and paper.
* Causality test: indicators/signals at *t* are identical when future data is removed.
* **Splits**: the last 30 % of history is a locked hold-out evaluated once.
  Walk-forward (rolling train → test, parameters chosen from a small pre-declared
  grid on the training window only) runs on the rest. Regime tags are
  evaluated separately (bull / bear / sideways / high-vol / low-vol).
* **Robustness**: parameter sensitivity (±neighbours of the chosen params),
  cost stress (2× fees/slippage), Monte Carlo (trade-resampling and
  trade-order shuffling for drawdown and end-equity distributions), multiple
  assets and timeframes.

## 6. Risk parameters (all configurable, defaults)

| Parameter | Default |
|---|---|
| risk per trade | 0.5 % of equity |
| max open positions | 2 |
| max total open risk | 1.0 % of equity |
| max leverage (total notional / equity) | 1.0× |
| max daily loss | 2 % → no new entries until next UTC day |
| max drawdown | 10 % → no new entries; reset after cooldown or manual |
| consecutive-loss cooldown | 3 losses → pause 24 bars |
| fees / spread / slippage | 10 bps taker / 2 bps half-spread / 5 bps |

## 7. Success / failure criteria (declared before testing)

Validation gates (`config/default.toml [gates]`), all must pass on out-of-sample
(hold-out + walk-forward) data **and** survive 2× costs:

1. Profit factor ≥ 1.25 and net profit > 0.
2. ≥ 100 trades (statistical weight).
3. Walk-forward: ≥ 60 % of test windows profitable.
4. Max drawdown ≤ 15 %.
5. Monte-Carlo 95th-percentile drawdown ≤ 25 %; 5th-percentile final equity > initial.
6. Parameter sensitivity: ≥ 70 % of neighbour parameter sets profitable.
7. Sharpe (daily, annualised) ≥ 0.8.
8. Positive result on more than one asset.

**The ₹2,000/day target** is evaluated separately and objectively
(`backtest/feasibility.py`): realised mean daily P&L with bootstrap confidence
interval, probability that mean ≥ ₹2,000, and the leverage/risk scale needed to
reach it together with the drawdown *that scale implies*. If the implied
drawdown breaches the risk limit, the report states "not achievable within
constraints". No strategy tuning is done to reach it.

Paper trading is refused unless the validation report says `passed: true`
(an explicit `--allow-unvalidated` flag exists for plumbing demos and is
labelled in the dashboard).

## 8. Failure & recovery design

* Bars are processed **once**: per-symbol `last_processed_bar` and deterministic
  `client_order_id = sha1(strategy|symbol|bar_open|intent)` with a UNIQUE DB
  constraint. Re-fetching overlapping candles after a reconnect is harmless.
* Each bar is committed in a single SQLite transaction (orders, trades, positions,
  portfolio snapshot, events). A crash mid-bar rolls back and the bar is redone.
* Data fetch errors → exponential back-off with jitter, health state `DEGRADED`,
  never a fabricated order. Gaps are back-filled before processing.
* Health file/endpoint reports last-bar age, last error, halted state.
