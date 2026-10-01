# Risk management & paper-only safety

## Paper-only guarantees

| Guarantee | How it is enforced |
|---|---|
| Config can't select live mode | `mode` must equal `"paper"`; `Config.validate()` raises otherwise |
| Process refuses to start near credentials | `safety.assert_paper_only` aborts if `BINANCE_API_KEY`, `*_SECRET`, `LIVE_TRADING`… are in the environment |
| No order-capable code | the only broker is `execution.PaperBroker` (pure in-memory); the only exchange client is a GET-only public-klines client |
| No signing / withdrawal / SDK | `tests/test_safety.py` scans the source for order endpoints, HMAC/signature code, `withdraw`, `api_key`, `ccxt`… and fails on any hit; exchange SDKs are rejected in dependency files |
| Paper trading needs evidence first | `paper.runner.load_validated_strategy` requires a passing `reports/validation.json` for the same symbols/timeframe |
| Everything is labelled | red PAPER TRADING / DEMO banner in dashboard, reports, logs, health file; unvalidated runs say UNVALIDATED |

**Adding real trading later** must be a deliberate project: a *separate* package with its own broker
implementing the same `fill_*` interface, separate config and credentials handling (least-privilege,
trade-only keys, no withdrawal), a staged rollout with tiny size, kill-switch, reconciliation against
exchange state, and a rewrite of the safety tests to cover that package. Nothing here pre-wires it.

## Risk controls (all configurable in `[risk]`)

* **Position sizing**: `qty = equity × risk_per_trade_pct ÷ (stop distance + round-trip cost allowance)`; then capped by
  `max_position_pct`, `max_leverage` (total notional ÷ equity) and the remaining `max_total_open_risk_pct`.
* **Stops**: every entry has a protective stop (ATR-based or band-based); gap-through-stop fills at the open (worse); if
  stop and target are both inside a bar the **stop wins**. Optional chandelier trailing stop ratchets in the trade's
  favour only; mean-reversion trades also have a take-profit at the mean and a time stop.
* **Daily loss limit** (`max_daily_loss_pct`, includes unrealised): no new entries until the next UTC day.
* **Drawdown limit** (`max_drawdown_pct`): no new entries, open positions flattened at the next open
  (`flatten_on_drawdown_halt`), pending entries cancelled. Reset:
  * `cooldown`: after `drawdown_cooldown_bars` bars the peak re-baselines to current equity and trading resumes (logged);
  * `manual`: stays halted until `RiskManager.manual_reset`.
* **Hard stop** (`hard_stop_drawdown_pct`, default 20 %): drawdown from the *all-time* peak. Cooldown cannot clear it,
  only a manual reset can. (This closes the loophole where repeated soft-halt resets would otherwise grant a fresh
  10 % allowance each time: found and fixed during development, see tests.)
* **Loss-streak cooldown**: after `max_consecutive_losses` consecutive losing trades, no entries for `loss_cooldown_bars`.
* **Anti-overtrading**: per-symbol re-entry cooldown, `max_trades_per_day`, `max_open_positions`.
* **Costs are pessimistic by default**: 10 bps taker fee + 2 bps half-spread + 5 bps slippage per fill, +5 bps on stops,
  funding 1 bp/8 h (longs pay, shorts receive). Validation also requires profit at **2× costs**.

Every halt, reset and cooldown is written to `risk_events` with a timestamp and shown on the dashboard.

## Fail-safe behaviour of the 24/7 runner

* Feed/network error → logged to `errors`, status `DEGRADED`, exponential back-off with jitter (cap 300 s), retry. After 20
  consecutive failures status `FAILED` (health check goes non-zero) but the loop keeps trying.
* A bar is **atomic**: orders, fees, positions, trades, equity and the engine-state snapshot commit in a single SQLite
  transaction. Any failure rolls back the in-memory engine to the last committed state and retries the bar.
* **No duplicate trades**: `client_order_id = sha1(strategy|symbol|bar|intent)`, UNIQUE in the DB, claimed once in the
  broker; already-processed bars are ignored; overlapping candles after reconnection are harmless; restarts resume
  from persisted state and process missed bars in order. (All covered by tests, including randomised crash/restart.)
* A bar is processed only once **all** symbols have it; the bot never guesses missing data.
* Candle gaps are logged (`errors`) and surfaced; stale data (> `max_data_age_bars`) degrades health.

## Known simplifications

* Bars are processed on close; stops are checked against bar high/low (no tick path). Real stop fills in fast markets can
  be worse than modelled; the 2×-cost stress partly covers this.
* Shorts are modelled as perpetual-style synthetic shorts at ≤ configured leverage; no liquidation engine or margin
  calls are simulated (leverage defaults to 1×).
* Fixed USDT/INR rate (`usdt_inr`) for INR reporting; spreads/slippage are fixed bps, not order-book based.
* Not modelled: Indian crypto tax (30 % flat on gains, no loss set-off, 1 % TDS on sells) - which would cut net daily
  profit by at least 30 % even if gross profit existed.
