# Strategy methodology

## Why not "just combine indicators"

Stacking indicators multiplies degrees of freedom; with enough combinations *something* looks profitable in-sample.
So the workflow is:

1. **Screen** (`scripts/run_research.py`, `research/study.py`): each condition (EMA trend/stack, MACD sign, RSI zones,
   Bollinger excursion, Donchian breakout, ADX trend/range, volume spike, volatility state) is tested in both directions
   (e.g. RSI<30 as a mean-reversion long *and* a momentum short) on **development data only**. Payoff = signed return over
   the next H bars from the next open, **minus round-trip costs**; samples are thinned to non-overlapping horizons so a
   t-test is approximately valid; pairs of conditions are tested too; all p-values go through **Benjamini-Hochberg FDR**.
   Only a condition that is positive *after costs* and survives FDR counts as evidence.
   Validated in tests: a planted +15 bps/bar effect is detected; the same pipeline on pure noise flags nothing.
2. **Candidates** use few parameters and simple, interpretable logic. Parameter grids are small and pre-declared.
3. **Selection** among candidates uses *development walk-forward out-of-sample* performance only. The locked hold-out is
   evaluated once for the single winner (and logged in `reports/holdout_ledger.json`, so repeated peeking is visible).

## Candidates

| Name | Idea | Entry | Exit |
|---|---|---|---|
| `trend_breakout` | trend following + breakout + volume | close breaks the prior-N-bar Donchian high (low) **and** is above (below) EMA200, ADX > floor, volume ratio > 1 | initial stop = k·ATR, chandelier trail, opposite Donchian(exit_n) break |
| `trend_pullback` | momentum continuation | EMA20>EMA50>EMA200, MACD hist>0, ADX>floor; RSI crosses back above a pullback level | stop k·ATR, trail, EMA20/50 cross |
| `mean_reversion` | fade extremes in ranges | ADX < ceiling **and** close beyond Bollinger band **and** RSI extreme | take-profit at the band mean, stop k·ATR, time stop |
| `regime_ensemble` | route by causal regime | trend regime (ADX≥25) → breakout (else pullback); range regime (ADX<20) → mean reversion; high-vol (ATR% > 2× median) → stand aside | each sub-strategy's own exits |

Optional multi-timeframe confirmation (`--htf` / `strategy.use_htf_confirmation`) requires the last *closed* higher-timeframe
EMA20/EMA50 trend to agree; it is aligned strictly by bar close time (tested for look-ahead).

## Backtest realism

Signal on bar close → fill at next open; costs on every fill; stop-wins-ties; gap fills at the open; trailing from closed
bars; funding; the same engine and risk rules as paper trading. Look-ahead is tested by truncating the future and checking
that every indicator/signal value is unchanged.

## Anti-overfitting

Locked hold-out · rolling walk-forward (parameters from the training slice only; **stay flat** if nothing is profitable
in-sample) · small pre-declared grids · neighbour-parameter sensitivity (±30 %) · 2× cost stress · Monte-Carlo (bootstrap
and shuffle) of trade P&L · multiple assets · FDR control in the research screen · ledger of hold-out evaluations.
**Nothing is tuned toward ₹2,000/day.** Feasibility is computed afterwards from what the validated strategy actually did.

## The ₹2,000/day question

₹2,000 on ₹1,00,000 is **2 % per day** ≈ 730 % per year simple (≈ 1.4 million % compounded). Professional systematic
funds target Sharpe ratios of roughly 1-2 and 10-30 % per year. For a given daily volatility σ the Sharpe needed is
`target/σ·√365`; `backtest/feasibility.py` reports it, the bootstrap probability that the true mean daily P&L ≥ target,
the scale-up needed and the drawdown that scale-up implies, and states "NOT ACHIEVABLE" when that breaches the risk
limits. Scaling risk to chase a target is *refused by construction*: it only reports that doing so would breach limits.
