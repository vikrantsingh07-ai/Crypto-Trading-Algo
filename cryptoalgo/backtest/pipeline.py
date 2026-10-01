"""End-to-end validation protocol (see docs/DESIGN.md sections 5 and 7).

 1. data integrity gate
 2. split: DEVELOPMENT (first 1-h of data) vs locked HOLD-OUT (last `holdout_fraction`)
 3. research study on DEVELOPMENT only (statistical screen of conditions)
 4. walk-forward on DEVELOPMENT for every candidate strategy -> pick ONE by OOS score (no hold-out peeking)
 5. choose parameters for the winner on DEVELOPMENT; sensitivity on DEVELOPMENT
 6. evaluate the winner ONCE on the HOLD-OUT (recorded in a ledger), at normal and 2x costs
 7. Monte-Carlo on hold-out trades, per-asset / per-regime / long-short breakdown
 8. feasibility of the INR target, then pass/fail gates
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from ..config import Config
from ..data.integrity import require_clean
from ..regime import label_blocks
from ..research.study import run_study
from ..strategies.library import REGISTRY, get_strategy
from . import report as rp
from .feasibility import assess_target
from .gates import evaluate_gates
from .metrics import by_label, compute_metrics
from .montecarlo import monte_carlo
from .runner import common_index, precompute, run_backtest
from .sensitivity import sensitivity
from .walkforward import objective, walk_forward


def _data_fingerprint(data: dict) -> str:
    h = hashlib.sha1()
    for s in sorted(data):
        df = data[s]
        h.update(f"{s}|{len(df)}|{df.index[0]}|{df.index[-1]}|{df['close'].iloc[-1]:.6f}".encode())
    return h.hexdigest()[:16]


def _pick_params(data, cfg, name, start, end, use_htf):
    """Best in-sample params from the pre-declared grid, on DEVELOPMENT data only."""
    tf = cfg.market.timeframe
    best, best_sc, best_m = None, float("-inf"), None
    for params in type(get_strategy(name)).param_grid():
        st = get_strategy(name, params, tf, cfg.market.htf, use_htf)
        r = run_backtest(data, cfg, st, tf, start, end, precompute(data, st))
        m = compute_metrics(r.trades, r.equity, cfg.account.initial_capital, tf)
        sc = objective(m, 30)
        if sc > best_sc:
            best, best_sc, best_m = params, sc, m
    return best, best_m


def run_validation(data: dict[str, pd.DataFrame], cfg: Config, strategies: list[str] | None = None,
                   use_htf: bool = False, out_dir: str = "reports", label: str = "validation",
                   wf_train_days: int | None = None, wf_test_days: int | None = None, do_research: bool = True,
                   ledger_path: str | None = None, verbose: bool = True) -> dict:
    t_start = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    tf = cfg.market.timeframe
    for s, df in data.items():
        require_clean(df, tf)
    g = cfg.gates
    idx = common_index(data)
    split = idx[int(len(idx) * (1 - g.holdout_fraction))]
    dev_days = (split - idx[0]).days
    wf_train = wf_train_days or max(120, int(dev_days * 0.45))
    wf_test = wf_test_days or max(30, int(dev_days * 0.10))
    rep: dict = {"label": label, "created_utc": pd.Timestamp.utcnow().isoformat(), "timeframe": tf,
                 "symbols": list(data), "data_fingerprint": _data_fingerprint(data), "use_htf": use_htf,
                 "development": [str(idx[0]), str(split)], "holdout": [str(split), str(idx[-1])],
                 "wf_train_days": wf_train, "wf_test_days": wf_test, "initial_capital": cfg.account.initial_capital}
    log(f"[1/8] data OK  dev={rep['development']}  holdout={rep['holdout']}")

    # 3. research (development only)
    dev_data = {s: df[df.index < split] for s, df in data.items()}
    if do_research:
        log("[2/8] research study (development data only)...")
        st = run_study(dev_data, horizon=12,
                       round_trip_cost_bps=2 * (cfg.costs.taker_fee_bps + cfg.costs.half_spread_bps + cfg.costs.slippage_bps))
        rep["research"] = {"n_hypotheses": int(len(st)), "n_significant_fdr10": int(st["significant"].sum()) if len(st) else 0,
                           "top": st.head(15).to_dict("records") if len(st) else []}

    # 4. walk-forward for every candidate on DEVELOPMENT; select ONE
    names = strategies or list(REGISTRY)
    log(f"[3/8] walk-forward on development for {names} (train={wf_train}d test={wf_test}d)...")
    cand = {}
    for name in names:
        wf = walk_forward(data, cfg, name, train_days=wf_train, test_days=wf_test, use_htf=use_htf, start=idx[0], end=split)
        oos = compute_metrics(wf.oos_trades if len(wf.oos_trades) else pd.DataFrame(columns=["net_pnl"]),
                              wf.oos_equity, cfg.account.initial_capital, tf) if len(wf.oos_equity) else None
        score = (oos["net_pnl"] / max(oos["max_drawdown_pct"], 1.0)) if oos and oos["trades"] >= 20 else float("-inf")
        cand[name] = {"wf": wf, "oos": oos, "score": score}
        log(f"      {name:16s} windows={wf.summary['windows']} traded={wf.summary['traded_windows']} "
            f"positive={wf.summary['positive_windows']} OOS net={wf.summary['oos_net_pnl']:,.0f} "
            f"trades={oos['trades'] if oos else 0}")
    rep["candidates"] = {n: {"wf_summary": c["wf"].summary, "wf_oos_metrics": c["oos"], "score": c["score"],
                             "windows": [{"train": [str(w.train[0]), str(w.train[1])], "test": [str(w.test[0]), str(w.test[1])],
                                          "params": w.params, "test_net_pnl": w.test_metrics.get("net_pnl"),
                                          "test_trades": w.test_metrics.get("trades")} for w in c["wf"].windows]}
                         for n, c in cand.items()}
    chosen = max(cand, key=lambda n: cand[n]["score"]) if any(c["score"] > float("-inf") for c in cand.values()) \
        else max(cand, key=lambda n: cand[n]["wf"].summary["oos_net_pnl"])
    rep["chosen_strategy"] = chosen
    rep["selection_note"] = (f"{len(names)} candidates were compared on DEVELOPMENT walk-forward OOS only; "
                             "the hold-out was evaluated for the single winner.")
    wf_c = cand[chosen]["wf"]
    wf_oos = cand[chosen]["oos"] or compute_metrics(pd.DataFrame(columns=["net_pnl"]), pd.Series(dtype=float),
                                                    cfg.account.initial_capital, tf)
    log(f"[4/8] chosen candidate: {chosen}")

    # 5. final params on dev + sensitivity on dev
    params, dev_m = _pick_params(data, cfg, chosen, idx[0], split, use_htf)
    rep["no_in_sample_edge"] = params is None
    if params is None:
        params = {}
    rep["params"] = params
    rep["dev_metrics"] = dev_m
    log(f"[5/8] params={params}; sensitivity on development...")
    sens = sensitivity(data, cfg, chosen, params, idx[0], split, use_htf)
    rep["sensitivity"] = sens

    # 6. HOLD-OUT, once
    ledger = Path(ledger_path or Path(out_dir) / "holdout_ledger.json")
    led = json.loads(ledger.read_text()) if ledger.exists() else []
    led.append({"when": rep["created_utc"], "strategy": chosen, "params": params, "data": rep["data_fingerprint"]})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(led, indent=2))
    rep["holdout_evaluations_on_this_data"] = sum(1 for e in led if e["data"] == rep["data_fingerprint"])
    st = get_strategy(chosen, params, tf, cfg.market.htf, use_htf)
    rows = precompute(data, st)
    log(f"[6/8] hold-out evaluation (evaluation #{rep['holdout_evaluations_on_this_data']} on this dataset)...")
    ho = run_backtest(data, cfg, st, tf, split, None, rows, keep_decisions=False)
    ho_m = compute_metrics(ho.trades, ho.equity, cfg.account.initial_capital, tf)
    rep["holdout_metrics"] = ho_m
    scfg = cfg.copy()
    scfg.costs = cfg.costs.scaled(g.cost_stress_multiplier)
    hs = run_backtest(data, scfg, st, tf, split, None, rows)
    hs_m = compute_metrics(hs.trades, hs.equity, cfg.account.initial_capital, tf)
    rep["holdout_stress_metrics"] = {k: hs_m[k] for k in ("net_pnl", "profit_factor", "max_drawdown_pct", "trades", "sharpe")}
    regimes = {}
    for s, df in data.items():
        lab = label_blocks(df[df.index >= split])
        regimes[s] = by_label(ho.trades[ho.trades["symbol"] == s], lab)
    rep["holdout_by_regime"] = regimes
    rep["holdout_risk_events"] = {k: sum(1 for e in ho.risk_events if e["kind"] == k) for k in {e["kind"] for e in ho.risk_events}}

    # 7. Monte Carlo
    log("[7/8] Monte Carlo + feasibility...")
    mc = monte_carlo(ho.trades["net_pnl"].to_numpy(), cfg.account.initial_capital, dd_limit_pct=g.mc_dd95_limit_pct)
    mc_shuffle = monte_carlo(ho.trades["net_pnl"].to_numpy(), cfg.account.initial_capital, mode="shuffle",
                             dd_limit_pct=g.mc_dd95_limit_pct)
    rep["monte_carlo"], rep["monte_carlo_shuffle"] = mc, mc_shuffle
    if len(wf_c.oos_trades):
        rep["monte_carlo_wf_oos"] = monte_carlo(wf_c.oos_trades["net_pnl"].to_numpy(), cfg.account.initial_capital,
                                                dd_limit_pct=g.mc_dd95_limit_pct)
    rep["feasibility_holdout"] = assess_target(ho.equity, cfg.account.initial_capital, cfg, ho_m["max_drawdown_pct"])
    if len(wf_c.oos_equity):
        rep["feasibility_wf_oos"] = assess_target(wf_c.oos_equity, cfg.account.initial_capital, cfg,
                                                  wf_oos["max_drawdown_pct"])

    # 8. gates
    per_asset = {s: v["net_pnl"] for s, v in ho_m["by_symbol"].items()}
    ok, rows_g = evaluate_gates(g, ho_m, wf_c.summary, wf_oos, mc, sens, hs_m, per_asset)
    rep["gates"], rep["passed"] = rows_g, bool(ok) and not rep["no_in_sample_edge"]
    rep["equity_holdout"] = [[str(t), float(v)] for t, v in ho.equity.iloc[::max(1, len(ho.equity) // 400)].items()]
    rep["runtime_seconds"] = round(time.time() - t_start, 1)
    log(f"[8/8] gates: {'PASSED' if rep['passed'] else 'FAILED'}  ({sum(r['pass'] for r in rows_g)}/{len(rows_g)} pass)")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rp.dump_json(rep, Path(out_dir) / f"{label}.json")
    (Path(out_dir) / f"{label}.md").write_text(render_markdown(rep))
    (Path(out_dir) / f"{label}.html").write_text(render_html(rep, ho.equity))
    rep["_holdout_equity"], rep["_holdout_trades"] = ho.equity, ho.trades
    return rep


def render_markdown(rep: dict) -> str:
    m = rep["holdout_metrics"]
    f = rep["feasibility_holdout"]
    L = [f"# Validation report: {rep['label']}", "", f"> **{rp.PAPER_LABEL}**", "",
         f"- Data: {', '.join(rep['symbols'])} @ {rep['timeframe']}, fingerprint `{rep['data_fingerprint']}`",
         f"- Development: {rep['development'][0][:10]} -> {rep['development'][1][:10]}; "
         f"locked hold-out: {rep['holdout'][0][:10]} -> {rep['holdout'][1][:10]}",
         f"- Candidates compared (development walk-forward only): {', '.join(rep['candidates'])}",
         f"- Chosen: **{rep['chosen_strategy']}** with params `{rep['params']}` "
         f"(no in-sample edge found: {rep['no_in_sample_edge']})",
         f"- Hold-out evaluated {rep['holdout_evaluations_on_this_data']}x on this dataset",
         f"- **Overall: {'PASSED' if rep['passed'] else 'FAILED'} all robustness gates**", "",
         "## Gates", "", "| Gate | Value | Threshold | Result |", "|---|---|---|---|"]
    L += [f"| {r['gate']} | {r['value']} | {r['threshold']} | {'PASS' if r['pass'] else 'FAIL'} |" for r in rep["gates"]]
    L += ["", "## Hold-out performance (never used for selection or tuning)", "", rp.metrics_md(m), "",
          "### Monthly (hold-out)", "", rp.monthly_md(m), "", "## ₹ target feasibility (hold-out)", "",
          f"- Target: {rp.inr(f['target_daily_pnl'])}/day = {f['target_pct_of_capital']:.2f}% of capital per day "
          f"(~{f['required_annual_return_simple_pct']:.0f}% per year, simple)"]
    if "mean_daily_pnl" in f:
        L += [f"- Achieved mean daily P&L: {rp.inr(f['mean_daily_pnl'])} (95% CI {rp.inr(f['mean_ci95'][0])} .. {rp.inr(f['mean_ci95'][1])})",
              f"- P(true mean >= target) by bootstrap: {f['prob_mean_ge_target']:.1%}; days >= target: {f['fraction_days_ge_target']:.1%}",
              f"- Sharpe needed for the target at this volatility: {f['sharpe_needed_for_target']:.1f} vs achieved {f['sharpe_achieved']:.2f}"]
        if f.get("scale_needed"):
            L.append(f"- Scale needed: {f['scale_needed']:.1f}x -> risk/trade {f['risk_per_trade_needed_pct']:.1f}%, "
                     f"implied drawdown {f['implied_max_drawdown_pct']:.0f}%")
    L += [f"- **Verdict: {f['verdict']}**", f"- {f['after_india_tax_note']}", ""]
    mc = rep["monte_carlo"]
    if not mc.get("insufficient"):
        L += ["## Monte Carlo (hold-out trades, bootstrap)", "",
              f"- final equity p5/p50/p95: {rp.inr(mc['final_equity']['p5'])} / {rp.inr(mc['final_equity']['p50'])} / {rp.inr(mc['final_equity']['p95'])}",
              f"- max drawdown p50/p95/p99: {mc['max_dd_pct']['p50']:.1f}% / {mc['max_dd_pct']['p95']:.1f}% / {mc['max_dd_pct']['p99']:.1f}%",
              f"- P(loss) {mc['prob_loss']:.1%}, P(DD > {mc['dd_limit_pct']:.0f}%) {mc['prob_dd_exceeds_limit']:.1%}", ""]
    s = rep["sensitivity"]
    L += ["## Parameter sensitivity (development)", "",
          f"{s['fraction_profitable']:.0%} of {s['n_neighbours']} neighbouring parameter sets profitable; median PF {s['median_pf']:.2f}", ""]
    L += ["## Walk-forward, per candidate (development)", "", "| Candidate | windows | traded | positive | OOS net | OOS trades |", "|---|---|---|---|---|---|"]
    for n, c in rep["candidates"].items():
        ws = c["wf_summary"]
        L.append(f"| {n} | {ws['windows']} | {ws['traded_windows']} | {ws['positive_windows']} | "
                 f"{rp.inr(ws['oos_net_pnl'])} | {(c['wf_oos_metrics'] or {}).get('trades', 0)} |")
    if "research" in rep:
        r = rep["research"]
        L += ["", "## Research screen (development, BH-FDR 10%)", "",
              f"{r['n_significant_fdr10']} of {r['n_hypotheses']} hypotheses significant after false-discovery control.", ""]
        if r["top"]:
            L += ["| Condition | Dir | n | mean net bps | t | q |", "|---|---|---|---|---|---|"]
            L += [f"| {t['condition']} | {t['direction']} | {t['n_events']} | {t['mean_net_bps']:.1f} | {t['t_stat']:.2f} | {t['fdr_q']:.3f} |"
                  for t in r["top"][:10]]
    return "\n".join(L) + "\n"


def render_html(rep: dict, equity: pd.Series) -> str:
    md = render_markdown(rep)
    import html as h
    body = (f"<h1>Validation report: {h.escape(rep['label'])}</h1>"
            f"<p class='{'pass' if rep['passed'] else 'fail'}'><b>{'PASSED' if rep['passed'] else 'FAILED'} robustness gates</b></p>"
            + rp.equity_svg(equity, rep["initial_capital"], title="Hold-out equity (virtual INR)")
            + f"<pre style='white-space:pre-wrap'>{h.escape(md)}</pre>")
    return rp.html_page("Validation report", body)
