"""
Implied vs Realized Vol & Delta-Hedged P&L
============================================
Runs the volatility-risk-premium study and the short-volatility hedged-option
backtest end to end.

    python run_vrp_backtest.py                       # SPY, VIX, 2010 -> today (needs internet)
    python run_vrp_backtest.py --start 2015-01-01 --tc-bps 1
    python run_vrp_backtest.py --hedge-vol daily
    python run_vrp_backtest.py --synthetic           # offline pipeline check, NOT real results

Outputs
-------
  - Console summary (VRP statistics with Newey-West t-stats, forecast regressions,
    hedged-P&L statistics and the worst trades)
  - plots/iv_vs_rv.png, plots/vrp.png, plots/hedged_pnl.png
"""

import argparse
import os

import numpy as np
import pandas as pd

from history import fetch_history, synthetic_history
from hedging import simulate_delta_hedged_short_call, summarize_hedged_pnl
from vol_analysis import build_vrp_frame, mincer_zarnowitz, vrp_summary
from visualise import plot_hedged_pnl, plot_iv_vs_rv, plot_vrp

PLOT_DIR = "plots"


def section(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon", type=int, default=21, help="option life / realized-vol window, trading days")
    ap.add_argument("--moneyness", type=float, default=1.0, help="strike / spot at entry")
    ap.add_argument("--hedge-vol", choices=["entry", "daily"], default="entry")
    ap.add_argument("--tc-bps", type=float, default=0.0, help="cost on hedge trades, bps of notional")
    ap.add_argument("--synthetic", action="store_true", help="use simulated data (offline check only)")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using data/history.csv")
    args = ap.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    h = args.horizon

    if args.synthetic:
        data = synthetic_history()
        source = "SYNTHETIC DATA: pipeline check only, not evidence about real markets"
    else:
        data = fetch_history(args.start, args.end, refresh=args.refresh)
        source = "SPY (dividend-adjusted), ^VIX, ^IRX via Yahoo Finance"

    section("Data")
    print(f"  {source}")
    print(f"  {data.index[0].date()} -> {data.index[-1].date()}   ({len(data):,} trading days)")

    spot = data["spy"]
    iv = data["vix"] / 100.0

    # ── Part 1: implied vs realized ──────────────────────────────────────────
    frame = build_vrp_frame(spot, iv, window=h)
    s = vrp_summary(frame, lags=h)

    section(f"Variance risk premium  (IV vs next-{h}-day realized vol)")
    v, q = s["vrp_vol"], s["vrp_var"]
    print(f"  observations              {s['n_obs']:,}  (overlapping; Newey-West lags = {h})")
    print(f"  mean implied vol          {s['mean_iv']*100:6.2f}%")
    print(f"  mean forward realized     {s['mean_rv_forward']*100:6.2f}%")
    print(f"  mean VRP (vol points)     {v['mean']*100:+6.2f}   HAC se {v['se']*100:.2f}   t = {v['t_stat']:+.2f}")
    print(f"  mean VRP (variance pts)   {q['mean']*1e4:+6.1f}   HAC se {q['se']*1e4:.1f}   t = {q['t_stat']:+.2f}   (x 1e-4)")
    print(f"  days with IV > RV         {s['pct_days_vrp_positive']*100:6.1f}%")
    pc = s["vrp_vol_percentiles"]
    print("  VRP percentiles (vol pts) " + "  ".join(f"p{k}={pc[k]*100:+.1f}" for k in (1, 5, 50, 95, 99)))

    mz = mincer_zarnowitz(frame, lags=h)
    section("Does implied vol forecast realized vol?  (HAC t-stats)")
    print("  rv_forward = a + b*iv")
    print(mz["iv_only"].round(4).to_string())
    print("\n  rv_forward = a + b*iv + c*rv_trailing")
    print(mz["iv_and_trailing"].round(4).to_string())
    print("\n  Unbiased forecast would give a = 0, b = 1.")

    # ── Part 2: delta-hedged short call ──────────────────────────────────────
    trades = simulate_delta_hedged_short_call(
        spot, iv, r=data["rate"].values, horizon=h, step=1,
        moneyness=args.moneyness, hedge_vol=args.hedge_vol, tc_bps=args.tc_bps,
    )
    ps = summarize_hedged_pnl(trades, horizon=h)

    section(f"Short {h}d call, daily delta-hedged  (moneyness {args.moneyness:.2f}, "
            f"hedge vol: {args.hedge_vol}, costs: {args.tc_bps:g} bp)")
    print(f"  non-overlapping trades    {ps['n_trades_nonoverlapping']}")
    print(f"  mean P&L / trade          {ps['mean_pct']*100:+.3f}% of spot   (std {ps['std_pct']*100:.3f}%)")
    print(f"  t-stat, non-overlapping   {ps['t_stat_nonoverlapping']:+.2f}")
    print(f"  t-stat, all starts (NW)   {ps['t_stat_overlapping_nw']:+.2f}")
    print(f"  annualised Sharpe         {ps['sharpe_annualised']:+.2f}")
    print(f"  hit rate                  {ps['hit_rate']*100:.1f}%")
    print(f"  skew of trade P&L         {ps['skew']:+.2f}")
    print(f"  max drawdown (cum. %)     {ps['max_drawdown_pct']*100:.1f}")
    print(f"  mean hedging cost         {ps['mean_costs_pct_of_spot']*100:.3f}% of spot per trade")
    print("  worst 5 trades (entry date -> P&L % of spot):")
    for d, x in ps["worst_trades"].items():
        print(f"      {d.date()}   {x*100:+.2f}%")

    # ── Plots ────────────────────────────────────────────────────────────────
    section("Plots")
    tag = "SYNTHETIC" if args.synthetic else "SPY / VIX"
    plot_iv_vs_rv(frame, f"{tag}: implied vs realized volatility",
                  save_path=os.path.join(PLOT_DIR, "iv_vs_rv.png"))
    plot_vrp(frame, f"{tag}: variance risk premium (IV − forward RV)",
             save_path=os.path.join(PLOT_DIR, "vrp.png"))
    plot_hedged_pnl(trades.iloc[::h], ps["cum_pct"], f"{tag}: short delta-hedged call, non-overlapping",
                    save_path=os.path.join(PLOT_DIR, "hedged_pnl.png"))


if __name__ == "__main__":
    main()
