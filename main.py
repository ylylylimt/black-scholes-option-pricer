"""
Options Pricing & Greeks Explorer
===================================
Main analysis pipeline. Run with:

    python main.py                  # default ticker SPY
    python main.py --ticker AAPL
    python main.py --ticker TSLA --expirations 3

Outputs
-------
  - Console summary table of near-ATM contracts
  - plots/vol_smile.png
  - plots/vol_surface.png
  - plots/greeks_profile.png
  - plots/model_fit.png
"""

import argparse
import os
import numpy as np
import pandas as pd

from black_scholes import price as bs_price, put_call_parity_check, implied_volatility
from market_data import fetch_options_chain, get_spot, get_risk_free_rate
from visualise import (
    plot_vol_smile,
    plot_vol_surface,
    plot_greeks_profile,
    plot_model_fit,
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def print_section(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def demo_pricer(S: float, K: float, T: float, r: float, sigma: float):
    """Print a side-by-side call/put pricing table."""
    print_section("Black-Scholes Pricer — Single Contract")
    header = f"  S={S}  K={K}  T={T:.2f}y  r={r*100:.1f}%  σ={sigma*100:.0f}%"
    print(header)
    print()

    rows = []
    for opt in ["call", "put"]:
        res = bs_price(S, K, T, r, sigma, opt)
        rows.append({
            "Type":   opt.capitalize(),
            "Price":  f"${res.price:.4f}",
            "Delta":  f"{res.delta:+.4f}",
            "Gamma":  f"{res.gamma:.6f}",
            "Theta":  f"{res.theta:+.6f} /day",
            "Vega":   f"{res.vega:.4f} /1%σ",
            "Rho":    f"{res.rho:+.4f} /1%r",
        })

    df = pd.DataFrame(rows).set_index("Type")
    print(df.to_string())

    parity = put_call_parity_check(S, K, T, r, sigma)
    print(f"\n  Put-call parity check: C-P={parity['C - P']:.4f}  "
          f"S-Ke^(-rT)={parity['S - K*exp(-rT)']:.4f}  "
          f"{'✓ holds' if parity['parity_holds'] else '✗ VIOLATED'}")


def demo_iv_solver(S: float, K: float, T: float, r: float):
    """Demonstrate the IV solver round-trip."""
    print_section("Implied Volatility Solver — Round-trip Test")

    test_vols = [0.10, 0.20, 0.30, 0.50]
    rows = []
    for true_vol in test_vols:
        for opt in ["call", "put"]:
            mkt_price = bs_price(S, K, T, r, true_vol, opt).price
            solved_iv = implied_volatility(mkt_price, S, K, T, r, opt)
            rows.append({
                "Type":     opt.capitalize(),
                "True IV":  f"{true_vol*100:.0f}%",
                "Mkt Price":f"${mkt_price:.4f}",
                "Solved IV": f"{solved_iv*100:.2f}%" if not np.isnan(solved_iv) else "n/a",
                "Error":    f"{abs(solved_iv - true_vol)*100:.6f}%" if not np.isnan(solved_iv) else "n/a",
            })

    print(pd.DataFrame(rows).to_string(index=False))


def live_analysis(ticker: str, max_expirations: int):
    """Fetch real market data and run full analysis."""
    print_section(f"Live Market Analysis — {ticker}")

    print("  Fetching options chain...")
    df, spot, r = fetch_options_chain(ticker, max_expirations=max_expirations)

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n  Contracts fetched : {len(df)}")
    print(f"  Expirations       : {df['expiry'].nunique()}")
    print(f"  Spot              : ${spot:.2f}")
    print(f"  Risk-free rate    : {r*100:.2f}%")

    # Near-ATM contracts (moneyness within 5%)
    atm = df[abs(df["moneyness"] - 1.0) < 0.05].copy()
    atm_calls = atm[atm["option_type"] == "call"].sort_values("T")
    atm_puts  = atm[atm["option_type"] == "put"].sort_values("T")

    print_section("Near-ATM Implied Vols by Expiry")
    cols = ["expiry", "T", "option_type", "strike", "mid_price", "implied_vol", "delta", "gamma", "vega"]
    summary = pd.concat([atm_calls, atm_puts])[cols].sort_values(["T", "option_type"])
    summary["implied_vol"] = summary["implied_vol"].map("{:.1%}".format)
    summary["delta"]  = summary["delta"].map("{:+.3f}".format)
    summary["gamma"]  = summary["gamma"].map("{:.5f}".format)
    summary["vega"]   = summary["vega"].map("{:.4f}".format)
    print(summary.to_string(index=False))

    # IV stats per expiry
    print_section("IV Term Structure (ATM Calls)")
    ts = (
        atm_calls.groupby("expiry")["implied_vol"]
        .agg(["mean", "min", "max"])
        .rename(columns={"mean": "Mean IV", "min": "Min IV", "max": "Max IV"})
    )
    ts = ts.applymap(lambda x: f"{x:.1%}")
    print(ts.to_string())

    # ── Plots ──────────────────────────────────────────────────────────────────
    print_section("Generating Plots")

    print("  [1/4] Vol smile...")
    plot_vol_smile(df, spot, ticker, save_path=f"{PLOT_DIR}/vol_smile.png")

    print("  [2/4] Vol surface...")
    plot_vol_surface(df, ticker, save_path=f"{PLOT_DIR}/vol_surface.png")

    # Use ATM IV from nearest expiry for Greeks profile
    if not atm_calls.empty:
        atm_row = atm_calls.iloc[0]
        K_atm = atm_row["strike"]
        T_atm = atm_row["T"]
        iv_atm = atm_row["implied_vol"]
    else:
        K_atm, T_atm, iv_atm = spot, 0.25, 0.20

    print("  [3/4] Greeks profile...")
    plot_greeks_profile(spot, K_atm, T_atm, r, iv_atm, ticker,
                        save_path=f"{PLOT_DIR}/greeks_profile.png")

    print("  [4/4] Model fit...")
    plot_model_fit(df, ticker, save_path=f"{PLOT_DIR}/model_fit.png")

    return df


def main():
    parser = argparse.ArgumentParser(description="BSM Options Pricing & Vol Analysis")
    parser.add_argument("--ticker",      type=str, default="SPY",  help="Ticker symbol (default: SPY)")
    parser.add_argument("--expirations", type=int, default=4,      help="Number of expiry dates (default: 4)")
    parser.add_argument("--demo-only",   action="store_true",       help="Run pricer/IV demo without fetching live data")
    args = parser.parse_args()

    # ── Standalone demos (no internet needed) ─────────────────────────────────
    demo_pricer(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
    demo_iv_solver(S=100, K=100, T=0.25, r=0.05)

    if not args.demo_only:
        live_analysis(args.ticker, args.expirations)
    else:
        print("\n  (Skipping live data fetch — demo-only mode)")


if __name__ == "__main__":
    main()
