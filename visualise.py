"""
Visualisation
==============
Produces publication-quality plots for:
  1. Volatility smile per expiry
  2. Implied volatility surface (3-D)
  3. Greeks profiles as a function of spot
  4. IV vs market price scatter (model fit)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
from matplotlib import cm

from black_scholes import price as bs_price

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.family": "monospace",
    "font.size": 10,
})

CALL_COLOR = "#4a90d9"
PUT_COLOR  = "#e05c5c"
GREEK_COLORS = ["#4a90d9", "#e05c5c", "#50b86c", "#f5a623", "#9b59b6"]


# ── 1. Vol smile ───────────────────────────────────────────────────────────────
def plot_vol_smile(df: pd.DataFrame, spot: float, ticker: str, save_path: str = None):
    """
    Plot IV vs strike for calls and puts, faceted by expiry.
    """
    expirations = df["expiry"].unique()
    n = len(expirations)
    ncols = min(n, 2)
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(f"{ticker} — Implied Volatility Smile", fontsize=13, fontweight="bold", y=1.01)

    for idx, expiry in enumerate(expirations):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["expiry"] == expiry]
        T_val = sub["T"].iloc[0]

        for opt_type, color, marker in [("call", CALL_COLOR, "o"), ("put", PUT_COLOR, "s")]:
            d = sub[sub["option_type"] == opt_type].sort_values("strike")
            if d.empty:
                continue
            ax.plot(d["strike"], d["implied_vol"] * 100, color=color,
                    marker=marker, markersize=4, linewidth=1.5, label=opt_type.capitalize())

        ax.axvline(spot, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Spot")
        ax.set_title(f"Expiry: {expiry}  (T={T_val:.3f}y)", fontsize=10)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Vol (%)")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()


# ── 2. Vol surface ─────────────────────────────────────────────────────────────
def plot_vol_surface(df: pd.DataFrame, ticker: str, save_path: str = None):
    """
    3-D vol surface: moneyness (K/S) × time-to-expiry × IV.
    """
    df_calls = df[df["option_type"] == "call"].copy()
    df_calls = df_calls.dropna(subset=["implied_vol"])

    if df_calls.empty:
        print("  Not enough data for vol surface.")
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df_calls["moneyness"],
        df_calls["T"],
        df_calls["implied_vol"] * 100,
        c=df_calls["implied_vol"] * 100,
        cmap=cm.RdYlGn_r,
        s=30,
        alpha=0.85,
        edgecolors="none",
    )

    ax.set_xlabel("Moneyness (K/S)", labelpad=8)
    ax.set_ylabel("Time to Expiry (years)", labelpad=8)
    ax.set_zlabel("Implied Vol (%)", labelpad=8)
    ax.set_title(f"{ticker} — Implied Volatility Surface (Calls)", fontsize=12, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="IV (%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()


# ── 3. Greeks profiles ─────────────────────────────────────────────────────────
def plot_greeks_profile(
    S: float, K: float, T: float, r: float, sigma: float,
    ticker: str, save_path: str = None
):
    """
    Plot how each Greek varies as a function of spot price.
    """
    spot_range = np.linspace(S * 0.6, S * 1.4, 300)
    greeks_data = {g: {"call": [], "put": []} for g in ["delta", "gamma", "theta", "vega", "rho"]}

    for s in spot_range:
        for opt in ["call", "put"]:
            try:
                res = bs_price(s, K, T, r, sigma, opt)
                for g in greeks_data:
                    greeks_data[g][opt].append(getattr(res, g))
            except Exception:
                for g in greeks_data:
                    greeks_data[g][opt].append(np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"{ticker} — Greeks Profile  (K={K:.0f}, T={T:.2f}y, σ={sigma*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )

    greek_names = list(greeks_data.keys())
    for i, g in enumerate(greek_names):
        ax = axes[i // 3][i % 3]
        ax.plot(spot_range, greeks_data[g]["call"], color=CALL_COLOR, linewidth=2, label="Call")
        ax.plot(spot_range, greeks_data[g]["put"],  color=PUT_COLOR,  linewidth=2, label="Put")
        ax.axvline(S, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Current spot")
        ax.axvline(K, color="gray",  linewidth=1, linestyle=":",  alpha=0.5, label="Strike")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(g.capitalize(), fontsize=11)
        ax.set_xlabel("Spot price")
        ax.legend(fontsize=7)

    # Use last subplot for price
    ax = axes[1][2]
    for opt, color in [("call", CALL_COLOR), ("put", PUT_COLOR)]:
        prices = []
        for s in spot_range:
            try:
                prices.append(bs_price(s, K, T, r, sigma, opt).price)
            except Exception:
                prices.append(np.nan)
        ax.plot(spot_range, prices, color=color, linewidth=2, label=opt.capitalize())
    ax.axvline(S, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(K, color="gray",  linewidth=1, linestyle=":",  alpha=0.5)
    ax.set_title("Option Price", fontsize=11)
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()


# ── 4. Model fit ───────────────────────────────────────────────────────────────
def plot_model_fit(df: pd.DataFrame, ticker: str, save_path: str = None):
    """
    Scatter: market mid price vs BSM model price.
    Perfect fit lies on the diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{ticker} — Market Price vs BSM Model Price", fontsize=12, fontweight="bold")

    for ax, opt_type, color in [(axes[0], "call", CALL_COLOR), (axes[1], "put", PUT_COLOR)]:
        sub = df[df["option_type"] == opt_type].dropna(subset=["bs_price", "mid_price"])
        if sub.empty:
            continue
        ax.scatter(sub["mid_price"], sub["bs_price"], alpha=0.6, s=20,
                   color=color, edgecolors="none", label=opt_type.capitalize())
        lim = max(sub["mid_price"].max(), sub["bs_price"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4, label="Perfect fit")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Market mid price ($)")
        ax.set_ylabel("BSM model price ($)")
        ax.set_title(opt_type.capitalize() + "s", fontsize=11)
        ax.legend(fontsize=8)

        residuals = sub["bs_price"] - sub["mid_price"]
        rmse = np.sqrt((residuals**2).mean())
        ax.text(0.05, 0.92, f"RMSE: ${rmse:.3f}", transform=ax.transAxes,
                fontsize=9, color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()
