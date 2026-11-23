"""
Delta-Hedged Option P&L Simulation
===================================
A mini-backtest of the textbook volatility-selling trade:

    sell a fixed-tenor (default 21 trading days) at-the-money call at the
    prevailing implied vol, delta-hedge it every day with the underlying,
    hold to expiry, and measure the P&L.

Why this is informative
-----------------------
For a continuously delta-hedged option, P&L is (to first order) independent
of the underlying's drift and equals

        P&L  ~  integral  0.5 * Gamma * S^2 * (sigma_implied^2 - sigma_realized_t^2) dt

so the trade earns money when realized variance comes in below implied
variance, and loses when it comes in above. This module lets you verify that
numerically and measure how big, and how fat-tailed, the result is.

Mechanics (per trade, one option on one unit of underlying)
-----------------------------------------------------------
t0 : receive the BSM premium C0, buy delta_0 units of the underlying, put the
     remainder in a cash account earning the risk-free rate r.
t_j: cash accrues at r, then the hedge is rebalanced to delta_j (computed from
     BSM with the remaining time and the hedging vol).
T  : unwind the hedge, pay max(S_T - K, 0). P&L is measured at expiry.

Simplifications (see README for the full list)
----------------------------------------------
* r is held constant over the life of each trade (set at entry).
* The underlying is treated as a non-dividend asset. Feed it a total-return
  (dividend-adjusted) price series, which behaves exactly like one.
* Proportional transaction costs on the underlying only, none on the option,
  and no bid/ask on the option itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from vol_analysis import newey_west_mean

TRADING_DAYS = 252


def bs_call_price_delta(S, K, tau, r, sigma):
    """Vectorised BSM European call price and delta. All inputs array-like, tau > 0."""
    S, K, tau, r, sigma = (np.asarray(a, dtype=float) for a in (S, K, tau, r, sigma))
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    call = S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)
    return call, ndtr(d1)


def simulate_delta_hedged_short_call(
    spot,
    iv,
    r=0.0,
    horizon: int = 21,
    step: int = 1,
    moneyness: float = 1.0,
    hedge_vol: str = "entry",
    tc_bps: float = 0.0,
    dt: float = 1.0 / TRADING_DAYS,
) -> pd.DataFrame:
    """
    Simulate a short, delta-hedged call opened on every `step`-th day.

    Parameters
    ----------
    spot       : prices of the underlying (Series or array), length N
    iv         : implied vol as DECIMAL, same length. Used for the premium
                 (at entry) and, if hedge_vol="daily", for each day's delta
    r          : continuously-compounded risk-free rate, scalar or length-N array
    horizon    : option life in trading days
    step       : days between trade entries (use step=horizon for non-overlapping)
    moneyness  : strike / spot at entry (1.0 = ATM)
    hedge_vol  : "entry"  -> delta from the vol the option was sold at (default)
                 "daily"  -> delta from that day's implied vol
    tc_bps     : proportional cost, in basis points of traded notional, charged
                 on the opening hedge, every rebalance and the final unwind
    dt         : year fraction per step

    Returns
    -------
    DataFrame indexed by entry date (or integer position) with columns
    spot_entry, premium, payoff, pnl, pnl_pct (pnl / spot at entry), iv_entry,
    rv_realized (RMS over the trade's life), vrp_vol (iv_entry - rv_realized),
    costs.
    """
    if hedge_vol not in ("entry", "daily"):
        raise ValueError("hedge_vol must be 'entry' or 'daily'")

    index = spot.index if isinstance(spot, pd.Series) else None
    S = np.asarray(spot, dtype=float)
    sig = np.asarray(iv, dtype=float)
    N = len(S)
    if len(sig) != N:
        raise ValueError("spot and iv must have the same length")
    rr = np.broadcast_to(np.asarray(r, dtype=float), S.shape)
    if N <= horizon:
        raise ValueError("need more observations than the option horizon")

    starts = np.arange(0, N - horizon, step)
    paths = S[starts[:, None] + np.arange(horizon + 1)[None, :]]   # trades x (h+1)
    S0 = paths[:, 0]
    K = moneyness * S0
    sig0 = sig[starts]
    r0 = rr[starts]
    tc = tc_bps / 1e4

    premium, _ = bs_call_price_delta(S0, K, horizon * dt, r0, sig0)

    cash = np.zeros(len(starts))
    costs = np.zeros(len(starts))
    prev = np.zeros(len(starts))
    growth = np.exp(r0 * dt)

    for j in range(horizon):
        tau = (horizon - j) * dt
        sigma_h = sig0 if hedge_vol == "entry" else sig[starts + j]
        Sj = paths[:, j]
        _, delta = bs_call_price_delta(Sj, K, tau, r0, sigma_h)

        if j == 0:
            trade = delta
            cash = premium - delta * Sj
        else:
            trade = delta - prev
            cash = cash * growth - trade * Sj
        fee = tc * np.abs(trade) * Sj
        cash -= fee
        costs += fee
        prev = delta

    cash = cash * growth                               # accrue the final interval
    ST = paths[:, horizon]
    unwind_fee = tc * np.abs(prev) * ST
    payoff = np.maximum(ST - K, 0.0)
    pnl = cash + prev * ST - unwind_fee - payoff
    costs += unwind_fee

    rets = np.log(paths[:, 1:] / paths[:, :-1])
    rv = np.sqrt((rets ** 2).mean(axis=1) / dt)

    out = pd.DataFrame(
        {
            "spot_entry": S0,
            "premium": premium,
            "payoff": payoff,
            "pnl": pnl,
            "pnl_pct": pnl / S0,
            "iv_entry": sig0,
            "rv_realized": rv,
            "vrp_vol": sig0 - rv,
            "costs": costs,
        },
        index=index[starts] if index is not None else starts,
    )
    return out


def summarize_hedged_pnl(trades: pd.DataFrame, horizon: int = 21) -> dict:
    """
    Performance statistics for a trade table from `simulate_delta_hedged_short_call`.

    Headline stats use NON-OVERLAPPING trades (every `horizon`-th entry) so
    observations are close to independent. The mean is also tested on the
    full overlapping sample with a Newey-West standard error as a cross-check.
    All P&L is in percent of spot at entry (per one unit of underlying).
    """
    nonover = trades.iloc[::horizon]
    p = nonover["pnl_pct"]
    cum = p.cumsum()
    drawdown = cum - cum.cummax()
    n = len(p)
    sd = float(p.std(ddof=1)) if n > 1 else float("nan")
    mean = float(p.mean())

    nw = newey_west_mean(trades["pnl_pct"].values, lags=horizon)
    worst = nonover["pnl_pct"].nsmallest(5)

    return {
        "n_trades_nonoverlapping": n,
        "mean_pct": mean,
        "std_pct": sd,
        "t_stat_nonoverlapping": mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan"),
        "t_stat_overlapping_nw": nw["t_stat"],
        "sharpe_annualised": mean / sd * np.sqrt(TRADING_DAYS / horizon) if sd > 0 else float("nan"),
        "hit_rate": float((p > 0).mean()),
        "skew": float(p.skew()),
        "worst_trades": worst,
        "max_drawdown_pct": float(drawdown.min()),
        "cum_pct": cum,
        "mean_costs_pct_of_spot": float((nonover["costs"] / nonover["spot_entry"]).mean()),
    }
