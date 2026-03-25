"""
Market Data Fetcher
====================
Pulls live spot prices and options chains from Yahoo Finance.
Computes implied volatility for each listed contract and structures
the data for vol surface analysis.
"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance: pip install yfinance")

from black_scholes import implied_volatility


def get_spot(ticker: str) -> float:
    """Return the current spot price for a ticker."""
    tk = yf.Ticker(ticker)
    info = tk.fast_info
    return float(info.last_price)


def get_risk_free_rate() -> float:
    """
    Approximate the risk-free rate from the 3-month US T-bill (^IRX).
    Falls back to 0.05 if unavailable.
    """
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.fast_info.last_price / 100
        return float(rate) if rate and rate > 0 else 0.05
    except Exception:
        return 0.05


def time_to_expiry(expiry_str: str) -> float:
    """Convert an expiry date string 'YYYY-MM-DD' to years from today."""
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
    today = datetime.today()
    days = (expiry - today).days
    return max(days / 365.0, 1e-6)


def fetch_options_chain(
    ticker: str,
    max_expirations: int = 4,
    moneyness_range: float = 0.30,
) -> pd.DataFrame:
    """
    Fetch the options chain for a ticker, compute IV for every contract,
    and return a tidy DataFrame.

    Parameters
    ----------
    ticker            : e.g. 'SPY', 'AAPL', 'TSLA'
    max_expirations   : number of expiry dates to pull
    moneyness_range   : keep strikes within +/- X% of spot (e.g. 0.30 = 30%)

    Returns
    -------
    DataFrame with columns:
        expiry, T, option_type, strike, moneyness, mid_price,
        bid, ask, volume, open_interest, implied_vol, bs_price,
        delta, gamma, theta, vega
    """
    tk = yf.Ticker(ticker)
    spot = get_spot(ticker)
    r = get_risk_free_rate()

    expirations = tk.options[:max_expirations]
    if not expirations:
        raise ValueError(f"No options data found for {ticker}")

    records = []

    for expiry in expirations:
        T = time_to_expiry(expiry)
        if T < 1 / 365:
            continue

        chain = tk.option_chain(expiry)

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()

            # Filter to near-the-money strikes
            lo = spot * (1 - moneyness_range)
            hi = spot * (1 + moneyness_range)
            df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]

            if df.empty:
                continue

            # Mid price from bid/ask; fall back to lastPrice
            df["mid"] = np.where(
                (df["bid"] > 0) & (df["ask"] > 0),
                (df["bid"] + df["ask"]) / 2,
                df["lastPrice"],
            )

            for _, row in df.iterrows():
                K = float(row["strike"])
                mid = float(row["mid"])

                if mid <= 0:
                    continue

                iv = implied_volatility(mid, spot, K, T, r, opt_type)
                if np.isnan(iv) or iv <= 0 or iv > 5:
                    continue

                from black_scholes import price as bs_price
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    greeks = bs_price(spot, K, T, r, iv, opt_type)

                moneyness = K / spot

                records.append({
                    "expiry": expiry,
                    "T": round(T, 4),
                    "option_type": opt_type,
                    "strike": K,
                    "moneyness": round(moneyness, 4),
                    "mid_price": round(mid, 4),
                    "bid": float(row.get("bid", np.nan)),
                    "ask": float(row.get("ask", np.nan)),
                    "volume": int(row.get("volume", 0) or 0),
                    "open_interest": int(row.get("openInterest", 0) or 0),
                    "implied_vol": round(iv, 4),
                    "bs_price": greeks.price,
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                })

    if not records:
        raise ValueError(f"No valid options contracts found for {ticker}")

    df_out = pd.DataFrame(records)
    df_out.sort_values(["T", "option_type", "strike"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    print(f"Fetched {len(df_out)} contracts for {ticker} | spot={spot:.2f} | r={r:.3f}")
    return df_out, spot, r
