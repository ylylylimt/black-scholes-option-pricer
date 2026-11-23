"""
Historical Data Loader
=======================
Daily SPY (dividend-adjusted), the CBOE VIX index (30-day implied vol on the
S&P 500) and the 13-week T-bill rate (^IRX) from Yahoo Finance, cached to CSV.

`synthetic_history` generates a stochastic-volatility market with a baked-in
variance risk premium. It exists only to exercise the pipeline offline and
in tests; results from it say nothing about real markets.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _extract_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Pull the Close column out of a yfinance frame (flat or MultiIndex columns)."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[symbol] if symbol in close.columns else close.iloc[:, 0]
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(symbol).dropna()


def fetch_history(
    start: str = "2010-01-01",
    end: str | None = None,
    cache_path: str = "data/history.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns
    spy   : dividend-adjusted close (a total-return index)
    vix   : VIX level in vol points (e.g. 18.5)
    rate  : 3-month T-bill yield as a decimal (forward-filled; 0.0 if unavailable)
    """
    if os.path.exists(cache_path) and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start:end] if end else df.loc[start:]

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    def grab(sym):
        raw = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"No data returned for {sym}")
        return _extract_close(raw, sym)

    spy, vix = grab("SPY"), grab("^VIX")
    try:
        irx = grab("^IRX") / 100.0
    except Exception:
        irx = pd.Series(dtype=float, name="^IRX")

    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    df["rate"] = irx.reindex(df.index).ffill().bfill().fillna(0.0) if len(irx) else 0.0

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path)
    return df


def synthetic_history(
    n: int = 3000,
    seed: int = 0,
    mean_vol: float = 0.16,
    vol_of_vol: float = 0.06,
    mean_reversion: float = 0.02,
    vrp_multiplier: float = 1.10,
    drift: float = 0.08,
    rate: float = 0.02,
) -> pd.DataFrame:
    """Synthetic market: log-vol Ornstein-Uhlenbeck, implied = 1.10 x smoothed true vol + noise."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = np.log(mean_vol)
    for t in range(1, n):
        x[t] = x[t - 1] + mean_reversion * (np.log(mean_vol) - x[t - 1]) + vol_of_vol * rng.standard_normal() / np.sqrt(21)
    sigma = np.exp(x)

    dt = 1 / 252
    rets = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    spy = 100 * np.exp(np.cumsum(rets))

    smooth = pd.Series(sigma).rolling(21, min_periods=1).mean().values
    iv = vrp_multiplier * smooth * np.exp(0.03 * rng.standard_normal(n))
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"spy": spy, "vix": iv * 100, "rate": rate}, index=idx)
