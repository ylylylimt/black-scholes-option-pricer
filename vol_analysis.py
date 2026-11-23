"""
Realized Volatility & Variance Risk Premium
============================================
Tools to compare implied volatility with subsequently realized volatility
and to test, with proper standard errors, whether the gap (the variance
risk premium, VRP) is statistically different from zero.

Conventions
-----------
* Realized variance over a window is the *root-mean-square* of daily log
  returns (no mean subtraction), annualised with 252 trading days. This is
  the same quantity a variance swap pays on, and avoids noisy mean estimates.
* "Forward" realized vol at date t uses returns on days t+1 ... t+window,
  i.e. exactly the period an option bought at t would live through.
  "Trailing" realized vol at t uses returns on days t-window+1 ... t.
* VRP is reported in volatility points (IV - RV) and variance points
  (IV^2 - RV^2). The variance version is the economically natural one
  (a variance swap's payoff is linear in it).

Statistics
----------
Daily observations of a 21-day forward window overlap by 20/21, so
residuals are strongly autocorrelated and naive OLS standard errors are far
too small. All inference here uses Newey-West (HAC) standard errors with a
Bartlett kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Realized volatility ───────────────────────────────────────────────────────
def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns (first value is NaN)."""
    return np.log(prices).diff()


def realized_vol_trailing(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualised RMS log-return vol over the last `window` returns (incl. today)."""
    r = log_returns(prices)
    return np.sqrt(TRADING_DAYS * (r ** 2).rolling(window).mean())


def realized_vol_forward(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualised RMS log-return vol over the NEXT `window` returns (t+1 ... t+window)."""
    return realized_vol_trailing(prices, window).shift(-window)


def build_vrp_frame(spot: pd.Series, iv: pd.Series, window: int = 21) -> pd.DataFrame:
    """
    Align spot prices with an implied-vol series and compute realized vol and VRP.

    Parameters
    ----------
    spot   : daily closes of the underlying (dividend-adjusted recommended)
    iv     : implied vol as a DECIMAL (e.g. VIX / 100), same dates
    window : trading days in the realized-vol window (21 ~ 30 calendar days)

    Returns
    -------
    DataFrame indexed by date with columns
    spot, iv, rv_trailing, rv_forward, vrp_vol, vrp_var.
    Rows without a full forward window are dropped.
    """
    idx = spot.index.intersection(iv.index)
    spot = spot.loc[idx].astype(float)
    iv = iv.loc[idx].astype(float)

    df = pd.DataFrame({"spot": spot, "iv": iv})
    df["rv_trailing"] = realized_vol_trailing(spot, window)
    df["rv_forward"] = realized_vol_forward(spot, window)
    df["vrp_vol"] = df["iv"] - df["rv_forward"]
    df["vrp_var"] = df["iv"] ** 2 - df["rv_forward"] ** 2
    return df.dropna(subset=["iv", "rv_forward"])


# ── HAC regression ────────────────────────────────────────────────────────────
@dataclass
class HACResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    n: int
    lags: int

    def table(self, names=None) -> pd.DataFrame:
        names = names or [f"x{i}" for i in range(len(self.beta))]
        return pd.DataFrame(
            {"coef": self.beta, "hac_se": self.se, "t_stat": self.t}, index=names
        )


def hac_ols(y, X, lags: int) -> HACResult:
    """
    OLS with Newey-West (Bartlett kernel) heteroskedasticity-and-autocorrelation
    consistent standard errors.

        V = (X'X)^-1 S (X'X)^-1,   S = G0 + sum_{l=1..L} w_l (G_l + G_l'),
        G_l = sum_t (x_t u_t)(x_{t-l} u_{t-l})',   w_l = 1 - l/(L+1)

    No small-sample correction is applied. `X` must already contain a
    constant column if you want an intercept.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if lags >= n:
        raise ValueError("lags must be smaller than the number of observations")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]

    S = xu.T @ xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = xu[l:].T @ xu[:-l]
        S += w * (G + G.T)

    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    return HACResult(beta=beta, se=se, t=beta / se, n=n, lags=lags)


def newey_west_mean(x, lags: int) -> dict:
    """Mean of a series with a Newey-West standard error and t-statistic."""
    x = np.asarray(x, dtype=float)
    res = hac_ols(x, np.ones(len(x)), lags)
    return {
        "mean": float(res.beta[0]),
        "se": float(res.se[0]),
        "t_stat": float(res.t[0]),
        "n": res.n,
        "lags": lags,
    }


# ── Summaries ─────────────────────────────────────────────────────────────────
def vrp_summary(frame: pd.DataFrame, lags: int) -> dict:
    """Headline VRP statistics with HAC inference."""
    vol = newey_west_mean(frame["vrp_vol"], lags)
    var = newey_west_mean(frame["vrp_var"], lags)
    return {
        "n_obs": len(frame),
        "start": frame.index[0],
        "end": frame.index[-1],
        "mean_iv": float(frame["iv"].mean()),
        "mean_rv_forward": float(frame["rv_forward"].mean()),
        "vrp_vol": vol,
        "vrp_var": var,
        "pct_days_vrp_positive": float((frame["vrp_vol"] > 0).mean()),
        "vrp_vol_percentiles": {
            p: float(np.percentile(frame["vrp_vol"], p)) for p in (1, 5, 25, 50, 75, 95, 99)
        },
    }


def mincer_zarnowitz(frame: pd.DataFrame, lags: int) -> dict:
    """
    Forecast-content regressions for forward realized vol (HAC standard errors):

      (1)  rv_forward = a + b * iv
      (2)  rv_forward = a + b * iv + c * rv_trailing

    If implied vol were an unbiased forecast, a = 0 and b = 1. b < 1 with a > 0
    means implied vol overshoots when high and undershoots when low; a
    significant c means past realized vol carries information that implied
    vol does not.
    """
    y = frame["rv_forward"].values
    ones = np.ones(len(frame))

    X1 = np.column_stack([ones, frame["iv"].values])
    r1 = hac_ols(y, X1, lags)

    sub = frame.dropna(subset=["rv_trailing"])
    X2 = np.column_stack(
        [np.ones(len(sub)), sub["iv"].values, sub["rv_trailing"].values]
    )
    r2 = hac_ols(sub["rv_forward"].values, X2, lags)

    return {
        "iv_only": r1.table(["const", "iv"]),
        "iv_and_trailing": r2.table(["const", "iv", "rv_trailing"]),
    }
