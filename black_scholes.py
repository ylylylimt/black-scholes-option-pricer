"""
Black-Scholes Options Pricing Engine
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Literal


@dataclass
class BSMResult:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% move in vol
    rho: float    # per 1% move in rate


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 terms in the BSM formula."""
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T and sigma must be positive (got T={T}, sigma={sigma})")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> BSMResult:
    """
    Price a European option and return all Greeks.

    Parameters
    ----------
    S     : Current spot price
    K     : Strike price
    T     : Time to expiry in years
    r     : Risk-free rate (annualised, continuous)
    sigma : Implied / historical volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    BSMResult with price and all Greeks
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    phi = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    if option_type == "call":
        opt_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (
            -S * phi * sigma / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        opt_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (
            -S * phi * sigma / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    gamma = phi / (S * sigma * sqrt_T)
    vega = S * phi * sqrt_T / 100  # per 1% vol move

    return BSMResult(
        price=round(opt_price, 6),
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied volatility using Newton-Raphson with Brent fallback.

    Returns IV as a decimal (e.g. 0.25 = 25%) or np.nan if unsolvable.
    """
    if T <= 0:
        return np.nan

    # Intrinsic value check — market price must exceed intrinsic
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price <= intrinsic + 1e-8:
        return np.nan

    def objective(sigma):
        try:
            return price(S, K, T, r, sigma, option_type).price - market_price
        except ValueError:
            return np.nan

    # Newton-Raphson
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        try:
            result = price(S, K, T, r, sigma, option_type)
            diff = result.price - market_price
            if abs(diff) < tol:
                return sigma
            # vega is per 1% — convert back to per unit for Newton step
            vega_unit = result.vega * 100
            if abs(vega_unit) < 1e-10:
                break
            sigma -= diff / vega_unit
            if sigma <= 0:
                sigma = 1e-6
        except Exception:
            break

    # Brent fallback over a wide IV range
    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except Exception:
        return np.nan


def put_call_parity_check(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Verify put-call parity: C - P = S - K*exp(-rT)."""
    call = price(S, K, T, r, sigma, "call")
    put = price(S, K, T, r, sigma, "put")
    lhs = call.price - put.price
    rhs = S - K * np.exp(-r * T)
    return {
        "call_price": call.price,
        "put_price": put.price,
        "C - P": round(lhs, 6),
        "S - K*exp(-rT)": round(rhs, 6),
        "parity_holds": abs(lhs - rhs) < 1e-4,
    }
