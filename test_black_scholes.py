"""
Unit Tests — Black-Scholes Engine
===================================
Tests cover:
  - Known analytical values
  - Put-call parity
  - Greek signs and magnitudes
  - IV solver round-trip accuracy
  - Edge cases
"""

import numpy as np
import pytest
from black_scholes import price, implied_volatility, put_call_parity_check


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def atm_params():
    """At-the-money call, standard params."""
    return dict(S=100, K=100, T=0.25, r=0.05, sigma=0.20)


@pytest.fixture
def itm_call():
    return dict(S=110, K=100, T=0.25, r=0.05, sigma=0.20)


@pytest.fixture
def otm_call():
    return dict(S=90, K=100, T=0.25, r=0.05, sigma=0.20)


# ── Pricing accuracy ───────────────────────────────────────────────────────────
class TestPricing:
    def test_atm_call_known_value(self, atm_params):
        """ATM call price ~$4.61 for S=K=100, T=0.25, r=5%, σ=20%."""
        res = price(**atm_params, option_type="call")
        assert abs(res.price - 4.61) < 0.05

    def test_atm_put_known_value(self, atm_params):
        """ATM put price ~$3.37 (by put-call parity from ~4.61 call)."""
        res = price(**atm_params, option_type="put")
        assert abs(res.price - 3.37) < 0.05

    def test_put_call_parity(self, atm_params):
        result = put_call_parity_check(**atm_params)
        assert result["parity_holds"], f"Parity violated: {result}"

    def test_put_call_parity_itm(self, itm_call):
        result = put_call_parity_check(**itm_call)
        assert result["parity_holds"]

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call ~ intrinsic S - K*exp(-rT)."""
        res = price(S=200, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        intrinsic = 200 - 100 * np.exp(-0.05 * 0.25)
        assert abs(res.price - intrinsic) < 1.0

    def test_deep_otm_call_near_zero(self):
        res = price(S=50, K=200, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert res.price < 0.01

    def test_price_positive(self, atm_params):
        for opt in ["call", "put"]:
            res = price(**atm_params, option_type=opt)
            assert res.price > 0

    def test_invalid_T_raises(self, atm_params):
        with pytest.raises(ValueError):
            price(**{**atm_params, "T": 0}, option_type="call")

    def test_invalid_sigma_raises(self, atm_params):
        with pytest.raises(ValueError):
            price(**{**atm_params, "sigma": 0}, option_type="call")


# ── Greeks ─────────────────────────────────────────────────────────────────────
class TestGreeks:
    def test_call_delta_range(self, atm_params):
        """Call delta ∈ (0, 1)."""
        res = price(**atm_params, option_type="call")
        assert 0 < res.delta < 1

    def test_put_delta_range(self, atm_params):
        """Put delta ∈ (-1, 0)."""
        res = price(**atm_params, option_type="put")
        assert -1 < res.delta < 0

    def test_call_put_delta_sum(self, atm_params):
        """Δ_call + |Δ_put| ≈ 1  (put-call delta parity)."""
        c = price(**atm_params, option_type="call")
        p = price(**atm_params, option_type="put")
        assert abs(c.delta + p.delta - 1.0) < 1e-4

    def test_gamma_positive(self, atm_params):
        """Gamma is always positive for both calls and puts."""
        for opt in ["call", "put"]:
            res = price(**atm_params, option_type=opt)
            assert res.gamma > 0

    def test_gamma_equal_calls_puts(self, atm_params):
        """Gamma is identical for calls and puts with same params."""
        c = price(**atm_params, option_type="call")
        p = price(**atm_params, option_type="put")
        assert abs(c.gamma - p.gamma) < 1e-8

    def test_vega_positive(self, atm_params):
        """Vega is always positive."""
        for opt in ["call", "put"]:
            res = price(**atm_params, option_type=opt)
            assert res.vega > 0

    def test_vega_equal_calls_puts(self, atm_params):
        """Vega is identical for calls and puts."""
        c = price(**atm_params, option_type="call")
        p = price(**atm_params, option_type="put")
        assert abs(c.vega - p.vega) < 1e-5

    def test_theta_negative(self, atm_params):
        """Theta is negative (time decay hurts long options)."""
        for opt in ["call", "put"]:
            res = price(**atm_params, option_type=opt)
            assert res.theta < 0

    def test_call_rho_positive(self, atm_params):
        """Call rho is positive (higher rates → higher call value)."""
        res = price(**atm_params, option_type="call")
        assert res.rho > 0

    def test_put_rho_negative(self, atm_params):
        """Put rho is negative."""
        res = price(**atm_params, option_type="put")
        assert res.rho < 0

    def test_delta_monotone_in_spot(self):
        """Call delta increases as spot increases."""
        deltas = [price(S=s, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call").delta
                  for s in [80, 90, 100, 110, 120]]
        assert all(deltas[i] < deltas[i+1] for i in range(len(deltas)-1))

    def test_vega_numerical(self, atm_params):
        """Vega matches finite-difference approximation."""
        h = 0.001
        params_hi = {**atm_params, "sigma": atm_params["sigma"] + h}
        params_lo = {**atm_params, "sigma": atm_params["sigma"] - h}
        fd_vega = (price(**params_hi, option_type="call").price -
                   price(**params_lo, option_type="call").price) / (2 * h)
        analytic_vega = price(**atm_params, option_type="call").vega * 100  # per unit
        assert abs(fd_vega - analytic_vega) < 0.01


# ── Implied volatility ─────────────────────────────────────────────────────────
class TestImpliedVol:
    @pytest.mark.parametrize("true_vol", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_roundtrip_call(self, true_vol):
        """IV solver recovers true vol from model price (calls)."""
        mkt = price(S=100, K=100, T=0.25, r=0.05, sigma=true_vol, option_type="call").price
        iv = implied_volatility(mkt, 100, 100, 0.25, 0.05, "call")
        assert not np.isnan(iv)
        assert abs(iv - true_vol) < 1e-4

    @pytest.mark.parametrize("true_vol", [0.10, 0.20, 0.30, 0.50])
    def test_roundtrip_put(self, true_vol):
        """IV solver recovers true vol from model price (puts)."""
        mkt = price(S=100, K=100, T=0.25, r=0.05, sigma=true_vol, option_type="put").price
        iv = implied_volatility(mkt, 100, 100, 0.25, 0.05, "put")
        assert not np.isnan(iv)
        assert abs(iv - true_vol) < 1e-4

    def test_iv_otm_call(self):
        true_vol = 0.25
        mkt = price(S=100, K=110, T=0.5, r=0.05, sigma=true_vol, option_type="call").price
        iv = implied_volatility(mkt, 100, 110, 0.5, 0.05, "call")
        assert abs(iv - true_vol) < 1e-4

    def test_iv_intrinsic_returns_nan(self):
        """Price at or below intrinsic value should return nan."""
        iv = implied_volatility(0.0, 100, 100, 0.25, 0.05, "call")
        assert np.isnan(iv)

    def test_iv_expired_returns_nan(self):
        iv = implied_volatility(5.0, 100, 100, 0.0, 0.05, "call")
        assert np.isnan(iv)
