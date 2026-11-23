"""
Unit Tests — Realized Vol, VRP and HAC statistics
===================================================
Each test checks a property that must hold if the code is correct,
using synthetic data where the true answer is known.
"""

import numpy as np
import pandas as pd
import pytest

from vol_analysis import (
    build_vrp_frame,
    hac_ols,
    mincer_zarnowitz,
    newey_west_mean,
    realized_vol_forward,
    realized_vol_trailing,
)


def _gbm_prices(n, vol, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)


class TestRealizedVol:
    def test_recovers_constant_vol(self):
        """RMS realized vol of GBM with sigma=20% is ~20%."""
        p = _gbm_prices(20000, 0.20, seed=1)
        rv = realized_vol_trailing(p, window=5000).dropna()
        assert abs(rv.mean() - 0.20) < 0.005

    def test_forward_trailing_alignment(self):
        """
        One 10% shock on day k. Trailing vol sees it from day k onward;
        forward vol sees it only from days BEFORE k. This is the look-ahead
        boundary: getting it wrong would leak the future into the VRP.
        """
        n, k, w = 100, 50, 21
        rets = np.zeros(n)
        rets[k] = 0.10
        p = pd.Series(100 * np.exp(np.cumsum(rets)))
        trail = realized_vol_trailing(p, w)
        fwd = realized_vol_forward(p, w)

        assert trail[k - 1] == 0.0 and trail[k] > 0
        assert trail[k + w - 1] > 0 and trail[k + w] == 0.0   # shock leaves window
        assert fwd[k - 1] > 0 and fwd[k] == 0.0               # shock is in the future of k-1, not of k
        assert fwd[k - w] > 0 and fwd[k - w - 1] == 0.0       # earliest day that sees it

    def test_forward_equals_shifted_trailing(self):
        p = _gbm_prices(500, 0.3, seed=2)
        w = 21
        a = realized_vol_forward(p, w)
        b = realized_vol_trailing(p, w).shift(-w)
        pd.testing.assert_series_equal(a, b)


class TestVRPFrame:
    def test_sign_and_size(self):
        """IV=30% vs true vol 20%: VRP ~ +10 vol pts, ~ +0.05 variance pts."""
        p = _gbm_prices(20000, 0.20, seed=3)
        iv = pd.Series(0.30, index=p.index)
        df = build_vrp_frame(p, iv, window=21)
        assert abs(df["vrp_vol"].mean() - 0.10) < 0.01
        assert abs(df["vrp_var"].mean() - (0.30 ** 2 - 0.20 ** 2)) < 0.005

    def test_drops_rows_without_forward_window(self):
        p = _gbm_prices(200, 0.2, seed=4)
        df = build_vrp_frame(p, pd.Series(0.2, index=p.index), window=21)
        # last usable day t needs returns t+1..t+21 -> t = N-1-21
        assert df.index[-1] == p.index[200 - 1 - 21]
        assert df["rv_forward"].notna().all()

    def test_aligns_on_common_dates(self):
        p = _gbm_prices(300, 0.2, seed=5)
        iv = pd.Series(0.2, index=p.index[10:])         # IV missing the first 10 days
        df = build_vrp_frame(p, iv, window=21)
        assert df.index[0] >= p.index[10]


class TestHAC:
    def test_ols_recovers_coefficients(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(5000)
        y = 1.0 + 2.0 * x + 0.1 * rng.standard_normal(5000)
        res = hac_ols(y, np.column_stack([np.ones(5000), x]), lags=5)
        assert abs(res.beta[0] - 1.0) < 0.01
        assert abs(res.beta[1] - 2.0) < 0.01

    def test_iid_hac_se_matches_classical(self):
        """With i.i.d. data the Newey-West SE of the mean ~ sigma/sqrt(n)."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal(20000)
        nw = newey_west_mean(x, lags=10)
        classical = x.std(ddof=1) / np.sqrt(len(x))
        assert 0.9 < nw["se"] / classical < 1.1

    def test_autocorrelated_se_is_much_larger(self):
        """AR(1), phi=0.9: true SE of the mean is ~4x naive. HAC must notice."""
        rng = np.random.default_rng(12)
        n, phi = 20000, 0.9
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        nw = newey_west_mean(x, lags=60)
        naive = x.std(ddof=1) / np.sqrt(n)
        assert nw["se"] / naive > 2.5

    def test_overlapping_window_series_has_inflated_se(self):
        """The reason HAC is needed: a rolling 21-day mean of noise is autocorrelated."""
        rng = np.random.default_rng(13)
        noise = pd.Series(rng.standard_normal(20000))
        roll = noise.rolling(21).mean().dropna().values
        nw = newey_west_mean(roll, lags=21)
        naive = roll.std(ddof=1) / np.sqrt(len(roll))
        assert nw["se"] / naive > 2.5

    def test_lags_too_large_raises(self):
        with pytest.raises(ValueError):
            hac_ols(np.arange(5.0), np.ones(5), lags=5)


class TestMincerZarnowitz:
    def test_recovers_known_relationship(self):
        """Build rv_forward = 0.02 + 0.5*iv + noise; the regression must find b ~ 0.5."""
        rng = np.random.default_rng(20)
        n = 4000
        idx = pd.bdate_range("2010-01-01", periods=n)
        iv = 0.15 + 0.05 * np.abs(rng.standard_normal(n))
        rv_f = 0.02 + 0.5 * iv + 0.01 * rng.standard_normal(n)
        frame = pd.DataFrame(
            {"iv": iv, "rv_forward": rv_f, "rv_trailing": 0.5 * rv_f + 0.05}, index=idx
        )
        out = mincer_zarnowitz(frame, lags=21)
        assert abs(out["iv_only"].loc["iv", "coef"] - 0.5) < 0.03
        assert abs(out["iv_only"].loc["const", "coef"] - 0.02) < 0.01
