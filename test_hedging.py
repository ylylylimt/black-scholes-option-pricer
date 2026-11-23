"""
Unit Tests — Delta-Hedged P&L Simulation
==========================================
The key validation: in a world where the Black-Scholes assumptions hold, a
delta-hedged option earns ~0 when realized vol equals implied vol, earns money
when realized < implied (short vol), and loses when realized > implied. The
simulator must reproduce all three, and must match the analytic gamma P&L
formula trade-by-trade.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from black_scholes import price
from hedging import bs_call_price_delta, simulate_delta_hedged_short_call, summarize_hedged_pnl

H = 21
DT = 1 / 252


def _gbm(n, vol, seed, drift=0.08, s0=100.0):
    rng = np.random.default_rng(seed)
    rets = (drift - 0.5 * vol ** 2) * DT + vol * np.sqrt(DT) * rng.standard_normal(n)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))


def _run(real_vol, impl_vol, seed=0, n=60000, **kw):
    S = _gbm(n, real_vol, seed)
    iv = np.full(len(S), impl_vol)
    return simulate_delta_hedged_short_call(S, iv, r=0.03, horizon=H, step=H, **kw)


def _mean_and_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


class TestVectorisedBSM:
    def test_matches_scalar_pricer(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            S = rng.uniform(50, 150)
            K = rng.uniform(50, 150)
            T = rng.uniform(0.02, 1.0)
            r = rng.uniform(0.0, 0.06)
            sig = rng.uniform(0.08, 0.6)
            ref = price(S, K, T, r, sig, "call")
            p, d = bs_call_price_delta(S, K, T, r, sig)
            assert abs(float(p) - ref.price) < 1e-5
            assert abs(float(d) - ref.delta) < 1e-5


class TestHedgeValidity:
    def test_zero_pnl_when_realized_equals_implied(self):
        t = _run(0.20, 0.20, seed=1)
        m, se = _mean_and_se(t["pnl_pct"])
        assert abs(m) < 3 * se
        assert abs(m) < 0.001            # < 0.1% of spot, vs a ~4% premium

    def test_zero_pnl_is_independent_of_drift(self):
        """Hedged P&L must not depend on the real-world drift."""
        for drift, seed in [(-0.20, 2), (0.30, 3)]:
            S = _gbm(60000, 0.20, seed, drift=drift)
            t = simulate_delta_hedged_short_call(
                S, np.full(len(S), 0.20), r=0.03, horizon=H, step=H
            )
            m, se = _mean_and_se(t["pnl_pct"])
            assert abs(m) < 3 * se

    def test_short_vol_profits_when_implied_above_realized(self):
        t = _run(0.20, 0.30, seed=4)
        m, se = _mean_and_se(t["pnl_pct"])
        assert m > 0 and m / se > 10

    def test_short_vol_loses_when_implied_below_realized(self):
        t = _run(0.30, 0.20, seed=5)
        m, se = _mean_and_se(t["pnl_pct"])
        assert m < 0 and m / se < -10

    def test_pnl_matches_gamma_formula_trade_by_trade(self):
        """
        Per trade:  P&L ~ sum_j 0.5 * Gamma_j * S_j^2 * (sigma_imp^2 dt - ret_j^2)
        (gamma at the hedging vol, rates ~ 0). Correlation across trades must be high.
        """
        sig_imp, r = 0.25, 0.0
        S = _gbm(21 * 400, 0.35, seed=6)
        t = simulate_delta_hedged_short_call(S, np.full(len(S), sig_imp), r=r, horizon=H, step=H)

        approx = []
        for i in range(len(t)):
            s0 = i * H
            path = S[s0 : s0 + H + 1]
            K = path[0]
            total = 0.0
            for j in range(H):
                tau = (H - j) * DT
                d1 = (np.log(path[j] / K) + 0.5 * sig_imp ** 2 * tau) / (sig_imp * np.sqrt(tau))
                gamma = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (path[j] * sig_imp * np.sqrt(tau))
                ret = path[j + 1] / path[j] - 1.0
                total += 0.5 * gamma * path[j] ** 2 * (sig_imp ** 2 * DT - ret ** 2)
            approx.append(total)
        approx = np.array(approx)

        corr = np.corrcoef(approx, t["pnl"].values)[0, 1]
        assert corr > 0.97
        assert abs(np.mean(approx) - t["pnl"].mean()) < 0.1 * np.std(t["pnl"].values)


class TestMechanics:
    def test_number_of_trades_and_columns(self):
        S = _gbm(99, 0.2, seed=7)                     # 100 prices
        t = simulate_delta_hedged_short_call(S, np.full(100, 0.2), r=0.0, horizon=H, step=1)
        assert len(t) == 100 - H
        for col in ["premium", "payoff", "pnl", "pnl_pct", "iv_entry", "rv_realized", "vrp_vol", "costs"]:
            assert col in t.columns

    def test_costs_only_reduce_pnl(self):
        S = _gbm(5000, 0.2, seed=8)
        iv = np.full(len(S), 0.2)
        free = simulate_delta_hedged_short_call(S, iv, r=0.03, horizon=H, step=1, tc_bps=0.0)
        paid = simulate_delta_hedged_short_call(S, iv, r=0.03, horizon=H, step=1, tc_bps=5.0)
        assert (paid["costs"] > 0).all()
        assert (paid["pnl"] <= free["pnl"] + 1e-12).all()

    def test_series_input_keeps_dates(self):
        S = pd.Series(_gbm(99, 0.2, seed=9), index=pd.bdate_range("2020-01-01", periods=100))
        iv = pd.Series(0.2, index=S.index)
        t = simulate_delta_hedged_short_call(S, iv, r=0.01, horizon=H)
        assert t.index[0] == S.index[0]
        assert t.index[-1] == S.index[len(S) - H - 1]

    def test_input_validation(self):
        S = _gbm(50, 0.2, seed=10)
        with pytest.raises(ValueError):
            simulate_delta_hedged_short_call(S, np.full(len(S) - 1, 0.2))
        with pytest.raises(ValueError):
            simulate_delta_hedged_short_call(S, np.full(len(S), 0.2), hedge_vol="bogus")
        with pytest.raises(ValueError):
            simulate_delta_hedged_short_call(S[:10], np.full(10, 0.2), horizon=H)

    def test_realized_vol_column_matches_path(self):
        S = _gbm(5000, 0.2, seed=11)
        t = simulate_delta_hedged_short_call(S, np.full(len(S), 0.2), r=0.0, horizon=H, step=H)
        assert abs(t["rv_realized"].mean() - 0.20) < 0.01


class TestSummary:
    def test_summary_fields(self):
        t = _run(0.20, 0.30, seed=12, n=21 * 300)
        s = summarize_hedged_pnl(t, horizon=H)
        assert s["n_trades_nonoverlapping"] == len(t.iloc[::H])
        assert s["max_drawdown_pct"] <= 0
        assert 0 <= s["hit_rate"] <= 1
        assert len(s["worst_trades"]) == 5
