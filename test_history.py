"""
Unit Tests — Data loader
=========================
The live Yahoo download cannot run in CI, so these tests cover the parts
that most often break: yfinance's changing column layouts, the CSV cache
path, and the synthetic generator.
"""

import numpy as np
import pandas as pd

from history import _extract_close, fetch_history, synthetic_history


def _idx(n=5):
    return pd.date_range("2024-01-02", periods=n, freq="B", tz="America/New_York")


def test_extract_close_flat_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0, np.nan, 4.0], "Open": [1, 1, 1, 1]}, index=_idx(4))
    s = _extract_close(df, "SPY")
    assert s.name == "SPY" and len(s) == 3
    assert s.index.tz is None


def test_extract_close_multiindex_columns():
    """Recent yfinance returns (field, ticker) MultiIndex columns even for one symbol."""
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["^VIX"]], names=["Price", "Ticker"])
    df = pd.DataFrame(np.arange(10.0).reshape(5, 2), index=_idx(5), columns=cols)
    s = _extract_close(df, "^VIX")
    assert s.name == "^VIX" and len(s) == 5
    assert s.iloc[0] == 0.0


def test_fetch_history_uses_cache(tmp_path):
    path = tmp_path / "h.csv"
    idx = pd.bdate_range("2020-01-01", periods=10)
    pd.DataFrame({"spy": 100.0, "vix": 20.0, "rate": 0.01}, index=idx).to_csv(path)
    out = fetch_history("2020-01-01", cache_path=str(path))      # no network needed
    assert list(out.columns) == ["spy", "vix", "rate"]
    assert len(out) == 10 and isinstance(out.index, pd.DatetimeIndex)


def test_fetch_history_respects_start_and_end(tmp_path):
    path = tmp_path / "h.csv"
    idx = pd.bdate_range("2020-01-01", periods=60)
    pd.DataFrame({"spy": 100.0, "vix": 20.0, "rate": 0.01}, index=idx).to_csv(path)
    out = fetch_history("2020-01-15", end="2020-02-14", cache_path=str(path))
    assert out.index[0] >= pd.Timestamp("2020-01-15")
    assert out.index[-1] <= pd.Timestamp("2020-02-14")


def test_synthetic_history_shape_and_reproducibility():
    a, b = synthetic_history(n=500, seed=3), synthetic_history(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["spy", "vix", "rate"]
    assert (a["spy"] > 0).all() and (a["vix"] > 0).all()
