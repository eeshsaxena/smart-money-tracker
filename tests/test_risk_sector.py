"""Tests for risk metrics, sector breakdown, and caching."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.risk_metrics import (
    compute_risk_metrics,
    compute_rolling_returns,
    compute_drawdown_series,
    compute_monthly_returns_heatmap,
)
from src.analysis.sector_breakdown import (
    normalize_sectors,
    compute_sector_weights,
    compute_sector_evolution,
)
from src.utils.cache import _cache_key


@pytest.fixture
def sample_nav():
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500))
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_holdings():
    return pd.DataFrame({
        "company": ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"],
        "sector": ["Energy", "IT", "IT", "Finance", "Finance"],
        "pct_aum": [30, 24, 18, 15, 13],
        "market_value_lakhs": [1000, 800, 600, 500, 400],
        "market_cap_category": ["large_cap"] * 5,
    })


class TestRiskMetrics:
    def test_basic_metrics(self, sample_nav):
        m = compute_risk_metrics(sample_nav)
        assert m["cagr_pct"] is not None
        assert m["volatility_pct"] is not None
        assert m["sharpe_ratio"] is not None
        assert m["sortino_ratio"] is not None
        assert m["max_drawdown_pct"] is not None
        assert m["max_drawdown_pct"] <= 0

    def test_empty_nav(self):
        m = compute_risk_metrics(pd.Series(dtype=float))
        assert m["cagr_pct"] is None
        assert m["sharpe_ratio"] is None

    def test_short_nav(self):
        m = compute_risk_metrics(pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3)))
        assert m["cagr_pct"] is None

    def test_calmar_ratio(self, sample_nav):
        m = compute_risk_metrics(sample_nav)
        assert "calmar_ratio" in m

    def test_var(self, sample_nav):
        m = compute_risk_metrics(sample_nav)
        assert m["var_95_pct"] is not None
        assert m["var_95_pct"] < 0

    def test_win_rate(self, sample_nav):
        m = compute_risk_metrics(sample_nav)
        assert 0 <= m["positive_days_pct"] <= 100


class TestRollingReturns:
    def test_rolling_columns(self, sample_nav):
        rolling = compute_rolling_returns(sample_nav)
        assert not rolling.empty
        assert "1Y" in rolling.columns

    def test_empty(self):
        rolling = compute_rolling_returns(pd.Series(dtype=float))
        assert rolling.empty


class TestDrawdown:
    def test_drawdown_negative(self, sample_nav):
        dd = compute_drawdown_series(sample_nav)
        assert not dd.empty
        assert dd.min() <= 0
        assert dd.max() <= 0.001

    def test_empty(self):
        dd = compute_drawdown_series(pd.Series(dtype=float))
        assert dd.empty


class TestMonthlyHeatmap:
    def test_heatmap_shape(self, sample_nav):
        hm = compute_monthly_returns_heatmap(sample_nav)
        assert not hm.empty
        assert "Jan" in hm.columns or "Feb" in hm.columns


class TestSectorBreakdown:
    def test_normalize_sectors(self, sample_holdings):
        result = normalize_sectors(sample_holdings)
        assert "sector_normalized" in result.columns
        assert "Financials" in result["sector_normalized"].values

    def test_compute_weights(self, sample_holdings):
        weights = compute_sector_weights(sample_holdings)
        assert not weights.empty
        assert "weight_pct" in weights.columns
        assert abs(weights["weight_pct"].sum() - 100) < 1

    def test_evolution(self, sample_holdings):
        history = pd.concat([
            sample_holdings.assign(date=pd.Timestamp("2024-01-01")),
            sample_holdings.assign(date=pd.Timestamp("2024-02-01")),
        ])
        evo = compute_sector_evolution(history)
        assert not evo.empty
        assert "sector" in evo.columns


class TestCache:
    def test_cache_key_deterministic(self):
        k1 = _cache_key("test", "a", "b")
        k2 = _cache_key("test", "a", "b")
        assert k1 == k2

    def test_cache_key_unique(self):
        k1 = _cache_key("test", "a")
        k2 = _cache_key("test", "b")
        assert k1 != k2
