"""Tests for backtesting engine, efficient frontier, and Venn diagram."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.backtest import (
    backtest_smart_money_signals,
    _compute_backtest_metrics,
)
from src.analysis.efficient_frontier import (
    compute_efficient_frontier,
    optimize_portfolio,
)
from src.analysis.overlap import create_venn_diagram


@pytest.fixture
def portfolio_history():
    np.random.seed(42)
    companies = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "WIPRO", "SBIN", "LT"]
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"])
    records = []
    for date in dates:
        for comp in companies:
            records.append({
                "company": comp,
                "date": date,
                "pct_aum": np.random.uniform(2, 15),
                "market_value_lakhs": np.random.uniform(100, 1000),
                "sector": "IT" if comp in ["TCS", "INFY", "WIPRO"] else "Finance",
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    return pd.DataFrame({
        "RELIANCE": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.015, 500)),
        "TCS": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, 500)),
        "INFY": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.018, 500)),
        "HDFC": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 500)),
    }, index=dates)


class TestBacktest:
    def test_backtest_returns_results(self, portfolio_history):
        result = backtest_smart_money_signals(portfolio_history, initial_capital=100000, top_n=5)
        assert "equity_curve" in result
        assert "metrics" in result
        assert not result["equity_curve"].empty

    def test_backtest_metrics_keys(self, portfolio_history):
        result = backtest_smart_money_signals(portfolio_history, initial_capital=100000)
        m = result["metrics"]
        assert "strategy_return_pct" in m
        assert "benchmark_return_pct" in m
        assert "alpha_pct" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown_pct" in m

    def test_backtest_empty(self):
        result = backtest_smart_money_signals(pd.DataFrame())
        assert result["equity_curve"].empty

    def test_backtest_too_few_dates(self):
        df = pd.DataFrame({
            "company": ["A"], "date": [pd.Timestamp("2024-01-01")], "pct_aum": [10],
        })
        result = backtest_smart_money_signals(df)
        assert result["equity_curve"].empty

    def test_equity_curve_starts_at_capital(self, portfolio_history):
        result = backtest_smart_money_signals(portfolio_history, initial_capital=200000)
        assert result["equity_curve"].iloc[0]["equity"] == 200000

    def test_compute_metrics_direct(self):
        eq = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "equity": [100000, 105000, 110000],
            "benchmark": [100000, 103000, 106000],
        })
        m = _compute_backtest_metrics(eq, 100000)
        assert m["strategy_return_pct"] == 10.0
        assert m["benchmark_return_pct"] == 6.0
        assert m["alpha_pct"] > 0


class TestEfficientFrontier:
    def test_frontier_portfolios(self, sample_prices):
        result = compute_efficient_frontier(sample_prices, num_portfolios=100)
        assert not result["portfolios"].empty
        assert len(result["portfolios"]) == 100
        assert "return_pct" in result["portfolios"].columns
        assert "volatility_pct" in result["portfolios"].columns
        assert "sharpe_ratio" in result["portfolios"].columns

    def test_optimal_portfolio(self, sample_prices):
        result = compute_efficient_frontier(sample_prices, num_portfolios=100)
        assert "optimal" in result
        assert result["optimal"]["sharpe_ratio"] > 0

    def test_min_vol_portfolio(self, sample_prices):
        result = compute_efficient_frontier(sample_prices, num_portfolios=100)
        assert "min_vol" in result
        assert result["min_vol"]["volatility_pct"] > 0

    def test_optimize_max_sharpe(self, sample_prices):
        result = optimize_portfolio(sample_prices, "max_sharpe")
        assert "return_pct" in result
        assert "allocation" in result
        assert sum(result["allocation"].values()) <= 101

    def test_optimize_min_vol(self, sample_prices):
        result = optimize_portfolio(sample_prices, "min_vol")
        assert "volatility_pct" in result

    def test_too_few_assets(self):
        prices = pd.DataFrame({"A": [1, 2, 3]})
        result = compute_efficient_frontier(prices)
        assert result["portfolios"].empty

    def test_empty_prices(self):
        result = optimize_portfolio(pd.DataFrame(), "max_sharpe")
        assert result == {}


class TestVennDiagram:
    def test_venn_two_funds(self):
        holdings = {
            "Fund A": pd.DataFrame({"company": ["RELIANCE", "TCS", "INFY"], "pct_aum": [10, 10, 10]}),
            "Fund B": pd.DataFrame({"company": ["RELIANCE", "HDFC", "INFY"], "pct_aum": [10, 10, 10]}),
        }
        fig = create_venn_diagram(holdings)
        assert fig.layout.title.text == "Portfolio Overlap — Venn Diagram"
        assert len(fig.layout.shapes) == 2  # Two circles

    def test_venn_three_funds(self):
        holdings = {
            "Fund A": pd.DataFrame({"company": ["RELIANCE", "TCS"], "pct_aum": [10, 10]}),
            "Fund B": pd.DataFrame({"company": ["RELIANCE", "HDFC"], "pct_aum": [10, 10]}),
            "Fund C": pd.DataFrame({"company": ["RELIANCE", "WIPRO"], "pct_aum": [10, 10]}),
        }
        fig = create_venn_diagram(holdings)
        assert len(fig.layout.shapes) == 3

    def test_venn_too_many_funds(self):
        holdings = {f"Fund {i}": pd.DataFrame({"company": ["A"], "pct_aum": [10]}) for i in range(5)}
        fig = create_venn_diagram(holdings)
        assert fig.data == ()  # Empty figure

    def test_venn_one_fund(self):
        holdings = {"Fund A": pd.DataFrame({"company": ["A"], "pct_aum": [10]})}
        fig = create_venn_diagram(holdings)
        assert fig.data == ()
