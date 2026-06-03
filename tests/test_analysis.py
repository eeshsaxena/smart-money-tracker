"""Tests for all analysis modules."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.overlap import compute_overlap_matrix, compute_weighted_overlap, find_common_holdings, compute_unique_holdings
from src.analysis.style_drift import compute_cap_allocation, detect_style_drift, compute_drift_score
from src.analysis.bhb_attribution import compute_bhb_attribution
from src.analysis.xirr_calc import compute_sip_xirr
from src.analysis.smart_money_flow import aggregate_cross_fund_signals, compute_conviction_scores
from src.analysis.style_fingerprint import (
    compute_cap_tilt,
    compute_sector_concentration,
    compute_turnover_ratio,
    compute_manager_fingerprint,
    cluster_managers,
)


@pytest.fixture
def sample_holdings():
    return pd.DataFrame({
        "company": ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"],
        "sector": ["Energy", "IT", "IT", "Finance", "Finance"],
        "quantity": [100, 200, 150, 300, 250],
        "market_value_lakhs": [1000, 800, 600, 500, 400],
        "pct_aum": [30, 24, 18, 15, 13],
        "market_cap_category": ["large_cap", "large_cap", "large_cap", "large_cap", "large_cap"],
    })


@pytest.fixture
def sample_fund_holdings():
    fund_a = pd.DataFrame({
        "company": ["RELIANCE", "TCS", "INFY"],
        "pct_aum": [40, 35, 25],
    })
    fund_b = pd.DataFrame({
        "company": ["RELIANCE", "HDFC", "INFY"],
        "pct_aum": [30, 40, 30],
    })
    fund_c = pd.DataFrame({
        "company": ["RELIANCE", "HDFC", "WIPRO"],
        "pct_aum": [35, 35, 30],
    })
    return {"Fund A": fund_a, "Fund B": fund_b, "Fund C": fund_c}


@pytest.fixture
def sample_portfolio_history():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01", "2024-03-01"])
    return pd.DataFrame({
        "company": ["RELIANCE", "TCS", "RELIANCE", "TCS", "RELIANCE", "WIPRO"],
        "sector": ["Energy", "IT", "Energy", "IT", "Energy", "IT"],
        "pct_aum": [30, 20, 32, 18, 35, 15],
        "market_value_lakhs": [1000, 800, 1100, 750, 1200, 600],
        "quantity": [100, 200, 110, 190, 120, 150],
        "date": dates,
        "market_cap_category": ["large_cap"] * 6,
    })


class TestOverlap:
    def test_overlap_matrix_symmetric(self, sample_fund_holdings):
        matrix = compute_overlap_matrix(sample_fund_holdings)
        assert matrix.shape == (3, 3)
        assert matrix.iloc[0, 0] == 100.0
        assert matrix.iloc[0, 1] == matrix.iloc[1, 0]

    def test_overlap_matrix_values(self, sample_fund_holdings):
        matrix = compute_overlap_matrix(sample_fund_holdings)
        # Fund A: {RELIANCE, TCS, INFY}, Fund B: {RELIANCE, HDFC, INFY}
        # Jaccard: 2/4 = 50%
        assert matrix.iloc[0, 1] == 50.0

    def test_weighted_overlap(self, sample_fund_holdings):
        matrix = compute_weighted_overlap(sample_fund_holdings)
        assert matrix.iloc[0, 0] == 100.0
        assert matrix.iloc[0, 1] > 0

    def test_find_common_holdings(self, sample_fund_holdings):
        common = find_common_holdings(sample_fund_holdings)
        assert not common.empty
        assert "RELIANCE" in common["company"].str.upper().values
        reliance = common[common["company"].str.upper() == "RELIANCE"]
        assert reliance.iloc[0]["num_funds"] == 3

    def test_unique_holdings(self, sample_fund_holdings):
        unique = compute_unique_holdings(sample_fund_holdings)
        assert "Fund A" in unique
        tcs_in_a = unique["Fund A"]["company"].str.upper().str.strip()
        assert "TCS" in tcs_in_a.values


class TestStyleDrift:
    def test_cap_allocation_all_large(self, sample_holdings):
        alloc = compute_cap_allocation(sample_holdings)
        assert alloc["large_cap"] == 100.0
        assert alloc["mid_cap"] == 0.0

    def test_no_drift_flexi_cap(self, sample_holdings):
        result = detect_style_drift(sample_holdings, "flexi_cap")
        assert not result["drifted"]

    def test_drift_detected_mid_cap(self, sample_holdings):
        result = detect_style_drift(sample_holdings, "mid_cap")
        assert result["drifted"]
        assert result["severity"] in ("critical", "warning", "minor")
        assert any(v["category"] == "mid_cap" for v in result["violations"])

    def test_drift_score(self, sample_holdings):
        history = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"),
            "large_cap": 100,
            "mid_cap": 0,
            "small_cap": 0,
        }])
        score = compute_drift_score(history, "mid_cap")
        assert score > 0


class TestBHBAttribution:
    def test_attribution_sums(self):
        sectors = ["IT", "Finance", "Energy"]
        port_w = pd.Series([0.4, 0.35, 0.25], index=sectors)
        bench_w = pd.Series([0.3, 0.4, 0.3], index=sectors)
        port_r = pd.Series([0.15, 0.10, 0.08], index=sectors)
        bench_r = pd.Series([0.12, 0.09, 0.07], index=sectors)

        df = compute_bhb_attribution(port_w, bench_w, port_r, bench_r)
        totals = df[df["sector"] == "TOTAL"].iloc[0]

        alloc = totals["allocation_effect"]
        sel = totals["selection_effect"]
        inter = totals["interaction_effect"]
        total = totals["total_effect"]
        assert abs((alloc + sel + inter) - total) < 0.01

    def test_attribution_zero_diff(self):
        sectors = ["A", "B"]
        weights = pd.Series([0.5, 0.5], index=sectors)
        returns = pd.Series([0.10, 0.10], index=sectors)

        df = compute_bhb_attribution(weights, weights, returns, returns)
        totals = df[df["sector"] == "TOTAL"].iloc[0]
        assert abs(totals["total_effect"]) < 0.01


class TestXIRRCalc:
    def test_sip_xirr_basic(self):
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        navs = pd.Series([100 + i * 2 for i in range(12)], index=dates)
        result = compute_sip_xirr(10000, navs)
        assert result["xirr"] is not None
        assert result["total_invested"] > 0
        assert result["current_value"] > 0
        assert result["num_installments"] == 12
        assert "growth_data" in result
        assert len(result["growth_data"]) == 12

    def test_sip_xirr_empty(self):
        result = compute_sip_xirr(10000, pd.Series(dtype=float))
        assert result["xirr"] is None
        assert result["total_invested"] == 0

    def test_sip_xirr_absolute_gain(self):
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        navs = pd.Series([100 + i * 5 for i in range(12)], index=dates)
        result = compute_sip_xirr(10000, navs)
        assert "absolute_gain" in result
        assert "absolute_gain_pct" in result
        assert result["absolute_gain"] > 0


class TestSmartMoneyFlow:
    def test_aggregate_signals(self, sample_portfolio_history):
        portfolios = {"Fund A": sample_portfolio_history}
        signals = aggregate_cross_fund_signals(portfolios)
        assert isinstance(signals, pd.DataFrame)

    def test_conviction_scores(self, sample_portfolio_history):
        portfolios = {"Fund A": sample_portfolio_history}
        scores = compute_conviction_scores(portfolios)
        assert isinstance(scores, pd.DataFrame)
        if not scores.empty:
            assert "conviction_score" in scores.columns

    def test_empty_portfolios(self):
        signals = aggregate_cross_fund_signals({})
        assert signals.empty


class TestStyleFingerprint:
    def test_cap_tilt(self, sample_holdings):
        tilt = compute_cap_tilt(sample_holdings)
        assert "large_cap_pct" in tilt
        assert tilt["large_cap_pct"] == 100.0

    def test_sector_concentration(self, sample_holdings):
        conc = compute_sector_concentration(sample_holdings)
        assert "herfindahl_index" in conc
        assert conc["herfindahl_index"] > 0
        assert conc["num_sectors"] > 0

    def test_turnover_ratio(self, sample_portfolio_history):
        turnover = compute_turnover_ratio(sample_portfolio_history)
        assert isinstance(turnover, float)
        assert turnover >= 0

    def test_manager_fingerprint(self, sample_holdings, sample_portfolio_history):
        fp = compute_manager_fingerprint(sample_holdings, sample_portfolio_history, "Test Manager")
        assert fp["manager"] == "Test Manager"
        assert "large_cap_pct" in fp
        assert "herfindahl_index" in fp
        assert "turnover_pct" in fp

    def test_cluster_managers(self, sample_holdings, sample_portfolio_history):
        fps = []
        for name in ["Manager A", "Manager B", "Manager C", "Manager D"]:
            fps.append(compute_manager_fingerprint(sample_holdings, sample_portfolio_history, name))
        clustered = cluster_managers(fps, n_clusters=2)
        assert "cluster" in clustered.columns
        assert "cluster_label" in clustered.columns
        assert len(clustered) == 4

    def test_cluster_fewer_than_n(self):
        fps = [{"manager": "A", "large_cap_pct": 80, "mid_cap_pct": 15, "small_cap_pct": 5,
                "herfindahl_index": 2000, "top_3_sector_pct": 60, "turnover_pct": 10}]
        clustered = cluster_managers(fps, n_clusters=3)
        assert len(clustered) == 1
        assert clustered.iloc[0]["cluster"] == 0
