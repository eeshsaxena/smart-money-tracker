"""Tests for analysis modules."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.overlap import compute_overlap_matrix, compute_weighted_overlap
from src.analysis.style_drift import compute_cap_allocation, detect_style_drift
from src.analysis.bhb_attribution import compute_bhb_attribution
from src.analysis.xirr_calc import compute_sip_xirr


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
    return {"Fund A": fund_a, "Fund B": fund_b}


class TestOverlap:
    def test_overlap_matrix_symmetric(self, sample_fund_holdings):
        matrix = compute_overlap_matrix(sample_fund_holdings)
        assert matrix.shape == (2, 2)
        assert matrix.iloc[0, 0] == 100.0
        assert matrix.iloc[1, 1] == 100.0
        assert matrix.iloc[0, 1] == matrix.iloc[1, 0]

    def test_overlap_matrix_values(self, sample_fund_holdings):
        matrix = compute_overlap_matrix(sample_fund_holdings)
        # Jaccard: {RELIANCE, INFY} / {RELIANCE, TCS, INFY, HDFC} = 2/4 = 50%
        assert matrix.iloc[0, 1] == 50.0

    def test_weighted_overlap(self, sample_fund_holdings):
        matrix = compute_weighted_overlap(sample_fund_holdings)
        assert matrix.iloc[0, 0] == 100.0
        assert matrix.iloc[0, 1] > 0


class TestStyleDrift:
    def test_cap_allocation_all_large(self, sample_holdings):
        alloc = compute_cap_allocation(sample_holdings)
        assert alloc["large_cap"] == 100.0
        assert alloc["mid_cap"] == 0.0

    def test_no_drift_flexi_cap(self, sample_holdings):
        result = detect_style_drift(sample_holdings, "flexi_cap")
        assert not result["drifted"]
        assert len(result["violations"]) == 0

    def test_drift_detected_mid_cap(self, sample_holdings):
        result = detect_style_drift(sample_holdings, "mid_cap")
        assert result["drifted"]
        assert any(v["category"] == "mid_cap" for v in result["violations"])


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

    def test_sip_xirr_empty(self):
        result = compute_sip_xirr(10000, pd.Series(dtype=float))
        assert result["xirr"] is None
        assert result["total_invested"] == 0
