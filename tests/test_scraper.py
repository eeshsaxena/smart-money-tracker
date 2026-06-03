"""Tests for scraper modules."""

import pandas as pd
import pytest

from src.scraper.nse_classifier import classify_stocks, _fallback_nifty500, match_holdings_to_classification
from src.scraper.value_research import get_fund_universe_metadata
from config import FUND_MANAGERS


class TestNSEClassifier:
    def test_fallback_data_exists(self):
        df = _fallback_nifty500()
        assert not df.empty
        assert "Symbol" in df.columns
        assert "Company Name" in df.columns
        assert "rank" in df.columns

    def test_fallback_has_full_names(self):
        df = _fallback_nifty500()
        reliance = df[df["Symbol"] == "RELIANCE"]
        assert not reliance.empty
        assert "Reliance" in reliance.iloc[0]["Company Name"]

    def test_classify_stocks(self):
        df = _fallback_nifty500()
        classified = classify_stocks(df)
        assert "market_cap_category" in classified.columns
        categories = classified["market_cap_category"].unique()
        assert "large_cap" in categories
        assert "mid_cap" in categories
        assert "small_cap" in categories

    def test_large_cap_range(self):
        df = _fallback_nifty500()
        classified = classify_stocks(df)
        large = classified[classified["market_cap_category"] == "large_cap"]
        assert all(large["rank"] <= 100)

    def test_mid_cap_range(self):
        df = _fallback_nifty500()
        classified = classify_stocks(df)
        mid = classified[classified["market_cap_category"] == "mid_cap"]
        assert all(mid["rank"] >= 101)
        assert all(mid["rank"] <= 250)

    def test_small_cap_range(self):
        df = _fallback_nifty500()
        classified = classify_stocks(df)
        small = classified[classified["market_cap_category"] == "small_cap"]
        assert all(small["rank"] >= 251)

    def test_match_holdings(self):
        df = _fallback_nifty500()
        classified = classify_stocks(df)
        holdings = pd.DataFrame({
            "company": ["Reliance Industries Ltd.", "Tata Consultancy Services Ltd."],
            "pct_aum": [30, 25],
        })
        merged = match_holdings_to_classification(holdings, classified)
        assert "market_cap_category" in merged.columns


class TestFundManagerConfig:
    def test_all_managers_have_funds(self):
        for key, mgr in FUND_MANAGERS.items():
            assert "name" in mgr
            assert "amc" in mgr
            assert "funds" in mgr
            assert len(mgr["funds"]) > 0

    def test_all_funds_have_required_fields(self):
        for key, mgr in FUND_MANAGERS.items():
            for fund in mgr["funds"]:
                assert "scheme_code" in fund
                assert "name" in fund
                assert "category" in fund

    def test_manager_count(self):
        assert len(FUND_MANAGERS) >= 8


class TestValueResearch:
    def test_fund_universe_metadata(self):
        df = get_fund_universe_metadata()
        assert not df.empty
        assert "manager_name" in df.columns
        assert "fund_name" in df.columns
        assert "category" in df.columns
        assert len(df) >= 20
