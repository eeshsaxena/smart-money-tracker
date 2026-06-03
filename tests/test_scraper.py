"""Tests for scraper modules."""

import pandas as pd
import pytest

from src.scraper.nse_classifier import classify_stocks, _fallback_nifty500


class TestNSEClassifier:
    def test_fallback_data_exists(self):
        df = _fallback_nifty500()
        assert not df.empty
        assert "Symbol" in df.columns
        assert "rank" in df.columns

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
