"""Tests for momentum scoring and ML signal prediction."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.momentum import (
    build_signal_features,
    train_signal_predictor,
    predict_signals,
    compute_smart_momentum_score,
)


@pytest.fixture
def portfolio_history():
    np.random.seed(42)
    companies = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "WIPRO", "SBIN", "LT", "MARUTI", "TITAN"]
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"])
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
def conviction_scores():
    return pd.DataFrame({
        "company": ["RELIANCE", "TCS", "INFY"],
        "conviction_score": [85.5, 72.3, 68.1],
        "avg_weight_pct": [12, 8, 6],
        "max_weight_pct": [18, 12, 10],
        "num_funds": [8, 6, 5],
    })


@pytest.fixture
def momentum_data():
    return pd.DataFrame({
        "symbol": ["RELIANCE", "TCS", "INFY"],
        "mom_3m": [15.2, 8.5, -2.1],
        "mom_6m": [22.1, 12.3, 5.4],
        "mom_12m": [35.0, 18.7, 10.2],
        "current_price": [2800, 3500, 1600],
        "volatility_20d": [18.5, 22.1, 15.3],
    })


class TestBuildFeatures:
    def test_feature_columns(self, portfolio_history):
        features = build_signal_features(portfolio_history)
        assert not features.empty
        assert "current_weight" in features.columns
        assert "weight_change" in features.columns
        assert "is_new_entry" in features.columns
        assert "num_months_held" in features.columns
        assert "label" in features.columns

    def test_label_binary(self, portfolio_history):
        features = build_signal_features(portfolio_history)
        assert set(features["label"].unique()).issubset({0, 1})

    def test_empty_history(self):
        features = build_signal_features(pd.DataFrame())
        assert features.empty

    def test_insufficient_dates(self):
        df = pd.DataFrame({
            "company": ["A", "B"],
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "pct_aum": [10, 20],
        })
        features = build_signal_features(df)
        assert features.empty


class TestTrainPredictor:
    def test_model_trains(self, portfolio_history):
        features = build_signal_features(portfolio_history)
        result = train_signal_predictor(features)
        assert result["model"] is not None
        assert result["accuracy"] is not None
        assert result["accuracy"] >= 0
        assert "feature_importance" in result
        assert len(result["feature_importance"]) > 0

    def test_empty_features(self):
        result = train_signal_predictor(pd.DataFrame())
        assert result["model"] is None
        assert result["accuracy"] is None

    def test_cv_score_reasonable(self, portfolio_history):
        features = build_signal_features(portfolio_history)
        result = train_signal_predictor(features)
        if result["accuracy"] is not None:
            assert 0 <= result["accuracy"] <= 100


class TestPredictSignals:
    def test_predictions(self, portfolio_history):
        features = build_signal_features(portfolio_history)
        model_result = train_signal_predictor(features)
        latest = features[features["date"] == features["date"].max()]
        preds = predict_signals(model_result, latest)
        if not preds.empty:
            assert "predicted_continue" in preds.columns
            assert "prediction" in preds.columns
            assert all(0 <= p <= 100 for p in preds["predicted_continue"])

    def test_no_model(self, portfolio_history):
        preds = predict_signals({"model": None}, portfolio_history)
        assert preds.empty


class TestSmartMomentum:
    def test_combined_score(self, conviction_scores, momentum_data):
        result = compute_smart_momentum_score(conviction_scores, momentum_data)
        if not result.empty:
            assert "smart_momentum_score" in result.columns

    def test_empty_inputs(self):
        result = compute_smart_momentum_score(pd.DataFrame(), pd.DataFrame())
        assert result.empty
