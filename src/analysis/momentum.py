"""Momentum scoring and ML signal prediction for Smart Money signals."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def compute_price_momentum(
    symbols: list[str],
    windows: dict[str, int] = None,
) -> pd.DataFrame:
    if windows is None:
        windows = {"mom_3m": 63, "mom_6m": 126, "mom_12m": 252}

    records = []
    for sym in symbols[:50]:
        try:
            data = yf.download(f"{sym}.NS", period="2y", progress=False)
            if data.empty or len(data) < 30:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            rec = {"symbol": sym, "current_price": close.iloc[-1]}
            for label, days in windows.items():
                if len(close) >= days:
                    rec[label] = round((close.iloc[-1] / close.iloc[-days] - 1) * 100, 2)
                else:
                    rec[label] = None

            if len(close) >= 20:
                rec["volatility_20d"] = round(close.pct_change().tail(20).std() * np.sqrt(252) * 100, 2)
            else:
                rec["volatility_20d"] = None

            records.append(rec)
        except Exception:
            continue

    return pd.DataFrame(records)


def compute_smart_momentum_score(
    conviction_scores: pd.DataFrame,
    momentum_df: pd.DataFrame,
) -> pd.DataFrame:
    if conviction_scores.empty or momentum_df.empty:
        return pd.DataFrame()

    conviction_scores = conviction_scores.copy()
    conviction_scores["symbol_key"] = conviction_scores["company"].str.upper().str.strip()
    momentum_df = momentum_df.copy()
    momentum_df["symbol_key"] = momentum_df["symbol"].str.upper().str.strip()

    merged = conviction_scores.merge(momentum_df, on="symbol_key", how="inner")

    if merged.empty:
        return pd.DataFrame()

    for col in ["mom_3m", "mom_6m", "mom_12m", "conviction_score"]:
        if col in merged.columns:
            vals = merged[col].dropna()
            if len(vals) > 0:
                merged[f"{col}_rank"] = merged[col].rank(pct=True, na_option="bottom")

    rank_cols = [c for c in merged.columns if c.endswith("_rank")]
    if rank_cols:
        merged["smart_momentum_score"] = merged[rank_cols].mean(axis=1).round(3) * 100
    else:
        merged["smart_momentum_score"] = merged.get("conviction_score", 0)

    return merged.sort_values("smart_momentum_score", ascending=False).reset_index(drop=True)


def build_signal_features(
    portfolio_history: pd.DataFrame,
    price_data: pd.DataFrame = None,
) -> pd.DataFrame:
    if portfolio_history.empty:
        return pd.DataFrame()

    portfolio_history = portfolio_history.sort_values("date")
    dates = sorted(portfolio_history["date"].unique())

    if len(dates) < 3:
        return pd.DataFrame()

    records = []
    for i in range(1, len(dates) - 1):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        next_date = dates[i + 1]

        prev_df = portfolio_history[portfolio_history["date"] == prev_date]
        curr_df = portfolio_history[portfolio_history["date"] == curr_date]
        next_df = portfolio_history[portfolio_history["date"] == next_date]

        for company in curr_df["company"].unique():
            curr_row = curr_df[curr_df["company"] == company]
            prev_row = prev_df[prev_df["company"] == company]
            next_row = next_df[next_df["company"] == company]

            curr_pct = curr_row["pct_aum"].values[0] if not curr_row.empty else 0
            prev_pct = prev_row["pct_aum"].values[0] if not prev_row.empty else 0
            next_pct = next_row["pct_aum"].values[0] if not next_row.empty else 0

            weight_change = curr_pct - prev_pct
            is_new = 1 if prev_row.empty else 0

            future_change = next_pct - curr_pct
            label = 1 if future_change > 0.5 else 0

            records.append({
                "company": company,
                "date": curr_date,
                "current_weight": curr_pct,
                "weight_change": weight_change,
                "is_new_entry": is_new,
                "num_months_held": len(portfolio_history[
                    (portfolio_history["company"] == company) &
                    (portfolio_history["date"] <= curr_date)
                ]["date"].unique()),
                "label": label,
            })

    return pd.DataFrame(records)


def train_signal_predictor(
    features_df: pd.DataFrame,
) -> dict:
    if features_df.empty or len(features_df) < 20:
        return {"model": None, "accuracy": None, "feature_importance": {}}

    feature_cols = ["current_weight", "weight_change", "is_new_entry", "num_months_held"]
    available = [c for c in feature_cols if c in features_df.columns]

    X = features_df[available].fillna(0)
    y = features_df["label"]

    if y.nunique() < 2:
        return {"model": None, "accuracy": None, "feature_importance": {}}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42, min_samples_leaf=5
    )

    scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y) // 5 + 1), scoring="accuracy")
    model.fit(X_scaled, y)

    importance = dict(zip(available, model.feature_importances_.round(4)))

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": round(scores.mean() * 100, 1),
        "cv_std": round(scores.std() * 100, 1),
        "feature_importance": importance,
        "features_used": available,
        "n_samples": len(features_df),
        "positive_rate": round(y.mean() * 100, 1),
    }


def predict_signals(
    model_result: dict,
    current_holdings: pd.DataFrame,
) -> pd.DataFrame:
    if not model_result.get("model") or current_holdings.empty:
        return pd.DataFrame()

    model = model_result["model"]
    scaler = model_result["scaler"]
    features = model_result["features_used"]

    X = current_holdings[features].fillna(0) if all(f in current_holdings.columns for f in features) else pd.DataFrame()
    if X.empty:
        return pd.DataFrame()

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)

    result = current_holdings[["company"]].copy()
    result["predicted_continue"] = (probs[:, 1] * 100).round(1) if probs.shape[1] > 1 else 50.0
    result["prediction"] = result["predicted_continue"].apply(
        lambda p: "LIKELY HOLD/INCREASE" if p >= 60 else "UNCERTAIN" if p >= 40 else "LIKELY REDUCE"
    )

    return result.sort_values("predicted_continue", ascending=False).reset_index(drop=True)


def create_momentum_chart(momentum_df: pd.DataFrame) -> go.Figure:
    if momentum_df.empty:
        return go.Figure()

    top = momentum_df.head(20)

    fig = go.Figure()
    for col, color, name in [
        ("mom_3m", "#2979ff", "3M"),
        ("mom_6m", "#00bfa5", "6M"),
        ("mom_12m", "#ff9100", "12M"),
    ]:
        if col in top.columns:
            fig.add_trace(go.Bar(
                x=top["symbol"],
                y=top[col],
                name=name,
                marker_color=color,
            ))

    fig.update_layout(
        title="Price Momentum — Top Conviction Stocks",
        barmode="group",
        xaxis_title="Stock",
        yaxis_title="Return (%)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_feature_importance_chart(importance: dict) -> go.Figure:
    if not importance:
        return go.Figure()

    df = pd.DataFrame([
        {"feature": k.replace("_", " ").title(), "importance": v}
        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
    ])

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#00bfa5",
    ))
    fig.update_layout(
        title="ML Feature Importance — What Drives Signal Prediction",
        xaxis_title="Importance",
        height=300,
        margin=dict(l=150),
    )
    return fig
