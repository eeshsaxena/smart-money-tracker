"""Fund Manager Style Fingerprint — P/E, P/B, Herfindahl, turnover, K-Means clustering."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_cap_tilt(holdings: pd.DataFrame) -> dict[str, float]:
    if holdings.empty or "market_cap_category" not in holdings.columns:
        return {"large_cap_pct": 0, "mid_cap_pct": 0, "small_cap_pct": 0}

    total = holdings["pct_aum"].sum()
    if total == 0:
        return {"large_cap_pct": 0, "mid_cap_pct": 0, "small_cap_pct": 0}

    result = {}
    for cap in ["large_cap", "mid_cap", "small_cap"]:
        pct = holdings[holdings["market_cap_category"] == cap]["pct_aum"].sum()
        result[f"{cap}_pct"] = round(pct / total * 100, 2)
    return result


def compute_valuation_score(holdings: pd.DataFrame) -> dict[str, float]:
    result = {"avg_pe": None, "avg_pb": None, "growth_vs_value_score": None}

    if holdings.empty:
        return result

    if "pe_ratio" in holdings.columns:
        pe_vals = holdings["pe_ratio"].dropna()
        if not pe_vals.empty:
            weights = holdings.loc[pe_vals.index, "pct_aum"]
            total_w = weights.sum()
            if total_w > 0:
                result["avg_pe"] = round((pe_vals * weights).sum() / total_w, 2)

    if "pb_ratio" in holdings.columns:
        pb_vals = holdings["pb_ratio"].dropna()
        if not pb_vals.empty:
            weights = holdings.loc[pb_vals.index, "pct_aum"]
            total_w = weights.sum()
            if total_w > 0:
                result["avg_pb"] = round((pb_vals * weights).sum() / total_w, 2)

    # Higher score = more growth-oriented, lower = value
    if result["avg_pe"] is not None:
        nifty_pe = 22.0
        result["growth_vs_value_score"] = round(result["avg_pe"] / nifty_pe * 50, 1)

    return result


def compute_sector_concentration(holdings: pd.DataFrame) -> dict:
    if holdings.empty or "sector" not in holdings.columns:
        return {"herfindahl_index": 0, "top_3_sector_pct": 0, "num_sectors": 0}

    from config import SECTOR_MAP
    holdings = holdings.copy()
    holdings["sector_normalized"] = holdings["sector"].map(SECTOR_MAP).fillna(holdings["sector"])

    total = holdings["pct_aum"].sum()
    if total == 0:
        return {"herfindahl_index": 0, "top_3_sector_pct": 0, "num_sectors": 0}

    sector_weights = holdings.groupby("sector_normalized")["pct_aum"].sum() / total
    hhi = (sector_weights ** 2).sum()

    top_3 = sector_weights.nlargest(3).sum() * 100

    return {
        "herfindahl_index": round(hhi * 10000, 0),
        "top_3_sector_pct": round(top_3, 1),
        "num_sectors": len(sector_weights),
    }


def compute_turnover_ratio(
    portfolio_history: pd.DataFrame,
) -> float:
    if portfolio_history.empty or "date" not in portfolio_history.columns:
        return 0.0

    dates = sorted(portfolio_history["date"].unique())
    if len(dates) < 2:
        return 0.0

    total_turnover = 0.0
    for i in range(1, len(dates)):
        prev_stocks = set(
            portfolio_history[portfolio_history["date"] == dates[i - 1]]["company"].str.upper()
        )
        curr_stocks = set(
            portfolio_history[portfolio_history["date"] == dates[i]]["company"].str.upper()
        )
        if not prev_stocks and not curr_stocks:
            continue
        union = prev_stocks | curr_stocks
        changes = len(prev_stocks.symmetric_difference(curr_stocks))
        total_turnover += changes / len(union) if union else 0

    avg_turnover = total_turnover / (len(dates) - 1)
    return round(avg_turnover * 100, 1)


def compute_manager_fingerprint(
    holdings: pd.DataFrame,
    portfolio_history: pd.DataFrame,
    manager_name: str,
) -> dict:
    cap_tilt = compute_cap_tilt(holdings)
    valuation = compute_valuation_score(holdings)
    concentration = compute_sector_concentration(holdings)
    turnover = compute_turnover_ratio(portfolio_history)

    return {
        "manager": manager_name,
        **cap_tilt,
        **valuation,
        **concentration,
        "turnover_pct": turnover,
    }


def cluster_managers(
    fingerprints: list[dict],
    n_clusters: int = 3,
) -> pd.DataFrame:
    df = pd.DataFrame(fingerprints)
    if len(df) < n_clusters:
        df["cluster"] = 0
        df["cluster_label"] = "Group A"
        return df

    feature_cols = [
        "large_cap_pct", "mid_cap_pct", "small_cap_pct",
        "herfindahl_index", "top_3_sector_pct", "turnover_pct",
    ]

    if "growth_vs_value_score" in df.columns and df["growth_vs_value_score"].notna().sum() > 0:
        feature_cols.append("growth_vs_value_score")

    features = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)

    cluster_labels = {0: "Conservative Blue-Chip", 1: "Aggressive Growth", 2: "Balanced Diversified"}
    df["cluster_label"] = df["cluster"].map(
        lambda c: cluster_labels.get(c, f"Group {chr(65 + c)}")
    )

    if scaled.shape[1] >= 2 and len(df) >= 3:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(scaled)
        df["pca_x"] = coords[:, 0].round(3)
        df["pca_y"] = coords[:, 1].round(3)
        df["pca_var_explained"] = round(sum(pca.explained_variance_ratio_) * 100, 1)

    return df


def create_fingerprint_radar(fingerprint: dict, manager_name: str) -> go.Figure:
    categories = [
        "Large Cap Tilt", "Mid Cap Tilt", "Small Cap Tilt",
        "Concentration", "Turnover",
    ]
    values = [
        fingerprint.get("large_cap_pct", 0),
        fingerprint.get("mid_cap_pct", 0),
        fingerprint.get("small_cap_pct", 0),
        min(fingerprint.get("top_3_sector_pct", 0), 100),
        min(fingerprint.get("turnover_pct", 0) * 5, 100),
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        go.Scatterpolar(r=values, theta=categories, fill="toself", name=manager_name)
    )
    fig.update_layout(
        title=f"Investment Style — {manager_name}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=450,
    )
    return fig


def create_cluster_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    if clustered_df.empty:
        return go.Figure()

    fig = px.scatter(
        clustered_df,
        x="large_cap_pct",
        y="top_3_sector_pct",
        color="cluster_label",
        text="manager",
        size="turnover_pct",
        size_max=25,
        title="Fund Manager Clusters — Who Thinks Alike?",
        labels={
            "large_cap_pct": "Large Cap Allocation (%)",
            "top_3_sector_pct": "Top 3 Sector Concentration (%)",
            "cluster_label": "Cluster",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=500)
    return fig


def create_pca_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    if clustered_df.empty or "pca_x" not in clustered_df.columns:
        return go.Figure()

    var_pct = clustered_df["pca_var_explained"].iloc[0] if "pca_var_explained" in clustered_df.columns else 0

    fig = px.scatter(
        clustered_df,
        x="pca_x",
        y="pca_y",
        color="cluster_label",
        text="manager",
        title=f"PCA — Manager Style Space ({var_pct:.0f}% variance explained)",
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2"},
    )
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=12)
    fig.update_layout(height=450)
    return fig


def create_fingerprint_comparison(fingerprints_df: pd.DataFrame) -> go.Figure:
    if fingerprints_df.empty:
        return go.Figure()

    metrics = ["large_cap_pct", "mid_cap_pct", "small_cap_pct", "top_3_sector_pct", "turnover_pct"]
    labels = ["Large Cap %", "Mid Cap %", "Small Cap %", "Top 3 Sector %", "Turnover %"]

    fig = go.Figure()
    for _, row in fingerprints_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[row.get(m, 0) for m in metrics],
                name=row["manager"],
            )
        )

    fig.update_layout(
        title="Manager Style Comparison",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
