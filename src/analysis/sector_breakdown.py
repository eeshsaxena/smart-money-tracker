"""Sector breakdown analysis — pie charts, treemaps, concentration metrics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SECTOR_MAP


def normalize_sectors(holdings: pd.DataFrame) -> pd.DataFrame:
    holdings = holdings.copy()
    if "sector" in holdings.columns:
        holdings["sector_normalized"] = holdings["sector"].map(SECTOR_MAP).fillna(holdings["sector"])
    else:
        holdings["sector_normalized"] = "Unknown"
    return holdings


def compute_sector_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    holdings = normalize_sectors(holdings)
    total = holdings["pct_aum"].sum()
    if total == 0:
        return pd.DataFrame()

    weights = holdings.groupby("sector_normalized").agg(
        weight_pct=("pct_aum", "sum"),
        num_stocks=("company", "nunique"),
        top_stock=("company", "first"),
        total_value_lakhs=("market_value_lakhs", "sum"),
    ).reset_index()

    weights["weight_pct"] = (weights["weight_pct"] / total * 100).round(2)
    return weights.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def compute_sector_evolution(portfolio_history: pd.DataFrame) -> pd.DataFrame:
    if portfolio_history.empty:
        return pd.DataFrame()

    portfolio_history = normalize_sectors(portfolio_history)
    records = []

    for date, group in portfolio_history.groupby("date"):
        total = group["pct_aum"].sum()
        if total == 0:
            continue
        sector_wt = group.groupby("sector_normalized")["pct_aum"].sum() / total * 100
        for sector, wt in sector_wt.items():
            records.append({"date": date, "sector": sector, "weight_pct": round(wt, 2)})

    return pd.DataFrame(records)


def create_sector_pie(holdings: pd.DataFrame, fund_name: str = "") -> go.Figure:
    weights = compute_sector_weights(holdings)
    if weights.empty:
        return go.Figure()

    top = weights.head(10)
    other_pct = weights.iloc[10:]["weight_pct"].sum() if len(weights) > 10 else 0
    if other_pct > 0:
        top = pd.concat([top, pd.DataFrame([{
            "sector_normalized": "Others",
            "weight_pct": round(other_pct, 2),
            "num_stocks": weights.iloc[10:]["num_stocks"].sum(),
        }])], ignore_index=True)

    fig = go.Figure(go.Pie(
        labels=top["sector_normalized"],
        values=top["weight_pct"],
        textinfo="label+percent",
        textposition="inside",
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3),
    ))
    fig.update_layout(
        title=f"Sector Allocation — {fund_name}" if fund_name else "Sector Allocation",
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05),
    )
    return fig


def create_sector_treemap(holdings: pd.DataFrame, fund_name: str = "") -> go.Figure:
    holdings = normalize_sectors(holdings)
    if holdings.empty:
        return go.Figure()

    plot_df = holdings[["company", "sector_normalized", "pct_aum"]].copy()
    plot_df = plot_df[plot_df["pct_aum"] > 0].head(40)
    plot_df["label"] = plot_df["company"] + " (" + plot_df["pct_aum"].round(1).astype(str) + "%)"

    fig = px.treemap(
        plot_df,
        path=["sector_normalized", "label"],
        values="pct_aum",
        color="pct_aum",
        color_continuous_scale="Blues",
        title=f"Holdings Treemap — {fund_name}" if fund_name else "Holdings Treemap",
    )
    fig.update_layout(height=550)
    return fig


def create_sector_evolution_chart(evolution_df: pd.DataFrame, fund_name: str = "") -> go.Figure:
    if evolution_df.empty:
        return go.Figure()

    top_sectors = (
        evolution_df.groupby("sector")["weight_pct"]
        .mean()
        .nlargest(8)
        .index.tolist()
    )
    filtered = evolution_df[evolution_df["sector"].isin(top_sectors)]

    fig = px.area(
        filtered,
        x="date",
        y="weight_pct",
        color="sector",
        title=f"Sector Allocation Over Time — {fund_name}" if fund_name else "Sector Evolution",
        labels={"weight_pct": "Weight (%)", "date": "Date"},
    )
    fig.update_layout(
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def compare_sector_allocation(
    fund_holdings: dict[str, pd.DataFrame],
) -> go.Figure:
    if not fund_holdings:
        return go.Figure()

    all_sectors = set()
    fund_weights = {}

    for name, holdings in fund_holdings.items():
        weights = compute_sector_weights(holdings)
        if weights.empty:
            continue
        sector_dict = dict(zip(weights["sector_normalized"], weights["weight_pct"]))
        fund_weights[name] = sector_dict
        all_sectors |= set(sector_dict.keys())

    sorted_sectors = sorted(all_sectors)

    fig = go.Figure()
    for fund_name, weights in fund_weights.items():
        fig.add_trace(go.Bar(
            x=sorted_sectors,
            y=[weights.get(s, 0) for s in sorted_sectors],
            name=fund_name,
        ))

    fig.update_layout(
        title="Sector Allocation Comparison",
        barmode="group",
        xaxis_title="Sector",
        yaxis_title="Weight (%)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
