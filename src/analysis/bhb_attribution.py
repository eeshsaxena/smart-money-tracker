"""Brinson-Hood-Beebower attribution model.

Decomposes portfolio alpha into allocation, selection, and interaction effects.
"""

import pandas as pd
import plotly.graph_objects as go


def compute_bhb_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    sectors = portfolio_weights.index.union(benchmark_weights.index)

    records = []
    for sector in sectors:
        wp = portfolio_weights.get(sector, 0)
        wb = benchmark_weights.get(sector, 0)
        rp = portfolio_returns.get(sector, 0)
        rb = benchmark_returns.get(sector, 0)

        allocation = (wp - wb) * rb
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        total = allocation + selection + interaction

        records.append(
            {
                "sector": sector,
                "portfolio_weight": round(wp * 100, 2),
                "benchmark_weight": round(wb * 100, 2),
                "portfolio_return": round(rp * 100, 2),
                "benchmark_return": round(rb * 100, 2),
                "allocation_effect": round(allocation * 100, 4),
                "selection_effect": round(selection * 100, 4),
                "interaction_effect": round(interaction * 100, 4),
                "total_effect": round(total * 100, 4),
            }
        )

    df = pd.DataFrame(records)

    totals = {
        "sector": "TOTAL",
        "portfolio_weight": round(df["portfolio_weight"].sum(), 2),
        "benchmark_weight": round(df["benchmark_weight"].sum(), 2),
        "portfolio_return": None,
        "benchmark_return": None,
        "allocation_effect": round(df["allocation_effect"].sum(), 4),
        "selection_effect": round(df["selection_effect"].sum(), 4),
        "interaction_effect": round(df["interaction_effect"].sum(), 4),
        "total_effect": round(df["total_effect"].sum(), 4),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df


def attribution_from_holdings(
    fund_holdings: pd.DataFrame,
    benchmark_holdings: pd.DataFrame,
    fund_returns_by_sector: pd.Series,
    benchmark_returns_by_sector: pd.Series,
) -> pd.DataFrame:
    port_weights = (
        fund_holdings.groupby("sector")["pct_aum"]
        .sum()
        .div(fund_holdings["pct_aum"].sum())
    )
    bench_weights = (
        benchmark_holdings.groupby("sector")["pct_aum"]
        .sum()
        .div(benchmark_holdings["pct_aum"].sum())
    )

    return compute_bhb_attribution(
        port_weights,
        bench_weights,
        fund_returns_by_sector,
        benchmark_returns_by_sector,
    )


def create_attribution_chart(attribution_df: pd.DataFrame) -> go.Figure:
    df = attribution_df[attribution_df["sector"] != "TOTAL"].copy()

    fig = go.Figure()
    for effect, color in [
        ("allocation_effect", "#2196F3"),
        ("selection_effect", "#4CAF50"),
        ("interaction_effect", "#FF9800"),
    ]:
        fig.add_trace(
            go.Bar(
                x=df["sector"],
                y=df[effect],
                name=effect.replace("_", " ").title(),
                marker_color=color,
            )
        )

    fig.update_layout(
        title="Brinson-Hood-Beebower Attribution by Sector",
        xaxis_title="Sector",
        yaxis_title="Effect (%)",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_attribution_waterfall(attribution_df: pd.DataFrame) -> go.Figure:
    totals = attribution_df[attribution_df["sector"] == "TOTAL"].iloc[0]

    measures = ["relative", "relative", "relative", "total"]
    labels = ["Allocation", "Selection", "Interaction", "Total Alpha"]
    values = [
        totals["allocation_effect"],
        totals["selection_effect"],
        totals["interaction_effect"],
        totals["total_effect"],
    ]

    fig = go.Figure(
        go.Waterfall(
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#4CAF50"}},
            decreasing={"marker": {"color": "#F44336"}},
            totals={"marker": {"color": "#2196F3"}},
        )
    )

    fig.update_layout(
        title="Alpha Decomposition (BHB)",
        yaxis_title="Effect (%)",
        height=400,
    )
    return fig
