"""Style-drift detector: flags funds deviating from mandated market-cap category."""

import pandas as pd
import plotly.graph_objects as go

from config import CATEGORY_MANDATES


def compute_cap_allocation(holdings: pd.DataFrame) -> dict[str, float]:
    if holdings.empty or "market_cap_category" not in holdings.columns:
        return {}

    total_aum = holdings["pct_aum"].sum()
    if total_aum == 0:
        return {}

    alloc = {}
    for cap in ["large_cap", "mid_cap", "small_cap"]:
        cap_pct = holdings[holdings["market_cap_category"] == cap]["pct_aum"].sum()
        alloc[cap] = round(cap_pct / total_aum * 100, 2)

    alloc["unclassified"] = round(
        holdings[holdings["market_cap_category"] == "unclassified"]["pct_aum"].sum()
        / total_aum * 100,
        2,
    )
    return alloc


def detect_style_drift(
    holdings: pd.DataFrame,
    fund_category: str,
) -> dict:
    mandate = CATEGORY_MANDATES.get(fund_category, {})
    alloc = compute_cap_allocation(holdings)

    if not alloc or not mandate:
        return {
            "drifted": False,
            "allocation": alloc,
            "mandate": mandate,
            "violations": [],
        }

    violations = []
    for cap, min_pct in mandate.items():
        actual = alloc.get(cap.replace("_min", ""), 0)
        required = min_pct * 100
        if actual < required:
            violations.append(
                {
                    "category": cap.replace("_min", ""),
                    "required_pct": required,
                    "actual_pct": actual,
                    "shortfall_pct": round(required - actual, 2),
                }
            )

    return {
        "drifted": len(violations) > 0,
        "allocation": alloc,
        "mandate": mandate,
        "violations": violations,
    }


def compute_drift_over_time(
    portfolio_history: pd.DataFrame,
    fund_category: str,
) -> pd.DataFrame:
    if portfolio_history.empty:
        return pd.DataFrame()

    records = []
    for date, group in portfolio_history.groupby("date"):
        alloc = compute_cap_allocation(group)
        alloc["date"] = date
        result = detect_style_drift(group, fund_category)
        alloc["drifted"] = result["drifted"]
        records.append(alloc)

    return pd.DataFrame(records).sort_values("date")


def create_drift_chart(drift_history: pd.DataFrame, fund_name: str) -> go.Figure:
    if drift_history.empty:
        return go.Figure()

    fig = go.Figure()
    for cap in ["large_cap", "mid_cap", "small_cap"]:
        if cap in drift_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=drift_history["date"],
                    y=drift_history[cap],
                    name=cap.replace("_", " ").title(),
                    mode="lines+markers",
                    stackgroup="one",
                )
            )

    fig.update_layout(
        title=f"Market-Cap Allocation Over Time — {fund_name}",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, 100]),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
