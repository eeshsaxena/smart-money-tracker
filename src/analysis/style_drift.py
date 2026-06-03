"""Style-drift detector: flags funds deviating from mandated market-cap category.

Tracks 24-month allocation history and flags SEBI mandate violations.
"""

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
            "severity": "none",
        }

    violations = []
    max_shortfall = 0
    for cap, min_pct in mandate.items():
        actual = alloc.get(cap.replace("_min", ""), 0)
        required = min_pct * 100
        if actual < required:
            shortfall = round(required - actual, 2)
            max_shortfall = max(max_shortfall, shortfall)
            violations.append(
                {
                    "category": cap.replace("_min", ""),
                    "required_pct": required,
                    "actual_pct": actual,
                    "shortfall_pct": shortfall,
                }
            )

    severity = "none"
    if violations:
        if max_shortfall >= 15:
            severity = "critical"
        elif max_shortfall >= 5:
            severity = "warning"
        else:
            severity = "minor"

    return {
        "drifted": len(violations) > 0,
        "allocation": alloc,
        "mandate": mandate,
        "violations": violations,
        "severity": severity,
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
        alloc["severity"] = result["severity"]
        records.append(alloc)

    return pd.DataFrame(records).sort_values("date")


def compute_drift_score(drift_history: pd.DataFrame, fund_category: str) -> float:
    if drift_history.empty:
        return 0.0

    mandate = CATEGORY_MANDATES.get(fund_category, {})
    if not mandate:
        return 0.0

    scores = []
    for _, row in drift_history.iterrows():
        month_score = 0
        for cap, min_pct in mandate.items():
            actual = row.get(cap.replace("_min", ""), 0)
            required = min_pct * 100
            if actual < required:
                month_score += (required - actual) ** 2
        scores.append(month_score ** 0.5)

    return round(sum(scores) / len(scores), 2) if scores else 0.0


def scan_all_funds_for_drift(
    all_portfolios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for fund_name, portfolio in all_portfolios.items():
        if portfolio.empty:
            continue

        category = portfolio["fund_category"].iloc[0] if "fund_category" in portfolio.columns else ""
        if not category:
            continue

        latest = portfolio[portfolio["date"] == portfolio["date"].max()]
        result = detect_style_drift(latest, category)

        records.append({
            "fund_name": fund_name,
            "category": category,
            "drifted": result["drifted"],
            "severity": result["severity"],
            "num_violations": len(result["violations"]),
            **result["allocation"],
        })

    return pd.DataFrame(records)


def create_drift_chart(drift_history: pd.DataFrame, fund_name: str) -> go.Figure:
    if drift_history.empty:
        return go.Figure()

    fig = go.Figure()

    colors = {"large_cap": "#2196F3", "mid_cap": "#FF9800", "small_cap": "#4CAF50"}
    for cap, color in colors.items():
        if cap in drift_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=drift_history["date"],
                    y=drift_history[cap],
                    name=cap.replace("_", " ").title(),
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    stackgroup="one",
                )
            )

    if "drifted" in drift_history.columns:
        drift_dates = drift_history[drift_history["drifted"]]["date"]
        for d in drift_dates:
            fig.add_vline(x=d, line_dash="dash", line_color="red", opacity=0.3)

    fig.update_layout(
        title=f"Market-Cap Allocation Over Time — {fund_name}",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, 100]),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_mandate_gauge(allocation: dict, category: str) -> go.Figure:
    mandate = CATEGORY_MANDATES.get(category, {})
    if not mandate:
        return go.Figure()

    fig = go.Figure()
    items = list(mandate.items())

    for i, (cap, min_pct) in enumerate(items):
        cap_name = cap.replace("_min", "")
        actual = allocation.get(cap_name, 0)
        required = min_pct * 100

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=actual,
            delta={"reference": required, "relative": False},
            title={"text": cap_name.replace("_", " ").title()},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50" if actual >= required else "#F44336"},
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": required,
                },
            },
            domain={"row": 0, "column": i},
        ))

    fig.update_layout(
        grid={"rows": 1, "columns": len(items), "pattern": "independent"},
        height=300,
        title=f"Mandate Compliance — {category.replace('_', ' ').title()}",
    )
    return fig
