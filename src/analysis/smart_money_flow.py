"""Smart Money Flow Detector — aggregate accumulation/exit signals across all funds."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def aggregate_cross_fund_signals(
    all_fund_portfolios: dict[str, pd.DataFrame],
    threshold_pct: float = 0.5,
) -> pd.DataFrame:
    combined = []
    for fund_name, portfolio in all_fund_portfolios.items():
        if portfolio.empty or "date" not in portfolio.columns:
            continue
        portfolio = portfolio.copy()
        portfolio["fund_name"] = fund_name
        combined.append(portfolio)

    if not combined:
        return pd.DataFrame()

    df = pd.concat(combined, ignore_index=True)
    df = df.sort_values("date")

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    latest = dates[-1]
    previous = dates[-2]

    latest_df = df[df["date"] == latest]
    prev_df = df[df["date"] == previous]

    latest_agg = latest_df.groupby("company").agg(
        current_total_pct=("pct_aum", "sum"),
        num_funds_holding=("fund_name", "nunique"),
        funds_list=("fund_name", lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()

    prev_agg = prev_df.groupby("company").agg(
        previous_total_pct=("pct_aum", "sum"),
        prev_num_funds=("fund_name", "nunique"),
    ).reset_index()

    merged = latest_agg.merge(prev_agg, on="company", how="outer").fillna(0)
    merged["net_change_pct"] = merged["current_total_pct"] - merged["previous_total_pct"]
    merged["funds_change"] = merged["num_funds_holding"] - merged["prev_num_funds"]

    significant = merged[merged["net_change_pct"].abs() >= threshold_pct].copy()
    significant["signal"] = significant["net_change_pct"].apply(
        lambda x: "ACCUMULATE" if x > 0 else "EXIT"
    )
    significant["strength"] = significant["net_change_pct"].abs().apply(
        lambda x: "STRONG" if x >= 2.0 else "MODERATE" if x >= 1.0 else "WEAK"
    )

    new_entries = merged[
        (merged["prev_num_funds"] == 0) & (merged["num_funds_holding"] > 0)
    ].copy()
    if not new_entries.empty:
        new_entries["signal"] = "NEW ENTRY"
        new_entries["strength"] = "STRONG"
        significant = pd.concat([significant, new_entries]).drop_duplicates(subset="company")

    complete_exits = merged[
        (merged["num_funds_holding"] == 0) & (merged["prev_num_funds"] > 0)
    ].copy()
    if not complete_exits.empty:
        complete_exits["signal"] = "COMPLETE EXIT"
        complete_exits["strength"] = "STRONG"
        significant = pd.concat([significant, complete_exits]).drop_duplicates(subset="company")

    return significant.sort_values("net_change_pct", ascending=False).reset_index(drop=True)


def compute_conviction_scores(
    all_fund_portfolios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = []
    for fund_name, portfolio in all_fund_portfolios.items():
        if portfolio.empty:
            continue
        latest_date = portfolio["date"].max()
        latest = portfolio[portfolio["date"] == latest_date].copy()
        latest["fund_name"] = fund_name
        combined.append(latest)

    if not combined:
        return pd.DataFrame()

    df = pd.concat(combined, ignore_index=True)

    scores = df.groupby("company").agg(
        avg_weight_pct=("pct_aum", "mean"),
        max_weight_pct=("pct_aum", "max"),
        num_funds=("fund_name", "nunique"),
        total_value_lakhs=("market_value_lakhs", "sum"),
        top_fund=("fund_name", "first"),
    ).reset_index()

    total_funds = df["fund_name"].nunique()
    scores["breadth_score"] = (scores["num_funds"] / total_funds * 100).round(1)
    scores["conviction_score"] = (
        scores["avg_weight_pct"] * 0.4
        + scores["max_weight_pct"] * 0.3
        + scores["breadth_score"] * 0.3
    ).round(2)

    return scores.sort_values("conviction_score", ascending=False).reset_index(drop=True)


def create_flow_chart(signals: pd.DataFrame) -> go.Figure:
    if signals.empty:
        return go.Figure()

    top_acc = signals[signals["net_change_pct"] > 0].head(15)
    top_exit = signals[signals["net_change_pct"] < 0].tail(15)
    plot_data = pd.concat([top_acc, top_exit]).sort_values("net_change_pct")

    colors = ["#4CAF50" if x > 0 else "#F44336" for x in plot_data["net_change_pct"]]

    fig = go.Figure(
        go.Bar(
            x=plot_data["net_change_pct"],
            y=plot_data["company"],
            orientation="h",
            marker_color=colors,
            text=plot_data["net_change_pct"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Smart Money Flow — Top Accumulations & Exits",
        xaxis_title="Net Change in Aggregate Holding (%)",
        yaxis_title="",
        height=max(400, len(plot_data) * 28),
        margin=dict(l=200),
    )
    return fig


def create_conviction_treemap(scores: pd.DataFrame) -> go.Figure:
    if scores.empty:
        return go.Figure()

    top = scores.head(30).copy()
    top["label"] = top["company"] + "<br>" + top["num_funds"].astype(str) + " funds"

    fig = px.treemap(
        top,
        path=["label"],
        values="conviction_score",
        color="conviction_score",
        color_continuous_scale="Greens",
        title="Top 30 Stocks by Conviction Score (across all fund managers)",
    )
    fig.update_layout(height=550)
    return fig
