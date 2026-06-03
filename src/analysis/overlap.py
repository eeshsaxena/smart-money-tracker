"""Portfolio overlap heatmap and common holdings analysis across funds."""

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


def compute_overlap_matrix(
    fund_holdings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fund_names = list(fund_holdings.keys())
    n = len(fund_names)
    matrix = pd.DataFrame(0.0, index=fund_names, columns=fund_names)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix.iloc[i, j] = 100.0
                continue
            stocks_i = set(fund_holdings[fund_names[i]]["company"].str.upper())
            stocks_j = set(fund_holdings[fund_names[j]]["company"].str.upper())
            if not stocks_i or not stocks_j:
                continue
            overlap = len(stocks_i & stocks_j)
            union = len(stocks_i | stocks_j)
            matrix.iloc[i, j] = round((overlap / union) * 100, 1)

    return matrix


def compute_weighted_overlap(
    fund_holdings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fund_names = list(fund_holdings.keys())
    n = len(fund_names)
    matrix = pd.DataFrame(0.0, index=fund_names, columns=fund_names)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix.iloc[i, j] = 100.0
                continue

            df_i = fund_holdings[fund_names[i]].copy()
            df_j = fund_holdings[fund_names[j]].copy()
            df_i["key"] = df_i["company"].str.upper()
            df_j["key"] = df_j["company"].str.upper()

            common = set(df_i["key"]) & set(df_j["key"])
            if not common:
                continue

            wt_i = df_i[df_i["key"].isin(common)]["pct_aum"].sum()
            wt_j = df_j[df_j["key"].isin(common)]["pct_aum"].sum()
            matrix.iloc[i, j] = round((wt_i + wt_j) / 2, 1)

    return matrix


def find_common_holdings(
    fund_holdings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    all_stocks = {}
    for fund_name, holdings in fund_holdings.items():
        for _, row in holdings.iterrows():
            key = row["company"].upper().strip()
            if key not in all_stocks:
                all_stocks[key] = {"company": row["company"], "funds": {}, "total_weight": 0}
            all_stocks[key]["funds"][fund_name] = row.get("pct_aum", 0)
            all_stocks[key]["total_weight"] += row.get("pct_aum", 0)

    records = []
    for key, info in all_stocks.items():
        if len(info["funds"]) >= 2:
            records.append({
                "company": info["company"],
                "num_funds": len(info["funds"]),
                "avg_weight": round(info["total_weight"] / len(info["funds"]), 2),
                "total_weight": round(info["total_weight"], 2),
                "held_by": ", ".join(sorted(info["funds"].keys())),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("num_funds", ascending=False)
    return df


def compute_unique_holdings(
    fund_holdings: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    all_keys_per_fund = {}
    for name, df in fund_holdings.items():
        all_keys_per_fund[name] = set(df["company"].str.upper().str.strip())

    result = {}
    for name, keys in all_keys_per_fund.items():
        other_keys = set()
        for other_name, other_k in all_keys_per_fund.items():
            if other_name != name:
                other_keys |= other_k
        unique = keys - other_keys
        df = fund_holdings[name]
        df = df.copy()
        df["_key"] = df["company"].str.upper().str.strip()
        result[name] = df[df["_key"].isin(unique)].drop(columns=["_key"])

    return result


def create_overlap_heatmap(
    matrix: pd.DataFrame, title: str = "Portfolio Overlap (%)"
) -> go.Figure:
    fig = ff.create_annotated_heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        annotation_text=matrix.round(1).astype(str).values,
        colorscale="RdYlGn",
        showscale=True,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Fund",
        yaxis_title="Fund",
        height=max(400, len(matrix) * 50 + 100),
    )
    return fig


def create_common_holdings_chart(common_df: pd.DataFrame) -> go.Figure:
    if common_df.empty:
        return go.Figure()

    top = common_df.head(20)
    fig = go.Figure(
        go.Bar(
            x=top["avg_weight"],
            y=top["company"],
            orientation="h",
            marker_color=top["num_funds"].apply(
                lambda n: "#F44336" if n >= 4 else "#FF9800" if n >= 3 else "#4CAF50"
            ),
            text=top["num_funds"].apply(lambda n: f"{n} funds"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Most Overlapping Holdings (held by 2+ funds)",
        xaxis_title="Average Weight (%)",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(top) * 28),
        margin=dict(l=200),
    )
    return fig


def create_overlap_sunburst(
    fund_holdings: dict[str, pd.DataFrame],
) -> go.Figure:
    labels, parents, values = [], [], []
    labels.append("All Funds")
    parents.append("")
    values.append(0)

    for fund_name, holdings in fund_holdings.items():
        labels.append(fund_name)
        parents.append("All Funds")
        values.append(holdings["pct_aum"].sum())

        for _, row in holdings.nlargest(10, "pct_aum").iterrows():
            labels.append(f"{row['company']}")
            parents.append(fund_name)
            values.append(row["pct_aum"])

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
    ))
    fig.update_layout(
        title="Fund Holdings Breakdown (Top 10 per Fund)",
        height=550,
    )
    return fig
