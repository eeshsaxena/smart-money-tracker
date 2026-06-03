"""Portfolio overlap heatmap across funds."""

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go


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
        height=500,
    )
    return fig
