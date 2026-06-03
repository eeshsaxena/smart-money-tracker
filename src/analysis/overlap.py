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


def create_venn_diagram(
    fund_holdings: dict[str, pd.DataFrame],
) -> go.Figure:
    """Create a Venn-style visualization for 2-3 funds showing overlap counts."""
    names = list(fund_holdings.keys())
    if len(names) < 2 or len(names) > 3:
        return go.Figure()

    sets = {}
    for name in names:
        sets[name] = set(fund_holdings[name]["company"].str.upper().str.strip())

    fig = go.Figure()

    colors = ["rgba(41,121,255,0.35)", "rgba(0,200,83,0.35)", "rgba(255,145,0,0.35)"]
    positions = [(0.35, 0.5), (0.65, 0.5), (0.5, 0.25)] if len(names) == 3 else [(0.38, 0.5), (0.62, 0.5)]

    for i, name in enumerate(names):
        x, y = positions[i]
        fig.add_shape(
            type="circle",
            x0=x - 0.28, y0=y - 0.38, x1=x + 0.28, y1=y + 0.38,
            fillcolor=colors[i],
            line=dict(color=colors[i].replace("0.35", "0.8"), width=2),
        )
        label_y = y + 0.32 if i < 2 else y - 0.32
        fig.add_annotation(
            x=x, y=label_y,
            text=f"<b>{name}</b><br>{len(sets[name])} stocks",
            showarrow=False,
            font=dict(size=11, color="#e8eaed"),
        )

    if len(names) == 2:
        a, b = sets[names[0]], sets[names[1]]
        only_a = len(a - b)
        only_b = len(b - a)
        both = len(a & b)
        fig.add_annotation(x=0.28, y=0.5, text=f"<b>{only_a}</b><br>only", showarrow=False, font=dict(size=14, color="#2979ff"))
        fig.add_annotation(x=0.50, y=0.5, text=f"<b>{both}</b><br>shared", showarrow=False, font=dict(size=14, color="#ff1744"))
        fig.add_annotation(x=0.72, y=0.5, text=f"<b>{only_b}</b><br>only", showarrow=False, font=dict(size=14, color="#00c853"))

    elif len(names) == 3:
        a, b, c = sets[names[0]], sets[names[1]], sets[names[2]]
        abc = len(a & b & c)
        ab = len((a & b) - c)
        ac = len((a & c) - b)
        bc = len((b & c) - a)
        only_a = len(a - b - c)
        only_b = len(b - a - c)
        only_c = len(c - a - b)

        fig.add_annotation(x=0.25, y=0.55, text=f"<b>{only_a}</b>", showarrow=False, font=dict(size=13, color="#2979ff"))
        fig.add_annotation(x=0.75, y=0.55, text=f"<b>{only_b}</b>", showarrow=False, font=dict(size=13, color="#00c853"))
        fig.add_annotation(x=0.50, y=0.15, text=f"<b>{only_c}</b>", showarrow=False, font=dict(size=13, color="#ff9100"))
        fig.add_annotation(x=0.50, y=0.58, text=f"<b>{ab}</b>", showarrow=False, font=dict(size=12, color="#e8eaed"))
        fig.add_annotation(x=0.38, y=0.32, text=f"<b>{ac}</b>", showarrow=False, font=dict(size=12, color="#e8eaed"))
        fig.add_annotation(x=0.62, y=0.32, text=f"<b>{bc}</b>", showarrow=False, font=dict(size=12, color="#e8eaed"))
        fig.add_annotation(x=0.50, y=0.40, text=f"<b>{abc}</b><br>all 3", showarrow=False, font=dict(size=14, color="#ff1744"))

    fig.update_layout(
        title="Portfolio Overlap — Venn Diagram",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        height=450,
        showlegend=False,
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
