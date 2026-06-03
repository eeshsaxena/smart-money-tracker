"""Risk & return analytics — Sharpe, Sortino, max drawdown, volatility, rolling returns."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import RISK_FREE_RATE


def compute_risk_metrics(
    nav_series: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    if nav_series.empty or len(nav_series) < 20:
        return {
            "total_return_pct": None, "cagr_pct": None, "volatility_pct": None,
            "sharpe_ratio": None, "sortino_ratio": None, "max_drawdown_pct": None,
            "calmar_ratio": None, "best_day_pct": None, "worst_day_pct": None,
            "positive_days_pct": None, "var_95_pct": None,
        }

    nav_series = nav_series.sort_index().dropna()
    daily_returns = nav_series.pct_change().dropna()

    total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0

    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    daily_rf = risk_free_rate / 252

    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < daily_rf]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    var_95 = np.percentile(daily_returns, 5) * 100

    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "volatility_pct": round(annual_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 2),
        "best_day_pct": round(daily_returns.max() * 100, 2),
        "worst_day_pct": round(daily_returns.min() * 100, 2),
        "positive_days_pct": round((daily_returns > 0).mean() * 100, 1),
        "var_95_pct": round(var_95, 2),
        "years": round(years, 1),
    }


def compute_rolling_returns(
    nav_series: pd.Series,
    windows: dict[str, int] = None,
) -> pd.DataFrame:
    if nav_series.empty:
        return pd.DataFrame()

    if windows is None:
        windows = {"1Y": 252, "3Y": 756, "5Y": 1260}

    nav_series = nav_series.sort_index().dropna()
    result = pd.DataFrame(index=nav_series.index)

    for label, days in windows.items():
        if len(nav_series) >= days:
            rolling = (nav_series / nav_series.shift(days)) ** (252 / days) - 1
            result[label] = rolling * 100

    return result.dropna(how="all")


def compute_drawdown_series(nav_series: pd.Series) -> pd.Series:
    if nav_series.empty:
        return pd.Series(dtype=float)

    nav_series = nav_series.sort_index().dropna()
    cummax = nav_series.cummax()
    return ((nav_series - cummax) / cummax * 100)


def compute_monthly_returns_heatmap(nav_series: pd.Series) -> pd.DataFrame:
    if nav_series.empty:
        return pd.DataFrame()

    nav_series = nav_series.sort_index().dropna()
    monthly = nav_series.resample("ME").last()
    monthly_ret = monthly.pct_change() * 100

    heatmap_data = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "return": monthly_ret.values,
    })

    pivot = heatmap_data.pivot_table(index="year", columns="month", values="return")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    return pivot


def create_rolling_returns_chart(
    rolling_df: pd.DataFrame,
    fund_name: str = "Fund",
) -> go.Figure:
    if rolling_df.empty:
        return go.Figure()

    fig = go.Figure()
    colors = {"1Y": "#2196F3", "3Y": "#4CAF50", "5Y": "#FF9800"}

    for col in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df[col],
            name=f"{col} Rolling",
            mode="lines",
            line=dict(color=colors.get(col, "#9C27B0"), width=1.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Rolling Returns — {fund_name}",
        xaxis_title="Date",
        yaxis_title="Annualized Return (%)",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_drawdown_chart(
    nav_series: pd.Series,
    fund_name: str = "Fund",
) -> go.Figure:
    dd = compute_drawdown_series(nav_series)
    if dd.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(244, 67, 54, 0.3)",
        line=dict(color="#F44336", width=1),
        name="Drawdown",
    ))

    max_dd_idx = dd.idxmin()
    fig.add_annotation(
        x=max_dd_idx, y=dd.min(),
        text=f"Max DD: {dd.min():.1f}%",
        showarrow=True, arrowhead=2,
        font=dict(color="#F44336", size=12),
    )

    fig.update_layout(
        title=f"Underwater (Drawdown) Chart — {fund_name}",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=350,
        yaxis=dict(ticksuffix="%"),
    )
    return fig


def create_monthly_heatmap(heatmap_df: pd.DataFrame, fund_name: str = "Fund") -> go.Figure:
    if heatmap_df.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.astype(str).tolist(),
        colorscale=[
            [0, "#F44336"], [0.35, "#FFCDD2"], [0.5, "#FFFFFF"],
            [0.65, "#C8E6C9"], [1, "#4CAF50"],
        ],
        zmid=0,
        text=heatmap_df.round(1).astype(str).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Return %"),
    ))

    fig.update_layout(
        title=f"Monthly Returns Heatmap — {fund_name}",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(300, len(heatmap_df) * 35 + 100),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_risk_return_scatter(
    fund_metrics: list[dict],
) -> go.Figure:
    if not fund_metrics:
        return go.Figure()

    df = pd.DataFrame(fund_metrics)
    df = df.dropna(subset=["volatility_pct", "cagr_pct"])

    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df,
        x="volatility_pct",
        y="cagr_pct",
        text="name",
        size="sharpe_ratio",
        size_max=30,
        color="sharpe_ratio",
        color_continuous_scale="RdYlGn",
        title="Risk-Return Scatter — Higher and Left is Better",
        labels={"volatility_pct": "Volatility (%)", "cagr_pct": "CAGR (%)"},
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=500)
    return fig


def create_risk_dashboard(
    metrics: dict,
    nav_series: pd.Series,
    fund_name: str,
) -> dict:
    rolling = compute_rolling_returns(nav_series)
    monthly = compute_monthly_returns_heatmap(nav_series)

    return {
        "metrics": metrics,
        "rolling_chart": create_rolling_returns_chart(rolling, fund_name),
        "drawdown_chart": create_drawdown_chart(nav_series, fund_name),
        "monthly_heatmap": create_monthly_heatmap(monthly, fund_name),
    }
