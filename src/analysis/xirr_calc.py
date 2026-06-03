"""SIP XIRR calculator with growth visualization and peer benchmarking."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pyxirr
import yfinance as yf
import plotly.graph_objects as go

from config import BENCHMARK_TICKER, PEER_GROUPS


def compute_sip_xirr(
    sip_amount: float,
    nav_series: pd.Series,
    start_date: Optional[datetime] = None,
    frequency: str = "monthly",
) -> dict:
    if nav_series.empty:
        return {"xirr": None, "total_invested": 0, "current_value": 0, "units": 0, "num_installments": 0, "growth_data": []}

    nav_series = nav_series.sort_index()
    if start_date:
        nav_series = nav_series[nav_series.index >= start_date]

    if frequency == "monthly":
        sip_dates = nav_series.resample("MS").first().dropna().index
    elif frequency == "weekly":
        sip_dates = nav_series.resample("W-MON").first().dropna().index
    else:
        sip_dates = nav_series.index

    cashflows = []
    dates = []
    total_units = 0.0
    growth_data = []
    running_invested = 0.0

    for date in sip_dates:
        closest_idx = nav_series.index.get_indexer([date], method="nearest")[0]
        nav = nav_series.iloc[closest_idx]
        if nav <= 0:
            continue

        units = sip_amount / nav
        total_units += units
        running_invested += sip_amount
        cashflows.append(-sip_amount)
        dates.append(date.to_pydatetime() if hasattr(date, "to_pydatetime") else date)

        current_value = total_units * nav
        growth_data.append({
            "date": date,
            "invested": running_invested,
            "value": round(current_value, 2),
            "nav": nav,
            "units": round(total_units, 4),
        })

    if not cashflows:
        return {"xirr": None, "total_invested": 0, "current_value": 0, "units": 0, "num_installments": 0, "growth_data": []}

    current_nav = nav_series.iloc[-1]
    current_value = total_units * current_nav
    cashflows.append(current_value)
    dates.append(
        nav_series.index[-1].to_pydatetime()
        if hasattr(nav_series.index[-1], "to_pydatetime")
        else nav_series.index[-1]
    )

    try:
        xirr_value = pyxirr.xirr(dates, cashflows)
    except Exception:
        xirr_value = None

    gain = current_value - running_invested
    gain_pct = (gain / running_invested * 100) if running_invested > 0 else 0

    return {
        "xirr": round(xirr_value * 100, 2) if xirr_value else None,
        "total_invested": round(running_invested, 2),
        "current_value": round(current_value, 2),
        "absolute_gain": round(gain, 2),
        "absolute_gain_pct": round(gain_pct, 2),
        "units": round(total_units, 4),
        "num_installments": len(cashflows) - 1,
        "growth_data": growth_data,
    }


def fetch_nav_series(ticker: str, period: str = "5y") -> pd.Series:
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    except Exception:
        return pd.Series(dtype=float)


def benchmark_sip(
    fund_ticker: str,
    sip_amount: float = 10000,
    period: str = "5y",
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> dict:
    fund_nav = fetch_nav_series(fund_ticker, period)
    bench_nav = fetch_nav_series(benchmark_ticker, period)

    fund_result = compute_sip_xirr(sip_amount, fund_nav)
    bench_result = compute_sip_xirr(sip_amount, bench_nav)

    alpha = None
    if fund_result["xirr"] is not None and bench_result["xirr"] is not None:
        alpha = round(fund_result["xirr"] - bench_result["xirr"], 2)

    return {
        "fund": fund_result,
        "benchmark": bench_result,
        "alpha": alpha,
    }


def compare_category_peers(
    fund_tickers: dict[str, str],
    sip_amount: float = 10000,
    period: str = "5y",
) -> pd.DataFrame:
    records = []
    for name, ticker in fund_tickers.items():
        nav = fetch_nav_series(ticker, period)
        result = compute_sip_xirr(sip_amount, nav)
        result.pop("growth_data", None)
        result["fund_name"] = name
        result["ticker"] = ticker
        records.append(result)

    df = pd.DataFrame(records)
    if not df.empty and "xirr" in df.columns:
        df = df.sort_values("xirr", ascending=False)
        best_xirr = df["xirr"].max()
        df["vs_best"] = df["xirr"].apply(
            lambda x: round(x - best_xirr, 2) if pd.notna(x) else None
        )
    return df


def get_peer_group_options() -> list[dict]:
    return [{"label": name, "value": name} for name in PEER_GROUPS.keys()]


def create_sip_growth_chart(
    fund_result: dict,
    bench_result: dict,
    fund_name: str = "Fund",
) -> go.Figure:
    fig = go.Figure()

    fund_growth = fund_result.get("growth_data", [])
    bench_growth = bench_result.get("growth_data", [])

    if fund_growth:
        gdf = pd.DataFrame(fund_growth)
        fig.add_trace(go.Scatter(
            x=gdf["date"], y=gdf["value"],
            name=f"{fund_name} (Value)",
            mode="lines",
            line=dict(color="#4CAF50", width=2),
            fill="tonexty" if bench_growth else None,
        ))
        fig.add_trace(go.Scatter(
            x=gdf["date"], y=gdf["invested"],
            name="Amount Invested",
            mode="lines",
            line=dict(color="#9E9E9E", width=2, dash="dash"),
        ))

    if bench_growth:
        bdf = pd.DataFrame(bench_growth)
        fig.add_trace(go.Scatter(
            x=bdf["date"], y=bdf["value"],
            name="Nifty 50 (Value)",
            mode="lines",
            line=dict(color="#2196F3", width=2),
        ))

    fig.update_layout(
        title=f"SIP Growth: {fund_name} vs Nifty 50",
        xaxis_title="Date",
        yaxis_title="Value (INR)",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_peer_comparison_chart(peer_df: pd.DataFrame) -> go.Figure:
    if peer_df.empty:
        return go.Figure()

    peer_df = peer_df.dropna(subset=["xirr"])
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in peer_df["xirr"]]

    fig = go.Figure(go.Bar(
        x=peer_df["xirr"],
        y=peer_df["fund_name"],
        orientation="h",
        marker_color=colors,
        text=peer_df["xirr"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="SIP XIRR Comparison Across Peers",
        xaxis_title="XIRR (%)",
        height=max(350, len(peer_df) * 40),
        margin=dict(l=200),
        yaxis=dict(autorange="reversed"),
    )
    return fig
