"""SIP XIRR calculator benchmarking returns against peers and Nifty 50."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyxirr
import yfinance as yf

from config import BENCHMARK_TICKER


def compute_sip_xirr(
    sip_amount: float,
    nav_series: pd.Series,
    start_date: Optional[datetime] = None,
    frequency: str = "monthly",
) -> dict:
    if nav_series.empty:
        return {"xirr": None, "total_invested": 0, "current_value": 0, "units": 0}

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

    for date in sip_dates:
        closest_idx = nav_series.index.get_indexer([date], method="nearest")[0]
        nav = nav_series.iloc[closest_idx]
        if nav <= 0:
            continue

        units = sip_amount / nav
        total_units += units
        cashflows.append(-sip_amount)
        dates.append(date.to_pydatetime() if hasattr(date, "to_pydatetime") else date)

    if not cashflows:
        return {"xirr": None, "total_invested": 0, "current_value": 0, "units": 0}

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

    return {
        "xirr": round(xirr_value * 100, 2) if xirr_value else None,
        "total_invested": round(abs(sum(cashflows[:-1])), 2),
        "current_value": round(current_value, 2),
        "units": round(total_units, 4),
        "num_installments": len(cashflows) - 1,
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
        result["fund_name"] = name
        result["ticker"] = ticker
        records.append(result)

    df = pd.DataFrame(records)
    if not df.empty and "xirr" in df.columns:
        df = df.sort_values("xirr", ascending=False)
    return df
