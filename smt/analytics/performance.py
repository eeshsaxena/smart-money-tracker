"""Return and risk metrics from NAV series. Pure functions on pandas objects.

Conventions:
  * a NAV series is a pd.Series indexed by date (business days, gaps allowed)
  * returns over 1 year are annualised (CAGR); shorter ones are absolute
  * the start value is the last NAV on or before the start date (no look-ahead)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smt import settings

PERIODS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "3Y": 3.0, "5Y": 5.0, "10Y": 10.0}


def _start_value(nav: pd.Series, start: pd.Timestamp) -> float | None:
    before = nav.loc[:start]
    return None if before.empty else float(before.iloc[-1])


def period_return(nav: pd.Series, years: float, end: pd.Timestamp | None = None) -> float | None:
    """Return over the trailing `years`: absolute below 1 year, annualised from 1 year."""
    nav = nav.dropna()
    if nav.empty:
        return None
    end = end or nav.index[-1]
    start = end - pd.DateOffset(days=round(years * 365.25))
    if nav.index[0] > start:
        return None                                  # not enough history
    v0, v1 = _start_value(nav, start), float(nav.loc[:end].iloc[-1])
    growth = v1 / v0
    return growth ** (1 / years) - 1 if years >= 1 else growth - 1


def trailing_returns(nav: pd.Series) -> dict[str, float | None]:
    return {label: period_return(nav, yrs) for label, yrs in PERIODS.items()}


def since_inception(nav: pd.Series) -> float | None:
    nav = nav.dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25 if len(nav) > 1 else 0
    if years < 1:
        return None
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1


def calendar_year_returns(nav: pd.Series) -> pd.Series:
    """Return per calendar year (the current year is year-to-date). Partial first years are dropped."""
    nav = nav.dropna()
    year_end = nav.groupby(nav.index.year).last()
    out = year_end.pct_change().dropna()
    return out


def rolling_returns(nav: pd.Series, years: float) -> pd.Series:
    """Annualised return of every `years`-long holding period ending on each date."""
    nav = nav.dropna()
    lagged = nav.reindex(nav.index - pd.DateOffset(days=round(years * 365.25)), method="ffill")
    lagged.index = nav.index
    out = (nav / lagged) ** (1 / years) - 1
    return out.dropna()


def drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    return nav / nav.cummax() - 1


def weekly_pair(nav: pd.Series, bench: pd.Series, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Weekly returns of a fund ('f') and its benchmark ('b') on common dates.

    Weekly, not daily. A fund's NAV and an index don't reflect the same closing moment:
    foreign holdings strike on another clock and thinly traded small caps react late.
    Daily beta therefore comes out too low, which makes alpha look too high.
    """
    both = pd.concat({"f": nav, "b": bench}, axis=1, join="inner", sort=True).dropna()
    if start is not None:
        both = both.loc[start:]
    return both.resample("W-FRI").last().pct_change().dropna()


def risk_metrics(nav: pd.Series, bench: pd.Series | None = None, years: float = 3.0,
                 rf: float = settings.RISK_FREE_RATE) -> dict[str, float | None]:
    """Risk statistics over the trailing window. Benchmark-relative ones need `bench`."""
    nav = nav.dropna()
    if nav.empty:
        return {}
    start = nav.index[-1] - pd.DateOffset(days=round(years * 365.25))
    if nav.index[0] > start:
        return {"window_years": None}
    window = nav.loc[start:]
    r = window.pct_change().dropna()
    n = settings.TRADING_DAYS
    daily_rf = (1 + rf) ** (1 / n) - 1
    excess = r - daily_rf
    # Downside deviation counts every day (gains as zero). The std of the losing days alone
    # measures how similar the losses are to each other, not how much downside there is.
    downside_dev = float(np.sqrt((excess.clip(upper=0) ** 2).mean())) * np.sqrt(n)
    ann_ret = period_return(nav, years)
    vol = r.std() * np.sqrt(n)
    dd = drawdown(window).min()
    out: dict[str, float | None] = {
        "window_years": years,
        "return": ann_ret,
        "volatility": vol,
        "sharpe": (excess.mean() * n) / vol if vol > 0 else None,
        "sortino": (excess.mean() * n) / downside_dev if downside_dev > 0 else None,
        "max_drawdown": dd,
        "calmar": ann_ret / abs(dd) if ann_ret is not None and dd < 0 else None,
        "var_95_daily": float(np.percentile(r, 5)),
    }
    if bench is not None:
        both = weekly_pair(nav, bench.dropna(), start)
        if len(both) >= 26:
            beta = both["f"].cov(both["b"]) / both["b"].var()
            b_ret = period_return(bench.dropna(), years)
            out["beta"] = beta
            have_returns = ann_ret is not None and b_ret is not None
            # Jensen's alpha, annualised: fund return minus CAPM-expected return
            out["alpha"] = (ann_ret - (rf + beta * (b_ret - rf))) if have_returns else None
            up, down = both[both["b"] > 0], both[both["b"] < 0]
            out["up_capture"] = up["f"].mean() / up["b"].mean() if len(up) else None
            out["down_capture"] = down["f"].mean() / down["b"].mean() if len(down) else None
            out["correlation"] = both["f"].corr(both["b"])
            active = both["f"] - both["b"]
            te = active.std() * np.sqrt(52)
            out["active_return"] = ann_ret - b_ret if have_returns else None
            out["tracking_error"] = te
            out["information_ratio"] = active.mean() * 52 / te if te > 0 else None
    return out


def rolling_vs_index(nav: pd.Series, bench: pd.Series, years: float = 3.0) -> dict[str, float | int | None]:
    """How often the fund beat the index over every `years`-long holding period.

    A single trailing number depends on the end date; this asks the question across all of them.
    Windows overlap heavily, so `windows` is not a count of independent observations.
    """
    both = pd.concat({"f": rolling_returns(nav, years), "b": rolling_returns(bench, years)},
                     axis=1, join="inner", sort=True).dropna()
    if both.empty:
        return {"hit_rate": None, "median_active": None, "windows": 0}
    active = both["f"] - both["b"]
    return {"hit_rate": float((active > 0).mean()), "median_active": float(active.median()), "windows": len(both)}


def growth_of(navs: pd.DataFrame, start: pd.Timestamp, amount: float = 10_000) -> pd.DataFrame:
    """Value of `amount` invested at `start` in each column (common dates only)."""
    window = navs.loc[start:].dropna(how="any")
    if window.empty:
        return window
    return window / window.iloc[0] * amount


def category_rank(values: pd.Series, categories: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank within category, e.g. '2/5'. NaN values are left unranked."""
    ranks = values.groupby(categories).rank(ascending=not higher_is_better, method="min")
    sizes = values.groupby(categories).transform("count")
    return pd.Series(
        [f"{int(r)}/{int(s)}" if pd.notna(r) else "" for r, s in zip(ranks, sizes, strict=True)],
        index=values.index,
    )
