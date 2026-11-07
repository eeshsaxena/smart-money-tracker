"""SIP simulation and XIRR (annualised return of irregular cash flows)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from scipy.optimize import brentq


def xirr(flows: list[tuple[date, float]]) -> float | None:
    """Rate r such that sum(cf / (1+r)^(days/365)) = 0. Investments negative, value positive."""
    if len(flows) < 2 or not any(cf > 0 for _, cf in flows) or not any(cf < 0 for _, cf in flows):
        return None
    t0 = min(d for d, _ in flows)

    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for d, cf in flows)

    try:
        return brentq(npv, -0.9999, 100.0, maxiter=500)
    except ValueError:
        return None


@dataclass
class SipResult:
    installments: pd.DataFrame   # date, nav, amount, units, cum_units, cum_invested, value
    invested: float
    value: float
    xirr: float | None


def simulate(nav: pd.Series, monthly: float, start: pd.Timestamp, end: pd.Timestamp | None = None,
             day: int = 1) -> SipResult:
    """Invest `monthly` on the first NAV date on or after `day` of each month."""
    nav = nav.dropna().sort_index()
    end = end or nav.index[-1]
    nav = nav.loc[:end]
    rows = []
    month = pd.Timestamp(start.year, start.month, 1)
    while month <= end:
        target = month + pd.Timedelta(days=day - 1)
        after = nav.loc[target:]
        if after.empty:
            break
        d, price = after.index[0], float(after.iloc[0])
        if not rows or d > rows[-1]["date"]:          # never two instalments on one day
            rows.append({"date": d, "nav": price, "amount": monthly, "units": monthly / price})
        month += pd.offsets.MonthBegin(1)
    df = pd.DataFrame(rows, columns=["date", "nav", "amount", "units"])
    if df.empty:
        return SipResult(df, 0.0, 0.0, None)
    df["cum_units"] = df["units"].cumsum()
    df["cum_invested"] = df["amount"].cumsum()
    df["value"] = df["cum_units"] * df["nav"]
    final_nav = float(nav.iloc[-1])
    value = float(df["cum_units"].iloc[-1] * final_nav)
    flows = [(d.date(), -a) for d, a in zip(df["date"], df["amount"], strict=True)] + [(nav.index[-1].date(), value)]
    return SipResult(df, float(df["amount"].sum()), value, xirr(flows))


def value_path(nav: pd.Series, result: SipResult) -> pd.DataFrame:
    """Daily invested amount and market value of the SIP, for charting."""
    nav = nav.dropna().loc[result.installments["date"].iloc[0]:]
    inst = result.installments.set_index("date")
    units = inst["units"].reindex(nav.index, fill_value=0).cumsum()
    invested = inst["amount"].reindex(nav.index, fill_value=0).cumsum()
    return pd.DataFrame({"invested": invested, "value": units * nav})
