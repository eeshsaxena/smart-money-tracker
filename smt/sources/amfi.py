"""AMFI sources: the daily scheme master (NAVAll.txt) and the half-yearly
large / mid / small cap classification of listed stocks.

Parsing is kept in pure functions (parse_*) so it can be tested offline;
fetch_* functions only do the downloading.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime

import pandas as pd
import requests

from smt import http

NAV_ALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
# AMFI moved its newest list to the portal host in 2026, and older lists have one-off names
# (all taken from amfiindia.com/otherdata/categorisation-of-stocks, which is rendered by script).
MARKET_CAP_URLS = (
    "https://portal.amfiindia.com/spages/AverageMarketCapitalization{label}.xlsx",
    "https://www.amfiindia.com/Themes/Theme1/downloads/AverageMarketCapitalization{label}.xlsx",
)
_OLD_LIST = ("https://www.amfiindia.com/Themes/Theme1/downloads/"
             "AverageMarketCapitalizationoflistedcompaniesduringthesixmonthsended{label}.xlsx")
MARKET_CAP_ONE_OFF_URLS = {
    "31Dec2024": _OLD_LIST.format(label="31Dec2024"),
    "30Jun2024": "https://www.amfiindia.com/uploads/Average_Market_Capitalization_30_Jun2024_2a1ab4c1d8.xlsx",
    "31Dec2023": _OLD_LIST.format(label="31Dec2023"),
    "30Jun2023": _OLD_LIST.format(label="30Jun2023"),
    "31Dec2022": _OLD_LIST.format(label="31Dec2022"),
    "30Jun2022": _OLD_LIST.format(label="30June2022"),
}

_CATEGORY_RE = re.compile(r"^(Open|Close|Interval) Ended Schemes\s*\((.+)\)\s*$", re.I)


# --------------------------------------------------------------------------- #
# Scheme master
# --------------------------------------------------------------------------- #
def parse_nav_all(text: str) -> pd.DataFrame:
    """Parse NAVAll.txt.

    Current format (2026), one scheme per line:
        code;isin_growth;isin_reinvest;scheme name;plan;option;nav;date
    Header lines give the SEBI category ("Open Ended Schemes(Equity Scheme - Flexi Cap Fund)")
    and the AMC ("PPFAS Mutual Fund") for the lines below them.
    The older 6-column format (name with plan/option folded in) is also accepted.
    """
    rows = []
    category = amc = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("scheme code"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) == 1:
            m = _CATEGORY_RE.match(line)
            if m:
                category = m.group(2).strip()
            else:
                amc = line
            continue
        if len(parts) >= 8:
            code, isin, _, name, plan, option, nav, dt = parts[:8]
        elif len(parts) >= 6:
            code, isin, _, name, nav, dt = parts[:6]
            plan = "Direct Plan" if "direct" in name.lower() else "Regular Plan"
            option = "Growth" if "growth" in name.lower() else ""
        else:
            continue
        rows.append({
            "scheme_code": code, "isin": isin, "scheme_name": name, "plan": plan,
            "option": option, "nav": pd.to_numeric(nav, errors="coerce"),
            "nav_date": pd.to_datetime(dt, format="%d-%b-%Y", errors="coerce"),
            "category": category, "amc": amc,
        })
    return pd.DataFrame(rows)


def fetch_scheme_master() -> pd.DataFrame:
    return parse_nav_all(http.get(NAV_ALL_URL).text)


# --------------------------------------------------------------------------- #
# Market-cap classification (SEBI: top 100 large, 101-250 mid, rest small)
# --------------------------------------------------------------------------- #
def classification_labels(today: date | None = None, count: int = 6) -> list[str]:
    """Most recent half-year labels first, e.g. ['30Jun2026', '31Dec2025', ...]."""
    today = today or date.today()
    labels = []
    y = today.year
    for _ in range(count):
        for month, day, mon in ((12, 31, "Dec"), (6, 30, "Jun")):
            if date(y, month, day) <= today:
                labels.append(f"{day}{mon}{y}")
        y -= 1
    return labels[:count]


def parse_market_cap(xlsx: bytes) -> pd.DataFrame:
    """AMFI 'Average Market Capitalization' workbook -> isin, company, nse_symbol, cap_class."""
    raw = pd.read_excel(io.BytesIO(xlsx), header=None)
    header_row = next(
        i for i, row in raw.iterrows()
        if any(str(v).strip().upper() == "ISIN" for v in row.values)
    )
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    cols = {c.lower(): c for c in df.columns}
    cap_col = next(c for k, c in cols.items() if "categori" in k)
    name_col = next(c for k, c in cols.items() if "company" in k)
    nse_col = next((c for k, c in cols.items() if k.startswith("nse symbol")), None)
    out = pd.DataFrame({
        "isin": df["ISIN"].astype(str).str.strip(),
        "company": df[name_col].astype(str).str.strip(),
        "nse_symbol": df[nse_col].astype(str).str.strip() if nse_col else "",
        "cap_class": df[cap_col].astype(str).str.strip(),
    })
    out = out[out["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$")]
    out = out[out["cap_class"].isin(["Large Cap", "Mid Cap", "Small Cap"])]
    return out.drop_duplicates("isin").reset_index(drop=True)


def label_date(label: str) -> date:
    """'30Jun2026' -> the last day of the half-year that list covers."""
    return datetime.strptime(label, "%d%b%Y").date()


def market_cap_urls(label: str) -> tuple[str, ...]:
    if label in MARKET_CAP_ONE_OFF_URLS:
        return (MARKET_CAP_ONE_OFF_URLS[label],)
    return tuple(u.format(label=label) for u in MARKET_CAP_URLS)


def fetch_market_cap(label: str) -> pd.DataFrame:
    """Download one half-year's list.

    Raises LookupError if AMFI has not published it at any known address, and
    RuntimeError if the site itself can't be reached.
    """
    unreachable = None
    for url in market_cap_urls(label):
        try:
            resp = http.get(url)
        except requests.HTTPError:
            continue                        # not at this address
        except requests.RequestException as e:
            unreachable = e
            continue
        if resp.content[:2] == b"PK":       # AMFI answers missing files with an HTML page
            return parse_market_cap(resp.content)
    if unreachable is not None:
        raise RuntimeError(f"AMFI website unreachable: {type(unreachable).__name__}") from unreachable
    raise LookupError(f"AMFI has no market-cap list for {label}")
