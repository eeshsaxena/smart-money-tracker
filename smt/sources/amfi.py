"""AMFI sources: the daily scheme master (NAVAll.txt) and the half-yearly
large / mid / small cap classification of listed stocks.

Parsing is kept in pure functions (parse_*) so it can be tested offline;
fetch_* functions only do the downloading.
"""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd
import requests

from smt import http

NAV_ALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
MARKET_CAP_URL = "https://www.amfiindia.com/Themes/Theme1/downloads/AverageMarketCapitalization{label}.xlsx"

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


def fetch_market_cap() -> tuple[str, pd.DataFrame]:
    """Download the newest classification AMFI has published. Returns (label, table)."""
    for label in classification_labels():
        try:
            resp = http.get(MARKET_CAP_URL.format(label=label))
        except requests.HTTPError:
            continue                        # not published for this half-year: try the previous one
        except requests.RequestException as e:
            # The host itself is unreachable; trying older names would only wait longer.
            raise RuntimeError(f"AMFI website unreachable: {type(e).__name__}") from e
        if resp.content[:2] != b"PK":       # AMFI answers missing files with an HTML page
            continue
        return label, parse_market_cap(resp.content)
    raise RuntimeError("No AMFI market-cap classification found for the last three years")
