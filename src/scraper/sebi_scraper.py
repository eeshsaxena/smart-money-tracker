"""Scrape SEBI monthly portfolio disclosures for mutual fund holdings."""

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
AMFI_PORTFOLIO_URL = "https://portal.amfiindia.com/DownloadSchemeData_Po.aspx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_scheme_portfolio(
    scheme_code: str, month: int, year: int
) -> Optional[pd.DataFrame]:
    params = {
        "mession": "24",
        "mession_code": scheme_code,
        "mf": month,
        "yr": year,
        "myession": "S",
    }
    try:
        resp = requests.get(
            AMFI_PORTFOLIO_URL, params=params, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    if not table:
        return None

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 5:
            rows.append(cells[:5])

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["company", "sector", "quantity", "market_value_lakhs", "pct_aum"],
    )
    df["quantity"] = pd.to_numeric(
        df["quantity"].str.replace(",", ""), errors="coerce"
    )
    df["market_value_lakhs"] = pd.to_numeric(
        df["market_value_lakhs"].str.replace(",", ""), errors="coerce"
    )
    df["pct_aum"] = pd.to_numeric(
        df["pct_aum"].str.replace("%", "").str.replace(",", ""), errors="coerce"
    )
    df["scheme_code"] = scheme_code
    df["month"] = month
    df["year"] = year
    df["date"] = pd.to_datetime(f"{year}-{month:02d}-01")
    return df.dropna(subset=["quantity"])


def fetch_multi_month_portfolio(
    scheme_code: str, months: int = 6
) -> pd.DataFrame:
    now = datetime.now()
    frames = []
    for i in range(months):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        df = fetch_scheme_portfolio(scheme_code, m, y)
        if df is not None:
            frames.append(df)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def detect_accumulation_signals(
    portfolio_history: pd.DataFrame, threshold_pct: float = 1.0
) -> pd.DataFrame:
    if portfolio_history.empty:
        return pd.DataFrame()

    portfolio_history = portfolio_history.sort_values("date")
    pivot = portfolio_history.pivot_table(
        index="company", columns="date", values="pct_aum", aggfunc="first"
    )

    dates = sorted(pivot.columns)
    if len(dates) < 2:
        return pd.DataFrame()

    latest = dates[-1]
    previous = dates[-2]

    signals = []
    for company in pivot.index:
        curr = pivot.loc[company, latest]
        prev = pivot.loc[company, previous]
        if pd.notna(curr) and pd.notna(prev):
            change = curr - prev
            if abs(change) >= threshold_pct:
                signals.append(
                    {
                        "company": company,
                        "previous_pct": round(prev, 2),
                        "current_pct": round(curr, 2),
                        "change_pct": round(change, 2),
                        "signal": "ACCUMULATE" if change > 0 else "EXIT",
                    }
                )

    return pd.DataFrame(signals).sort_values("change_pct", ascending=False)


def fetch_nav_data() -> pd.DataFrame:
    try:
        resp = requests.get(AMFI_NAV_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()

    lines = resp.text.strip().split("\n")
    records = []
    current_category = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Scheme"):
            continue
        parts = line.split(";")
        if len(parts) == 1:
            current_category = line
        elif len(parts) >= 5:
            records.append(
                {
                    "scheme_code": parts[0].strip(),
                    "isin_growth": parts[1].strip(),
                    "isin_reinvest": parts[2].strip(),
                    "scheme_name": parts[3].strip(),
                    "nav": parts[4].strip(),
                    "date": parts[5].strip() if len(parts) > 5 else "",
                    "category": current_category,
                }
            )
    return pd.DataFrame(records)
