"""Scrape Value Research Online for fund metadata — AUM, category, manager mapping."""

from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

VR_SEARCH_URL = "https://www.valueresearchonline.com/funds/fundSelector/"
VR_FUND_URL = "https://www.valueresearchonline.com/funds/{fund_code}/fundcard/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

FUND_METADATA_CACHE: dict[str, dict] = {}


def scrape_fund_metadata(fund_name: str) -> Optional[dict]:
    if fund_name in FUND_METADATA_CACHE:
        return FUND_METADATA_CACHE[fund_name]

    try:
        resp = requests.get(
            VR_SEARCH_URL,
            params={"q": fund_name, "type": "fund"},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        results = soup.select(".fund-name a, .search-result a")
        if not results:
            return None

        fund_link = results[0].get("href", "")
        if not fund_link:
            return None

        detail_resp = requests.get(
            f"https://www.valueresearchonline.com{fund_link}",
            headers=HEADERS,
            timeout=15,
        )
        detail_resp.raise_for_status()
        detail_soup = BeautifulSoup(detail_resp.text, "lxml")

        metadata = _parse_fund_page(detail_soup, fund_name)
        FUND_METADATA_CACHE[fund_name] = metadata
        return metadata

    except requests.RequestException:
        return None


def _parse_fund_page(soup: BeautifulSoup, fund_name: str) -> dict:
    meta = {
        "fund_name": fund_name,
        "aum_cr": None,
        "category": None,
        "fund_manager": None,
        "expense_ratio": None,
        "benchmark": None,
        "rating": None,
    }

    for row in soup.select("tr, .fund-detail-row, .info-row"):
        text = row.get_text(strip=True).lower()
        cells = row.find_all(["td", "span", "div"])
        if len(cells) < 2:
            continue
        value = cells[-1].get_text(strip=True)

        if "aum" in text and "cr" in text:
            meta["aum_cr"] = _parse_number(value)
        elif "category" in text:
            meta["category"] = value
        elif "fund manager" in text or "manager" in text:
            meta["fund_manager"] = value
        elif "expense" in text:
            meta["expense_ratio"] = _parse_number(value.replace("%", ""))
        elif "benchmark" in text:
            meta["benchmark"] = value

    stars = soup.select(".star-rating, .rating")
    if stars:
        meta["rating"] = stars[0].get_text(strip=True)

    return meta


def _parse_number(s: str) -> Optional[float]:
    try:
        cleaned = s.replace(",", "").replace("₹", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def get_fund_universe_metadata() -> pd.DataFrame:
    from config import FUND_MANAGERS

    records = []
    for mgr_key, mgr in FUND_MANAGERS.items():
        for fund in mgr["funds"]:
            records.append({
                "manager_key": mgr_key,
                "manager_name": mgr["name"],
                "amc": mgr["amc"],
                "scheme_code": fund["scheme_code"],
                "fund_name": fund["name"],
                "category": fund.get("category", ""),
                "ticker": fund.get("ticker", ""),
            })
    return pd.DataFrame(records)


def enrich_with_valuation_data(
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    if holdings.empty or "Symbol" not in holdings.columns:
        return holdings

    symbols = holdings["Symbol"].dropna().unique().tolist()
    pe_data = {}
    pb_data = {}

    import yfinance as yf

    for sym in symbols[:30]:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            info = ticker.info
            pe_data[sym] = info.get("trailingPE")
            pb_data[sym] = info.get("priceToBook")
        except Exception:
            continue

    holdings["pe_ratio"] = holdings["Symbol"].map(pe_data)
    holdings["pb_ratio"] = holdings["Symbol"].map(pb_data)
    return holdings
