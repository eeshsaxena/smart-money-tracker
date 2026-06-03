"""NSE market-cap classification for style-drift detection."""

from typing import Optional

import pandas as pd
import yfinance as yf

from config import MARKET_CAP_THRESHOLDS


def fetch_nifty500_constituents() -> pd.DataFrame:
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv",
    }
    try:
        df = pd.read_csv(url, storage_options={"headers": headers})
        return df
    except Exception:
        return _fallback_nifty500()


def _fallback_nifty500() -> pd.DataFrame:
    large_cap = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
        "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
        "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", "HCLTECH",
    ]
    mid_cap = [
        "PERSISTENT", "COFORGE", "MPHASIS", "LTTS", "TATAELXSI",
        "BALKRISIND", "AUROPHARMA", "BIOCON", "MUTHOOTFIN", "MANAPPURAM",
        "CROMPTON", "VOLTAS", "BLUESTARCO", "KAJARIACER", "CENTURYTEX",
    ]
    small_cap = [
        "ROUTE", "HAPPSTMNDS", "MASTEK", "NUCLEUS", "RATEGAIN",
        "ECLERX", "NIITLTD", "DATAPATTNS", "NEWGEN", "INTELLECT",
    ]

    records = []
    for i, sym in enumerate(large_cap, 1):
        records.append({"Symbol": sym, "Company Name": sym, "rank": i})
    for i, sym in enumerate(mid_cap, 101):
        records.append({"Symbol": sym, "Company Name": sym, "rank": i})
    for i, sym in enumerate(small_cap, 251):
        records.append({"Symbol": sym, "Company Name": sym, "rank": i})

    return pd.DataFrame(records)


def classify_stocks(constituents: pd.DataFrame) -> pd.DataFrame:
    if "rank" not in constituents.columns:
        constituents = constituents.reset_index(drop=True)
        constituents["rank"] = constituents.index + 1

    def _classify(rank: int) -> str:
        for cap, bounds in MARKET_CAP_THRESHOLDS.items():
            if bounds["rank_start"] <= rank <= bounds["rank_end"]:
                return cap
        return "micro_cap"

    constituents["market_cap_category"] = constituents["rank"].apply(_classify)
    return constituents


def get_stock_classification() -> pd.DataFrame:
    constituents = fetch_nifty500_constituents()
    return classify_stocks(constituents)


def match_holdings_to_classification(
    holdings: pd.DataFrame,
    classification: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if classification is None:
        classification = get_stock_classification()

    classification["company_upper"] = classification["Company Name"].str.upper().str.strip()
    holdings["company_upper"] = holdings["company"].str.upper().str.strip()

    merged = holdings.merge(
        classification[["company_upper", "market_cap_category", "Symbol"]],
        on="company_upper",
        how="left",
    )
    merged["market_cap_category"] = merged["market_cap_category"].fillna("unclassified")
    merged.drop(columns=["company_upper"], inplace=True)
    return merged
