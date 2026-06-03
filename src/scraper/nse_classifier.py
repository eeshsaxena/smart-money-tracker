"""NSE/BSE market-cap classification for style-drift detection."""

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
        ("RELIANCE", "Reliance Industries Ltd."),
        ("TCS", "Tata Consultancy Services Ltd."),
        ("HDFCBANK", "HDFC Bank Ltd."),
        ("INFY", "Infosys Ltd."),
        ("ICICIBANK", "ICICI Bank Ltd."),
        ("HINDUNILVR", "Hindustan Unilever Ltd."),
        ("ITC", "ITC Ltd."),
        ("SBIN", "State Bank of India"),
        ("BHARTIARTL", "Bharti Airtel Ltd."),
        ("BAJFINANCE", "Bajaj Finance Ltd."),
        ("KOTAKBANK", "Kotak Mahindra Bank Ltd."),
        ("LT", "Larsen & Toubro Ltd."),
        ("AXISBANK", "Axis Bank Ltd."),
        ("ASIANPAINT", "Asian Paints Ltd."),
        ("MARUTI", "Maruti Suzuki India Ltd."),
        ("TITAN", "Titan Company Ltd."),
        ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd."),
        ("ULTRACEMCO", "UltraTech Cement Ltd."),
        ("WIPRO", "Wipro Ltd."),
        ("HCLTECH", "HCL Technologies Ltd."),
        ("TATAMOTORS", "Tata Motors Ltd."),
        ("NTPC", "NTPC Ltd."),
        ("POWERGRID", "Power Grid Corporation of India Ltd."),
        ("ADANIGREEN", "Adani Green Energy Ltd."),
        ("ADANIENT", "Adani Enterprises Ltd."),
        ("M&M", "Mahindra & Mahindra Ltd."),
        ("BAJAJFINSV", "Bajaj Finserv Ltd."),
        ("TECHM", "Tech Mahindra Ltd."),
        ("ONGC", "Oil & Natural Gas Corporation Ltd."),
        ("JSWSTEEL", "JSW Steel Ltd."),
        ("TATASTEEL", "Tata Steel Ltd."),
        ("HDFCLIFE", "HDFC Life Insurance Company Ltd."),
        ("DRREDDY", "Dr. Reddy's Laboratories Ltd."),
        ("DIVISLAB", "Divi's Laboratories Ltd."),
        ("CIPLA", "Cipla Ltd."),
        ("GRASIM", "Grasim Industries Ltd."),
        ("BRITANNIA", "Britannia Industries Ltd."),
        ("SBILIFE", "SBI Life Insurance Company Ltd."),
        ("NESTLEIND", "Nestle India Ltd."),
        ("COALINDIA", "Coal India Ltd."),
    ]
    mid_cap = [
        ("PERSISTENT", "Persistent Systems Ltd."),
        ("COFORGE", "Coforge Ltd."),
        ("MPHASIS", "Mphasis Ltd."),
        ("LTTS", "L&T Technology Services Ltd."),
        ("TATAELXSI", "Tata Elxsi Ltd."),
        ("BALKRISIND", "Balkrishna Industries Ltd."),
        ("AUROPHARMA", "Aurobindo Pharma Ltd."),
        ("BIOCON", "Biocon Ltd."),
        ("MUTHOOTFIN", "Muthoot Finance Ltd."),
        ("MANAPPURAM", "Manappuram Finance Ltd."),
        ("CROMPTON", "Crompton Greaves Consumer Electricals Ltd."),
        ("VOLTAS", "Voltas Ltd."),
        ("BLUESTARCO", "Blue Star Ltd."),
        ("KAJARIACER", "Kajaria Ceramics Ltd."),
        ("CENTURYTEX", "Century Textiles & Industries Ltd."),
        ("FEDERALBNK", "Federal Bank Ltd."),
        ("IDFCFIRSTB", "IDFC First Bank Ltd."),
        ("PAGEIND", "Page Industries Ltd."),
        ("ASTRAL", "Astral Ltd."),
        ("SUNTV", "Sun TV Network Ltd."),
        ("CUMMINSIND", "Cummins India Ltd."),
        ("ESCORTS", "Escorts Kubota Ltd."),
        ("TRENT", "Trent Ltd."),
        ("ZYDUSLIFE", "Zydus Lifesciences Ltd."),
        ("JUBLFOOD", "Jubilant FoodWorks Ltd."),
    ]
    small_cap = [
        ("ROUTE", "Route Mobile Ltd."),
        ("HAPPSTMNDS", "Happiest Minds Technologies Ltd."),
        ("MASTEK", "Mastek Ltd."),
        ("NUCLEUS", "Nucleus Software Exports Ltd."),
        ("RATEGAIN", "RateGain Travel Technologies Ltd."),
        ("ECLERX", "eClerx Services Ltd."),
        ("NIITLTD", "NIIT Ltd."),
        ("DATAPATTNS", "Data Patterns (India) Ltd."),
        ("NEWGEN", "Newgen Software Technologies Ltd."),
        ("INTELLECT", "Intellect Design Arena Ltd."),
        ("GPPL", "Gujarat Pipavav Port Ltd."),
        ("TANLA", "Tanla Platforms Ltd."),
        ("IIFL", "IIFL Finance Ltd."),
        ("MAHLIFE", "Mahindra Lifespace Developers Ltd."),
        ("CLEAN", "Clean Science and Technology Ltd."),
        ("LXCHEM", "Laxmi Organic Industries Ltd."),
        ("GRINDWELL", "Grindwell Norton Ltd."),
        ("FINEORG", "Fine Organic Industries Ltd."),
        ("KPITTECH", "KPIT Technologies Ltd."),
        ("SONATSOFTW", "Sonata Software Ltd."),
    ]

    records = []
    for i, (sym, name) in enumerate(large_cap, 1):
        records.append({"Symbol": sym, "Company Name": name, "rank": i, "Industry": "Various"})
    for i, (sym, name) in enumerate(mid_cap, 101):
        records.append({"Symbol": sym, "Company Name": name, "rank": i, "Industry": "Various"})
    for i, (sym, name) in enumerate(small_cap, 251):
        records.append({"Symbol": sym, "Company Name": name, "rank": i, "Industry": "Various"})

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

    classification = classification.copy()
    classification["company_upper"] = classification["Company Name"].str.upper().str.strip()
    holdings = holdings.copy()
    holdings["company_upper"] = holdings["company"].str.upper().str.strip()

    merged = holdings.merge(
        classification[["company_upper", "market_cap_category", "Symbol"]],
        on="company_upper",
        how="left",
    )
    merged["market_cap_category"] = merged["market_cap_category"].fillna("unclassified")
    merged.drop(columns=["company_upper"], inplace=True)
    return merged


def fetch_stock_fundamentals(symbols: list[str], max_stocks: int = 30) -> pd.DataFrame:
    records = []
    for sym in symbols[:max_stocks]:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            info = ticker.info
            records.append({
                "Symbol": sym,
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap_cr": (info.get("marketCap", 0) or 0) / 1e7,
                "dividend_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "sector": info.get("sector", ""),
            })
        except Exception:
            continue
    return pd.DataFrame(records)
