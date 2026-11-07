"""Generic parser for SEBI-format monthly portfolio disclosures.

SEBI requires every AMC to publish month-end portfolios with the same columns:
Name of the Instrument, ISIN, Industry / Rating, Quantity, Market Value
(Rs. in lakhs) and % to Net Assets, under a title giving the scheme name and
the "as on" date. Layouts differ (extra columns, sections, side tables,
fractions vs percentages), so the parser finds the header row and the columns
by their text instead of by position.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
# Indian ISIN: IN + E (company) + 4-char issuer + 2-digit security type + 2 + check digit.
# Security type 01 is equity shares.
EQUITY_ISIN_RE = re.compile(r"^INE[A-Z0-9]{4}01[A-Z0-9]{2}\d$")
DATE_RE = re.compile(r"as\s+on\s*[:\-]?\s*(.+)", re.I)


@dataclass
class ParsedPortfolio:
    scheme_title: str
    as_of: date | None
    holdings: pd.DataFrame   # instrument, isin, industry, quantity, market_value_lakh, pct_nav, asset_class


# --------------------------------------------------------------------------- #
# Workbook reading
# --------------------------------------------------------------------------- #
def read_workbook(content: bytes) -> dict[str, pd.DataFrame]:
    """Return {sheet name: raw grid} for .xlsx, legacy .xls, or HTML saved as .xls."""
    if content[:2] == b"PK":
        engine = "openpyxl"
    elif content[:4] == b"\xd0\xcf\x11\xe0":
        engine = "xlrd"
    else:
        tables = pd.read_html(io.BytesIO(content), header=None)
        return {f"table{i}": t for i, t in enumerate(tables)}
    return pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, engine=engine)


# --------------------------------------------------------------------------- #
# Sheet parsing
# --------------------------------------------------------------------------- #
def _text(v) -> str:
    return "" if pd.isna(v) else " ".join(str(v).split())


def _find_header(raw: pd.DataFrame) -> int | None:
    for i in range(min(len(raw), 40)):
        cells = [_text(v).lower() for v in raw.iloc[i].values]
        if "isin" in cells and any("instrument" in c or c.startswith("name of") for c in cells):
            return i
    return None


def _map_columns(header: list[str]) -> dict[str, int]:
    cols: dict[str, int] = {}
    for idx, h in enumerate(header):
        h = h.lower()
        if not h:
            continue
        if "name" not in cols and ("instrument" in h or h.startswith("name of")):
            cols["name"] = idx
        elif h == "isin":
            cols["isin"] = idx
        elif "industry" not in cols and ("industry" in h or "rating" in h):
            cols["industry"] = idx
        elif "quantity" not in cols and "quantity" in h:
            cols["quantity"] = idx
        elif "value" not in cols and ("market" in h or "fair value" in h):
            cols["value"] = idx
            cols["value_in_crore"] = int("crore" in h or " cr" in h)
        elif "pct" not in cols and "%" in h and ("net asset" in h or "nav" in h or "aum" in h):
            cols["pct"] = idx
    return cols


def _title_and_date(raw: pd.DataFrame, header_row: int) -> tuple[str, date | None]:
    title, as_of = "", None
    for i in range(header_row):
        for v in raw.iloc[i].values:
            t = _text(v)
            if not t:
                continue
            m = DATE_RE.search(t)
            if m and as_of is None:
                as_of = parse_date(m.group(1))
            elif not title and "fund" in t.lower() and "portfolio" not in t.lower():
                title = t
    return title, as_of


def parse_date(text: str) -> date | None:
    """'September 30, 2026' / '30 Sep 2026' / '30th September, 2026' / '30-09-2026' -> date."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.I)
    text = re.split(r"[()]", text)[0].strip(" .,")
    ts = pd.to_datetime(text, errors="coerce", dayfirst=not re.match(r"^[A-Za-z]", text))
    return None if pd.isna(ts) else ts.date()


# Headings that start a new major section. Sub-headings such as "(a) Listed / awaiting
# listing" do not, so they leave the major section unchanged.
MAJOR_SECTION_WORDS = ("equity", "foreign", "debt", "money market", "others", "reit", "invit",
                       "government", "treps", "mutual fund", "cash", "bond", "certificate",
                       "commercial paper", "treasury", "derivative")


def classify(isin: str, section: str, arbitrage: bool = False) -> str:
    """Asset class from the ISIN and the section heading the row sits under.

    'Equity' is reserved for directional Indian stock holdings. Stocks held in an
    arbitrage book (bought and hedged with futures) are kept apart, because they
    are not a bet on the stock.
    """
    if "reit" in section or "invit" in section:
        return "REITs & InvITs"
    if EQUITY_ISIN_RE.match(isin):
        return "Equity - Arbitrage" if arbitrage else "Equity"
    if isin.startswith("INF"):
        return "Mutual Fund Units"
    if not isin.startswith("IN"):
        if "mutual fund" in section:
            return "Overseas Funds"
        if "equity" in section or "foreign" in section:
            return "Foreign Equity"
    return "Debt & Others"


def parse_sheet(raw: pd.DataFrame) -> ParsedPortfolio | None:
    header_row = _find_header(raw)
    if header_row is None:
        return None
    cols = _map_columns([_text(v) for v in raw.iloc[header_row].values])
    if not {"name", "isin", "pct"} <= cols.keys():
        return None
    title, as_of = _title_and_date(raw, header_row)

    rows, section, arbitrage = [], "", False
    for i in range(header_row + 1, len(raw)):
        r = raw.iloc[i].values
        name = _text(r[cols["name"]])
        isin = _text(r[cols["isin"]]).upper()
        if not ISIN_RE.match(isin):
            heading = name.lower()
            if heading.startswith("grand total"):
                break       # below this: derivatives, footnote tables, performance data
            if heading and not isin:
                if any(w in heading for w in MAJOR_SECTION_WORDS):
                    section, arbitrage = heading, False
                elif "arbitrage" in heading:
                    arbitrage = True
                elif heading not in ("sub total", "subtotal", "total"):
                    arbitrage = False
            continue
        rows.append({
            "instrument": name,
            "isin": isin,
            "industry": _text(r[cols["industry"]]) if "industry" in cols else "",
            "quantity": pd.to_numeric(r[cols["quantity"]], errors="coerce") if "quantity" in cols else None,
            "market_value_lakh": pd.to_numeric(r[cols["value"]], errors="coerce") if "value" in cols else None,
            "pct_nav": pd.to_numeric(r[cols["pct"]], errors="coerce"),
            "asset_class": classify(isin, section, arbitrage),
        })
    df = pd.DataFrame(rows, columns=["instrument", "isin", "industry", "quantity",
                                     "market_value_lakh", "pct_nav", "asset_class"])
    if df.empty:
        return ParsedPortfolio(title, as_of, df)

    df["pct_nav"] = df["pct_nav"].astype(float)
    if df["pct_nav"].abs().sum() <= 1.5:            # published as fractions (0.0775 = 7.75%)
        df["pct_nav"] = df["pct_nav"] * 100
    if cols.get("value_in_crore"):
        df["market_value_lakh"] = df["market_value_lakh"] * 100
    # The same ISIN can appear twice (e.g. in two sections); combine it.
    df = (df.groupby(["isin", "asset_class"], as_index=False, sort=False)
            .agg(instrument=("instrument", "first"), industry=("industry", "first"),
                 quantity=("quantity", "sum"), market_value_lakh=("market_value_lakh", "sum"),
                 pct_nav=("pct_nav", "sum")))
    return ParsedPortfolio(title, as_of, df[["instrument", "isin", "industry", "quantity",
                                             "market_value_lakh", "pct_nav", "asset_class"]])


def parse_workbook(content: bytes) -> list[ParsedPortfolio]:
    """Parse every sheet that looks like a portfolio statement."""
    out = []
    for raw in read_workbook(content).values():
        parsed = parse_sheet(raw)
        if parsed is not None and not parsed.holdings.empty:
            out.append(parsed)
    return out
