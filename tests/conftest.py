"""Shared test fixtures: synthetic NAVs, SEBI-format workbooks and a fully offline pipeline."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from smt import universe
from smt.holdings.connectors.base import Connector, DisclosureFile
from smt.pipeline import Sources


# --------------------------------------------------------------------------- #
# NAV series
# --------------------------------------------------------------------------- #
def make_nav(seed: int, years: float = 4, drift: float = 0.12, vol: float = 0.15,
             end: str = "2026-10-30") -> pd.Series:
    idx = pd.bdate_range(end=end, periods=int(252 * years))
    rng = np.random.default_rng(seed)
    r = rng.normal(drift / 252, vol / np.sqrt(252), len(idx))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


# --------------------------------------------------------------------------- #
# SEBI-format portfolio workbooks
# --------------------------------------------------------------------------- #
HEADER = ["Name of the Instrument", "ISIN", "Industry / Rating", "Quantity",
          "Market/Fair Value (Rs. in Lakhs)", "% to Net Assets"]


def write_sheet(ws, title: str, as_of_text: str, body: list, fraction: bool = True,
                lead_col: bool = True, footnote: bool = True) -> None:
    """body items: a str is a section heading; a tuple is (name, isin, industry, qty, value, pct%)."""
    pad = [None] if lead_col else []
    ws.append(pad + [title])
    ws.append([])
    ws.append(pad + [f"Monthly Portfolio Statement as on {as_of_text}"])
    ws.append((["Code"] if lead_col else []) + HEADER)
    for item in body:
        if isinstance(item, str):
            ws.append(pad + [item])
        else:
            name, isin, ind, qty, val, pct = item
            ws.append(pad + [name, isin, ind, qty, val, pct / 100 if fraction else pct])
    ws.append(pad + ["GRAND TOTAL"])
    if footnote:   # a table below the portfolio with different meanings; must be ignored
        ws.append(pad + ["Defaulted security", "INE975G08140", "0", 0, 5965.0, 372.15])


def workbook(sheets: dict[str, tuple]) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    for name, args in sheets.items():
        write_sheet(wb.create_sheet(name), *args)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def zip_of(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def portfolio(month: int) -> list:
    """A small equity portfolio; month 9 vs month 8 has a new stock, an exit, a buy, a trim and a split."""
    later = month >= 9
    body = ["Equity & Equity related", "(a) Listed / awaiting listing on Stock Exchanges",
            ("HDFC Bank Limited", "INE040A01034", "Banks", 1000 if later else 800, 1800.0, 30.0),
            ("ICICI Bank Limited", "INE090A01021", "Banks", 500 if later else 600, 700.0, 20.0),
            ("Infosys Limited", "INE009A01021", "IT - Software", 2000 if later else 1000, 750.0, 15.0)]
    body.append(("Coal India Limited", "INE522F01014", "Consumable Fuels", 400, 600.0, 10.0) if later
                else ("ITC Limited", "INE154A01025", "Diversified FMCG", 300, 600.0, 10.0))
    body += ["Sub Total", "Arbitrage",
             ("Reliance Industries Limited", "INE002A01018", "Petroleum Products", 100, 120.0, 2.0),
             "Sub Total", "(b) Reits",
             ("Embassy Office Parks REIT", "INE041025011", "Realty", 50, 300.0, 5.0),
             "Equity & Equity related Foreign Investments",
             ("Alphabet Inc A", "US02079K3059", "Software", 10, 480.0, 8.0),
             "Money Market Instruments",
             ("Canara Bank CD", "INE476A16F78", "CRISIL A1+", 5, 300.0, 5.0)]
    return body


@pytest.fixture
def market_cap_df() -> pd.DataFrame:
    return pd.DataFrame({
        "isin": ["INE040A01034", "INE090A01021", "INE009A01021", "INE522F01014", "INE154A01025", "INE002A01018"],
        "company": ["HDFC Bank", "ICICI Bank", "Infosys", "Coal India", "ITC", "Reliance"],
        "nse_symbol": ["HDFCBANK", "ICICIBANK", "INFY", "COALINDIA", "ITC", "RELIANCE"],
        "cap_class": ["Large Cap", "Large Cap", "Large Cap", "Mid Cap", "Large Cap", "Large Cap"],
    })


# --------------------------------------------------------------------------- #
# Fake connectors and sources for an offline pipeline run
# --------------------------------------------------------------------------- #
class FakeConnector(Connector):
    def __init__(self, name: str, titles: list[str]):
        self.name = name
        self.titles = titles
        self.downloads = 0

    def discover(self, html=None):
        return [DisclosureFile(self.name, f"https://example.test/{self.name}/{m}.zip", date(2026, m, 28 if m == 2 else 30), "all")
                for m in (9, 8, 6)]

    def download(self, f):
        self.downloads += 1
        month = f.as_of.month
        sheets = {f"S{i}": (t, f"{f.as_of:%B %d, %Y}", portfolio(month)) for i, t in enumerate(self.titles)}
        return zip_of({"equity.xlsx": workbook(sheets), "debt.xlsx": workbook({"D": ("Debt Fund", "x", [])})})

    def keep_member(self, filename):
        return "debt" not in filename


def disclosure_titles(connector: str) -> list[str]:
    """Scheme titles as an AMC would print them: suffixes and spelling differ from AMFI's names."""
    titles = []
    for f in universe.with_holdings(connector):
        t = f.expected_name.replace("Midcap", "Mid Cap")
        titles.append(t + " (An open-ended equity scheme)" if connector == "ppfas" else t)
    return titles + ["Some Fund We Do Not Track"]


def fake_master(mismatch: str | None = None) -> pd.DataFrame:
    rows = []
    for f in universe.FUNDS:
        rows.append({"scheme_code": f.scheme_code, "scheme_name": "Some Other Fund" if f.scheme_code == mismatch
                     else f.expected_name, "plan": "Direct Plan", "option": "Growth",
                     "category": "Equity Scheme - Flexi Cap Fund", "amc": f.amc})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_sources(tmp_path, market_cap_df):
    navs = {f.scheme_code: make_nav(i) for i, f in enumerate(universe.FUNDS)}

    def nav_history(code):
        return "x", pd.DataFrame({"date": navs[code].index, "nav": navs[code].values})

    return Sources(
        scheme_master=lambda: fake_master(),
        nav_history=nav_history,
        market_cap=lambda: ("31Dec2025", market_cap_df),
        connectors={name: FakeConnector(name, disclosure_titles(name)) for name in ("ppfas", "dsp")},
        raw_dir=tmp_path / "raw",
    )
