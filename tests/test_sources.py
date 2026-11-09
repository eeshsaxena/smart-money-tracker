"""AMFI and mfapi parsing, offline."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
from openpyxl import Workbook

from smt.sources import amfi, mfapi

NAV_ALL = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Flexi Cap Fund)

PPFAS Mutual Fund

122639;INF879O01027;-;Parag Parikh Flexi Cap Fund;Direct Plan;Growth;93.9483;06-Nov-2026
122640;INF879O01035;-;Parag Parikh Flexi Cap Fund;Regular Plan;Growth;85.1;06-Nov-2026

Open Ended Schemes(Equity Scheme - Mid Cap Fund)

DSP Mutual Fund

119071;INF740K01PX1;-;DSP Midcap Fund;Direct Plan;Growth;167.9720;06-Nov-2026
100001;INF000000001;-;Old Style Fund - Direct Plan - Growth;12.5;05-Nov-2026
"""


def test_parse_nav_all_reads_categories_amcs_and_both_formats():
    df = amfi.parse_nav_all(NAV_ALL).set_index("scheme_code")
    assert len(df) == 4
    assert df.at["122639", "scheme_name"] == "Parag Parikh Flexi Cap Fund"
    assert df.at["122639", "category"] == "Equity Scheme - Flexi Cap Fund"
    assert df.at["122639", "amc"] == "PPFAS Mutual Fund"
    assert df.at["119071", "category"] == "Equity Scheme - Mid Cap Fund"
    assert df.at["119071", "nav"] == 167.972
    assert df.at["119071", "nav_date"] == pd.Timestamp("2026-11-06")
    # The older 6-column format still parses.
    assert df.at["100001", "nav"] == 12.5 and df.at["100001", "plan"] == "Direct Plan"


def test_classification_labels_newest_first():
    assert amfi.classification_labels(date(2026, 11, 7), 3) == ["30Jun2026", "31Dec2025", "30Jun2025"]
    assert amfi.classification_labels(date(2026, 3, 1), 2) == ["31Dec2025", "30Jun2025"]


def test_market_cap_addresses_cover_the_moved_host_and_old_names():
    assert any("portal.amfiindia.com" in u for u in amfi.market_cap_urls("30Jun2026"))
    assert any("www.amfiindia.com" in u for u in amfi.market_cap_urls("30Jun2026"))     # both are tried
    assert amfi.market_cap_urls("30Jun2024") == (amfi.MARKET_CAP_ONE_OFF_URLS["30Jun2024"],)
    assert amfi.label_date("30Jun2026") == date(2026, 6, 30)


def test_parse_market_cap():
    wb = Workbook()
    ws = wb.active
    ws.append(["Average Market Capitalization of listed companies during the six months ended 31 December 2025"])
    ws.append(["Sr. No.", "Company name", "ISIN", "BSE Symbol", "NSE Symbol",
               "Categorization as per SEBI Circular dated Oct 6, 2017"])
    ws.append([1, "Reliance Industries Ltd", "INE002A01018", "RELIANCE", "RELIANCE", "Large Cap"])
    ws.append([150, "Some Midcap", "INE999X01011", "-", "SOMEMID", "Mid Cap"])
    ws.append([4000, "Tiny Co", "INE888Y01012", "-", "TINY", "Small Cap"])
    ws.append([None, "Footnote", "not an isin", None, None, None])
    buf = io.BytesIO()
    wb.save(buf)
    df = amfi.parse_market_cap(buf.getvalue())
    assert list(df["cap_class"]) == ["Large Cap", "Mid Cap", "Small Cap"]
    assert df.loc[0, "nse_symbol"] == "RELIANCE"


def test_parse_mfapi_history_cleans_rows():
    payload = {"meta": {"scheme_name": "X Fund"}, "data": [
        {"date": "06-11-2026", "nav": "10.5"}, {"date": "05-11-2026", "nav": "10.0"},
        {"date": "05-11-2026", "nav": "10.0"}, {"date": "bad", "nav": "1"}, {"date": "04-11-2026", "nav": "0"},
    ]}
    name, df = mfapi.parse_history(payload)
    assert name == "X Fund"
    assert list(df["nav"]) == [10.0, 10.5]                     # sorted, de-duplicated, zero dropped
    assert df["date"].is_monotonic_increasing
