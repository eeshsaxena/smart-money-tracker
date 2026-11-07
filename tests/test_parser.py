"""Generic SEBI-format portfolio parser."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest
from openpyxl import Workbook

from smt.holdings import parser
from tests.conftest import portfolio, workbook, write_sheet


def _parse_one(**kw):
    wb = Workbook()
    write_sheet(wb.active, "Parag Parikh Flexi Cap Fund (An open-ended scheme)", "September 30, 2026",
                portfolio(9), **kw)
    buf = io.BytesIO()
    wb.save(buf)
    out = parser.parse_workbook(buf.getvalue())
    assert len(out) == 1
    return out[0]


def test_title_date_and_asset_classes():
    p = _parse_one()
    assert p.scheme_title.startswith("Parag Parikh Flexi Cap Fund")
    assert p.as_of == date(2026, 9, 30)
    mix = p.holdings.groupby("asset_class")["pct_nav"].sum().round(2).to_dict()
    assert mix == {"Equity": 75.0, "Equity - Arbitrage": 2.0, "REITs & InvITs": 5.0,
                   "Foreign Equity": 8.0, "Debt & Others": 5.0}


def test_rows_below_grand_total_are_ignored():
    p = _parse_one()
    assert "INE975G08140" not in set(p.holdings["isin"])      # the footnote table


@pytest.mark.parametrize("fraction", [True, False])
def test_percent_scale_is_detected(fraction):
    p = _parse_one(fraction=fraction)
    assert p.holdings["pct_nav"].sum() == pytest.approx(95.0)


def test_layout_without_a_leading_code_column():
    p = _parse_one(lead_col=False)
    assert len(p.holdings) == 8


def test_multi_sheet_workbook_parses_every_scheme():
    content = workbook({"A": ("DSP Flexi Cap Fund", "30 Sep 2026", portfolio(9)),
                        "B": ("DSP Mid Cap Fund", "30 Sep 2026", portfolio(8)),
                        "notes": ("Notes", "x", [])})
    titles = [p.scheme_title for p in parser.parse_workbook(content)]
    assert titles == ["DSP Flexi Cap Fund", "DSP Mid Cap Fund"]     # the empty sheet is skipped


def test_duplicate_isin_rows_are_combined():
    body = ["Equity", ("HDFC Bank Limited", "INE040A01034", "Banks", 100, 10.0, 1.0),
            ("HDFC Bank Limited", "INE040A01034", "Banks", 50, 5.0, 0.5)]
    p = parser.parse_workbook(workbook({"S": ("X Fund", "30 Sep 2026", body)}))[0]
    row = p.holdings.iloc[0]
    assert (row.quantity, row.pct_nav) == (150, pytest.approx(1.5))


def test_value_in_crore_is_converted_to_lakh():
    raw = pd.DataFrame([["X Fund"], ["Portfolio as on 31-08-2026"],
                        ["Name of Instrument", "ISIN", "Industry", "Quantity", "Market Value (Rs. in Crore)",
                         "% to NAV"],
                        ["HDFC Bank Limited", "INE040A01034", "Banks", 10, 2.5, 4.0]])
    p = parser.parse_sheet(raw)
    assert p.holdings.loc[0, "market_value_lakh"] == 250
    assert p.as_of == date(2026, 8, 31)


@pytest.mark.parametrize("text,expected", [
    ("September 30, 2026", date(2026, 9, 30)), ("30 Sep 2026", date(2026, 9, 30)),
    ("30th September, 2026", date(2026, 9, 30)), ("31-08-2026", date(2026, 8, 31)),
    ("31.08.2026 (Unaudited)", date(2026, 8, 31)), ("not a date", None)])
def test_parse_date(text, expected):
    assert parser.parse_date(text) == expected


@pytest.mark.parametrize("isin,section,arb,expected", [
    ("INE040A01034", "equity & equity related", False, "Equity"),
    ("INE040A01034", "equity & equity related", True, "Equity - Arbitrage"),
    ("INE476A16F78", "money market instruments", False, "Debt & Others"),
    ("INF879O01027", "others", False, "Mutual Fund Units"),
    ("US5949181045", "foreign securities and/or overseas etf(s)", False, "Foreign Equity"),
    ("IE00BYZ0B213", "overseas mutual fund", False, "Overseas Funds"),
    ("INE041025011", "(b) reits", False, "REITs & InvITs"),
])
def test_classify(isin, section, arb, expected):
    assert parser.classify(isin, section, arb) == expected


def test_html_saved_as_xls_is_read():
    html = ("<table><tr><td>Y Fund</td></tr><tr><td>Portfolio as on 30 Sep 2026</td></tr>"
            "<tr><td>Name of the Instrument</td><td>ISIN</td><td>% to Net Assets</td></tr>"
            "<tr><td>HDFC Bank Limited</td><td>INE040A01034</td><td>5.5</td></tr></table>").encode()
    out = parser.parse_workbook(html)
    assert out[0].scheme_title == "Y Fund" and out[0].holdings.loc[0, "pct_nav"] == 5.5
