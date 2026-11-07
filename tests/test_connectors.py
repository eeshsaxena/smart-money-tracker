"""AMC connectors: link discovery from saved page snippets, and zip handling."""
from __future__ import annotations

from datetime import date

from smt import universe
from smt.holdings.connectors import REGISTRY
from smt.holdings.connectors.base import month_year
from smt.holdings.connectors.dsp import DSPConnector
from smt.holdings.connectors.ppfas import PPFASConnector
from tests.conftest import portfolio, workbook, zip_of

PPFAS_HTML = """
<a href="/downloads/portfolio-disclosure/2026/PPFCF_PPFAS_Monthly_Portfolio_Report_September_30_2026.xlsx?07102026">x</a>
<a href="/downloads/portfolio-disclosure/2026/PPTSF_PPFAS_Monthly_Portfolio_Report_September_30_2026.xlsx?07102026">x</a>
<a href="/downloads/portfolio-disclosure/2026/PPLF_PPFAS_Monthly_Portfolio_Report_September_30_2026.xlsx">liquid</a>
<a href="/downloads/portfolio-disclosure/2026/PPFCF_PPFAS_Monthly_Portfolio_Report_February_28_2026.xls?09032026">x</a>
<a href="/downloads/portfolio-disclosure/2024/PPFAS_Monthly_Portfolio_Report_December_31_2024.xlsx">consolidated</a>
<a href="#collapseApril2026">menu</a>
"""

DSP_HTML = """
<a href="https://www.dspim.com/media/pages/x/7737577fda-1/dsp-monthend-portfolio-as-on-30-september-2026.zip">a</a>
<a href="https://www.dspim.com/media/pages/x/8a6dbe504f-1/dsp-monthend-portfolio-as-on-31-aug-2026.zip">b</a>
<a href="https://www.dspim.com/media/pages/x/081d3fbd38-1/monthend-portfolios_31_july_2026.zip">c</a>
<a href="https://www.dspim.com/media/pages/x/1c747fb85f-1/monthend-portfolio-february-28-2026.zip">d</a>
<a href="https://www.dspim.com/media/pages/x/03bb8a756b-1/half-yearly-portfolio_march-2026.zip">half-yearly</a>
<a href="https://www.dspim.com/media/pages/x/fe4d/dsp-isin-debt-portfolio-as-on-15th-february-2023.xlsx">debt</a>
"""


def test_ppfas_discovery_maps_scheme_prefix_and_month_end():
    files = PPFASConnector().discover(PPFAS_HTML)
    got = [(f.label, f.as_of) for f in files]
    assert got == [("PPTSF", date(2026, 9, 30)), ("PPFCF", date(2026, 9, 30)), ("PPFCF", date(2026, 2, 28))]
    assert files[0].url.startswith("https://amc.ppfas.com/downloads/")   # relative links are resolved


def test_dsp_discovery_handles_inconsistent_names():
    files = DSPConnector().discover(DSP_HTML)
    assert [f.as_of for f in files] == [date(2026, 9, 30), date(2026, 8, 31), date(2026, 7, 31), date(2026, 2, 28)]


def test_month_year_tokens():
    assert month_year(["dsp", "monthend", "31", "aug", "2026", "zip"]) == (2026, 8)
    assert month_year(["sept", "2025"]) == (2025, 9)
    assert month_year(["portfolio", "zip"]) is None


def test_zip_parsing_skips_debt_workbook():
    content = zip_of({"DSP Equity Portfolio.xlsx": workbook({"A": ("DSP Flexi Cap Fund", "30 Sep 2026", portfolio(9))}),
                      "DSP ISIN DEBT Portfolio.xlsx": workbook({"B": ("DSP Bond Fund", "30 Sep 2026", portfolio(9))})})
    titles = [p.scheme_title for p in DSPConnector().parse(content)]
    assert titles == ["DSP Flexi Cap Fund"]


def test_plain_xlsx_is_not_treated_as_a_zip_of_files():
    content = workbook({"A": ("Parag Parikh Flexi Cap Fund", "September 30, 2026", portfolio(9))})
    assert len(PPFASConnector().parse(content)) == 1


def test_every_holdings_source_in_universe_has_a_connector():
    assert {f.holdings for f in universe.with_holdings()} <= set(REGISTRY)
