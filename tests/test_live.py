"""Live checks against the real sources. Skipped unless SMT_LIVE_TESTS=1.

    SMT_LIVE_TESTS=1 python -m pytest tests/test_live.py

These catch the failure the offline tests can't: a source changing under us
(an AMC renaming its files, AMFI changing its format, a scheme code reused).
"""
from __future__ import annotations

import os

import pytest

from smt import universe
from smt.holdings.connectors import default_connectors
from smt.sources import amfi, mfapi

pytestmark = pytest.mark.skipif(os.getenv("SMT_LIVE_TESTS") != "1", reason="set SMT_LIVE_TESTS=1 to run")


def test_every_scheme_code_is_the_fund_we_think_it_is():
    master = amfi.fetch_scheme_master().set_index("scheme_code")
    wrong = [f"{f.scheme_code}: {master['scheme_name'].get(f.scheme_code)} != {f.expected_name}"
             for f in universe.FUNDS
             if universe.normalise(str(master["scheme_name"].get(f.scheme_code, ""))) != universe.normalise(f.expected_name)]
    assert not wrong, wrong


def test_nav_history_available():
    name, df = mfapi.fetch_history(universe.BENCHMARK_CODE)
    assert len(df) > 1000 and "nifty 50" in name.lower()


@pytest.mark.parametrize("name", ["ppfas", "dsp"])
def test_connector_finds_and_parses_latest_month(name):
    connector = default_connectors()[name]
    files = connector.discover()
    assert files, "no files found on the disclosure page"
    targets = {universe.normalise(f.expected_name) for f in universe.with_holdings(name)}
    latest = [f for f in files if f.as_of == files[0].as_of]
    found = set()
    for f in latest:
        found |= {universe.normalise(p.scheme_title) for p in connector.parse(connector.download(f))}
    assert targets <= found, targets - found


def test_market_cap_list_available():
    label, df = amfi.fetch_market_cap()
    assert df["cap_class"].value_counts()["Large Cap"] == 100
