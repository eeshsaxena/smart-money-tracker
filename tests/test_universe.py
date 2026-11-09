"""Fund universe: the list itself must be internally consistent."""
from __future__ import annotations

import pytest

from smt import universe


def test_scheme_codes_are_unique():
    codes = [f.scheme_code for f in universe.FUNDS]
    assert len(codes) == len(set(codes))       # the old config reused codes for different funds


def test_reference_series_are_flagged_and_labelled():
    for code in (universe.BENCHMARK_CODE, universe.CASH_PROXY, *universe.STYLE_FACTORS.values()):
        fund = universe.by_code(code)
        assert fund.is_benchmark and fund.label


@pytest.mark.parametrize("category,code", [
    ("Equity Scheme - Large Cap Fund", universe.NIFTY_100),
    ("Equity Scheme - Large & Mid Cap Fund", universe.LARGEMID_250),   # must not be caught by "Mid Cap"
    ("Equity Scheme - Mid Cap Fund", universe.MIDCAP_150),
    ("Equity Scheme - Small Cap Fund", universe.SMALLCAP_250),
    ("Equity Scheme - Flexi Cap Fund", universe.NIFTY_500),
    ("Equity Scheme - ELSS", universe.NIFTY_500),
    (None, universe.NIFTY_500),
])
def test_each_category_is_judged_against_its_own_index(category, code):
    assert universe.category_index(category) == code
    assert universe.by_code(code).is_benchmark


def test_renamed_funds_can_be_found_under_their_old_names():
    names = universe.names_in_disclosures(universe.by_code("119250"))
    assert names[0] == "DSP Large Cap Fund" and "DSP Top 100 Equity Fund" in names


@pytest.mark.parametrize("a,b", [
    ("DSP Midcap Fund", "DSP Mid Cap Fund"),
    ("DSP Large & Mid Cap Fund", "DSP Large and Mid Cap Fund"),
    ("Parag Parikh Flexi Cap Fund", "Parag Parikh Flexi Cap Fund (An open-ended dynamic equity scheme)"),
    ("SBI SMALL CAP FUND", "SBI Small Cap Fund"),
])
def test_normalise_matches_spelling_variants(a, b):
    assert universe.normalise(a) == universe.normalise(b)


def test_normalise_keeps_different_funds_apart():
    assert universe.normalise("DSP Midcap Fund") != universe.normalise("DSP Nifty Midcap 150 Index Fund")
