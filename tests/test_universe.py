"""Fund universe: the list itself must be internally consistent."""
from __future__ import annotations

import pytest

from smt import universe


def test_scheme_codes_are_unique():
    codes = [f.scheme_code for f in universe.FUNDS]
    assert len(codes) == len(set(codes))       # the old config reused codes for different funds


def test_exactly_one_benchmark():
    assert sum(f.is_benchmark for f in universe.FUNDS) == 1
    assert universe.by_code(universe.BENCHMARK_CODE).is_benchmark


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
