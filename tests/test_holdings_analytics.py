"""Holdings analytics: quantity-based changes, flows, overlap and SEBI style rules."""
from __future__ import annotations

import pandas as pd
import pytest

from smt.analytics import holdings as H

AUG, SEP = pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")


def row(code, as_of, isin, name, qty, pct, value=None, cap="Large Cap", cls="Equity"):
    return {"scheme_code": code, "as_of": as_of, "isin": isin, "instrument": name, "industry": "Banks",
            "quantity": qty, "pct_nav": pct, "market_value_lakh": value if value is not None else qty,
            "asset_class": cls, "cap_class": cap}


@pytest.fixture
def h() -> pd.DataFrame:
    rows = [
        # fund A, August
        row("A", AUG, "I1", "Kept", 100, 10), row("A", AUG, "I2", "Bought more", 100, 10),
        row("A", AUG, "I3", "Trimmed", 100, 10), row("A", AUG, "I4", "Sold out", 100, 10),
        row("A", AUG, "I6", "Split", 100, 10),
        # fund A, September: price of 'Kept' doubled (weight up) but no shares bought
        row("A", SEP, "I1", "Kept", 100, 20, value=200), row("A", SEP, "I2", "Bought more", 150, 12, value=150),
        row("A", SEP, "I3", "Trimmed", 60, 6, value=60), row("A", SEP, "I5", "Brand new", 80, 8, value=80),
        row("A", SEP, "I6", "Split", 200, 10, value=100),
        # fund B, two months
        row("B", AUG, "I2", "Bought more", 10, 5), row("B", SEP, "I2", "Bought more", 30, 7, value=30),
        row("B", SEP, "I5", "Brand new", 20, 3, value=20),
    ]
    return pd.DataFrame(rows)


def test_changes_are_classified_by_quantity_not_weight(h):
    c = H.changes(h, "A").set_index("instrument")["status"]
    assert c["Kept"] == "Unchanged"            # weight doubled from price; no shares bought
    assert c["Bought more"] == "Added"
    assert c["Trimmed"] == "Trimmed"
    assert c["Sold out"] == "Exited"
    assert c["Brand new"] == "New"
    assert c["Split"] == "Split/bonus?"        # shares doubled, weight flat


def test_flow_in_crore(h):
    c = H.changes(h, "A").set_index("instrument")
    # 50 extra shares at 1 lakh each = 50 lakh = 0.5 crore
    assert c.loc["Bought more", "flow_cr"] == pytest.approx(0.5)
    assert c.loc["Sold out", "flow_cr"] == pytest.approx(-1.0)        # priced at last month's value
    assert c.loc["Split", "flow_cr"] == 0


def test_smart_money_flow_aggregates_funds_and_skips_splits(h):
    f = H.smart_money_flow(h, ["A", "B"]).set_index("instrument")
    assert f.loc["Bought more", "funds_buying"] == 2
    assert f.loc["Brand new", "new_positions"] == 2
    assert f.loc["Sold out", "full_exits"] == 1
    assert "Split" not in f.index


def test_overlap(h):
    o = H.overlap(H.snapshot(h, "A", SEP), H.snapshot(h, "B", SEP))
    # common: I2 (12 vs 7 -> 7) and I5 (8 vs 3 -> 3)
    assert o["weighted_pct"] == pytest.approx(10) and o["common_stocks"] == 2
    m = H.overlap_matrix(h, ["A", "B"])
    assert m.loc["A", "A"] == 100 and m.loc["A", "B"] == m.loc["B", "A"] == pytest.approx(10)


def test_concentration_effective_stocks():
    snap = pd.DataFrame([row("X", SEP, f"I{i}", f"S{i}", 1, 10) for i in range(10)])
    c = H.concentration(snap)
    assert c["stocks"] == 10 and c["effective_stocks"] == pytest.approx(10) and c["top10_pct"] == 100


def test_cap_mix_and_asset_mix():
    snap = pd.DataFrame([row("X", SEP, "I1", "L", 1, 50), row("X", SEP, "I2", "M", 1, 30, cap="Mid Cap"),
                         row("X", SEP, "I3", "F", 1, 10, cap=None, cls="Foreign Equity")])
    assert H.cap_mix(snap).to_dict() == {"Large Cap": 50, "Mid Cap": 30, "Foreign": 10}
    assert H.asset_mix(snap)["Cash, TREPS & receivables"] == pytest.approx(10)


@pytest.mark.parametrize("category,label", [
    ("Equity Scheme - Large Cap Fund", "≥80% large cap"),
    ("Equity Scheme - Large & Mid Cap Fund", "≥35% large and ≥35% mid cap"),
    ("Equity Scheme - Mid Cap Fund", "≥65% mid cap"),
    ("Equity Scheme - Focused Fund", "at most 30 stocks"),
    ("Equity Scheme - ELSS", "≥80% in equity"),
])
def test_category_rules(category, label):
    assert H.category_rule(category)["label"] == label


def test_style_drift_flags_months_below_the_rule():
    rows = [row("M", AUG, "I1", "a", 1, 70, cap="Mid Cap"), row("M", AUG, "I2", "b", 1, 25),
            row("M", SEP, "I1", "a", 1, 60, cap="Mid Cap"), row("M", SEP, "I2", "b", 1, 35)]
    d = H.style_drift(pd.DataFrame(rows), "M", "Equity Scheme - Mid Cap Fund")
    assert list(d["meets_rule"]) == [True, False]
