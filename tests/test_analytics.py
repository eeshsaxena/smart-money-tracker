"""Performance, SIP and portfolio maths against hand-checkable answers."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from smt.analytics import performance as perf
from smt.analytics import portfolio as pf
from smt.analytics import sip
from tests.conftest import make_nav


def compounding(rate: float, start="2018-01-01", end="2026-10-30") -> pd.Series:
    """A NAV growing at exactly `rate` a year, every calendar day."""
    idx = pd.date_range(start, end, freq="D")
    days = (idx - idx[0]).days
    return pd.Series(100 * (1 + rate) ** (days / 365.25), index=idx)


def test_period_returns_on_a_known_growth_rate():
    nav = compounding(0.10)
    assert perf.period_return(nav, 1) == pytest.approx(0.10, abs=1e-3)
    assert perf.period_return(nav, 5) == pytest.approx(0.10, abs=1e-3)
    assert perf.period_return(nav, 0.5) == pytest.approx(1.10 ** 0.5 - 1, abs=1e-3)   # absolute below 1Y
    assert perf.period_return(nav, 10) is None                                       # not enough history


def test_start_value_uses_last_nav_on_or_before_start():
    nav = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2024-01-01", "2025-01-02", "2026-01-01"]))
    # 1Y back from 2026-01-01 is 2025-01-01, a non-trading day: use 2024-01-01's NAV, never a later one.
    assert perf.period_return(nav, 1) == pytest.approx(0.21)


def test_drawdown_and_calendar_returns():
    nav = pd.Series([100, 120, 90, 130], index=pd.to_datetime(["2024-12-31", "2025-06-30", "2025-12-31", "2026-03-31"]))
    assert perf.drawdown(nav).min() == pytest.approx(-0.25)
    cal = perf.calendar_year_returns(nav)
    assert cal.loc[2025] == pytest.approx(-0.10) and cal.loc[2026] == pytest.approx(130 / 90 - 1)


def test_rolling_returns_constant_for_constant_growth():
    r = perf.rolling_returns(compounding(0.08), 3)
    assert r.min() == pytest.approx(0.08, abs=2e-3) and r.max() == pytest.approx(0.08, abs=2e-3)


def test_risk_metrics_against_itself():
    nav = make_nav(1)
    m = perf.risk_metrics(nav, nav)
    assert m["beta"] == pytest.approx(1.0) and m["correlation"] == pytest.approx(1.0)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert m["up_capture"] == pytest.approx(1.0) and m["down_capture"] == pytest.approx(1.0)
    assert m["max_drawdown"] <= 0 and m["volatility"] > 0


def test_risk_metrics_need_full_window():
    assert perf.risk_metrics(make_nav(2, years=1), years=3) == {"window_years": None}


def test_category_rank():
    vals = pd.Series([0.1, 0.3, 0.2, None, 0.5])
    cats = pd.Series(["A", "A", "A", "A", "B"])
    assert list(perf.category_rank(vals, cats)) == ["3/3", "1/3", "2/3", "", "1/1"]


# ---------------------------------------------------------------- SIP / XIRR
def test_xirr_simple_and_irregular():
    assert sip.xirr([(date(2025, 1, 1), -100), (date(2026, 1, 1), 110)]) == pytest.approx(0.10, abs=1e-4)
    flows = [(date(2024, 1, 1), -1000), (date(2024, 7, 1), -1000), (date(2025, 1, 1), 2200)]
    r = sip.xirr(flows)
    t0 = date(2024, 1, 1)
    assert sum(cf / (1 + r) ** ((d - t0).days / 365) for d, cf in flows) == pytest.approx(0, abs=1e-6)
    assert sip.xirr([(date(2025, 1, 1), -100)]) is None


def test_sip_on_flat_nav_returns_nothing_and_invests_monthly():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    res = sip.simulate(pd.Series(50.0, index=idx), 1000, pd.Timestamp("2024-01-01"))
    assert len(res.installments) == 12
    assert res.invested == 12000 and res.value == pytest.approx(12000)
    assert res.xirr == pytest.approx(0.0, abs=1e-6)
    assert res.installments["date"].dt.month.tolist() == list(range(1, 13))


def test_sip_xirr_matches_growth_rate():
    nav = compounding(0.12, start="2020-01-01", end="2025-12-31")
    res = sip.simulate(nav, 5000, pd.Timestamp("2021-01-01"))
    assert res.xirr == pytest.approx(0.12, abs=2e-3)
    path = sip.value_path(nav, res)
    assert path["invested"].iloc[-1] == res.invested


# ---------------------------------------------------------------- portfolio
def test_optimiser_long_only_and_prefers_low_volatility_for_min_vol():
    navs = pd.DataFrame({"calm": make_nav(3, vol=0.08, drift=0.10), "wild": make_nav(4, vol=0.30, drift=0.10),
                         "mid": make_nav(5, vol=0.15, drift=0.12)})
    opt = pf.optimise(navs)
    for key in ("max_sharpe", "min_volatility"):
        w = opt[key]["weights"]
        assert w.sum() == pytest.approx(1.0, abs=1e-6) and (w >= -1e-9).all()
    assert opt["min_volatility"]["weights"].idxmax() == "calm"
    assert opt["min_volatility"]["volatility"] <= opt["funds"]["volatility"].min() + 1e-6
    assert opt["frontier"]["volatility"].is_monotonic_increasing


def test_correlation_matrix_and_input_checks():
    navs = pd.DataFrame({"a": make_nav(6), "b": make_nav(7)})
    corr = pf.correlation(navs)
    assert np.allclose(np.diag(corr), 1.0) and corr.loc["a", "b"] == corr.loc["b", "a"]
    with pytest.raises(ValueError):
        pf.optimise(navs[["a"]])
