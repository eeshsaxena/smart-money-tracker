"""The analyses behind the Evidence tab, on synthetic data with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smt import report, universe
from smt.analytics import evidence, flows, style
from smt.analytics import holdings as H
from smt.analytics import performance as perf
from smt.analytics import portfolio as pf
from tests.conftest import make_nav

MONTHS = list(pd.date_range("2025-01-31", periods=8, freq="ME"))


def holding(code, as_of, isin, name, qty, price_lakh, cap="Large Cap", cls="Equity", pct=5.0):
    return {"scheme_code": code, "as_of": as_of, "isin": isin, "instrument": name, "industry": "x",
            "quantity": float(qty), "pct_nav": pct, "market_value_lakh": qty * price_lakh,
            "asset_class": cls, "cap_class": cap}


# ---------------------------------------------------------------- risk measures
def test_sortino_uses_downside_deviation_over_all_days():
    """Hand-computed: the std of the losing days alone would be zero here, since the losses are identical."""
    idx = pd.bdate_range("2023-01-02", periods=800)
    r = np.where(np.arange(len(idx)) % 2 == 0, 0.01, -0.005)
    nav = pd.Series(100 * np.cumprod(1 + r), index=idx)
    m = perf.risk_metrics(nav, years=3, rf=0.0)
    window = nav.loc[nav.index[-1] - pd.DateOffset(days=round(3 * 365.25)):].pct_change().dropna()
    expected = window.mean() * 252 / (np.sqrt((window.clip(upper=0) ** 2).mean()) * np.sqrt(252))
    assert m["sortino"] == pytest.approx(expected)
    assert np.isfinite(m["sortino"])


def test_beta_is_not_fooled_by_a_one_day_lag():
    """A fund that follows the index a day late has a true beta of 1. Daily data says about zero."""
    idx = pd.bdate_range("2021-01-01", periods=252 * 4)
    rng = np.random.default_rng(7)
    b = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    f = b.shift(1).fillna(0)
    bench, fund = 100 * (1 + b).cumprod(), 100 * (1 + f).cumprod()
    daily_beta = f.cov(b) / b.var()
    assert abs(daily_beta) < 0.2
    assert perf.risk_metrics(fund, bench)["beta"] == pytest.approx(1.0, abs=0.25)


def test_tracking_error_and_information_ratio():
    bench = make_nav(1)
    m = perf.risk_metrics(bench, bench)
    assert m["active_return"] == pytest.approx(0.0, abs=1e-9) and m["tracking_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["information_ratio"] is None


def test_rolling_vs_index_counts_windows_the_fund_won():
    idx = pd.date_range("2018-01-01", "2026-10-30", freq="D")
    days = (idx - idx[0]).days
    index = pd.Series(100 * 1.10 ** (days / 365.25), index=idx)
    fund = pd.Series(100 * 1.12 ** (days / 365.25), index=idx)
    r = perf.rolling_vs_index(fund, index)
    assert r["hit_rate"] == 1.0 and r["median_active"] == pytest.approx(0.02, abs=2e-3) and r["windows"] > 1000
    assert perf.rolling_vs_index(index, fund)["hit_rate"] == 0.0
    assert perf.rolling_vs_index(fund.tail(100), index.tail(100))["windows"] == 0


# ---------------------------------------------------------------- corporate actions
def filler(as_of):
    return [holding(c, as_of, f"INE9{c}", "Filler", 1000, 1.0) for c in ("A", "B")]


def test_bonus_is_found_when_share_count_and_price_agree():
    aug, sep = MONTHS[0], MONTHS[1]
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE849A01020", "Trent", 100, 3.0), holding("A", sep, "INE849A01020", "Trent", 150, 2.1),
        holding("B", aug, "INE849A01020", "Trent", 100, 3.0), holding("B", sep, "INE849A01020", "Trent", 160, 2.1)])
    acts = H.corporate_actions(h, aug, sep)
    assert acts.set_index("isin").loc["INE849A01020", "factor"] == 1.5     # a rule looking for "doubled" misses this
    a = H.changes(h, "A", sep, aug).set_index("instrument").loc["Trent"]
    assert a["status"] == "Split/bonus?" and a["flow_cr"] == 0 and a["quantity_prev"] == 150
    # Fund B also bought 10 shares on top of the bonus, and only those count: 10 x 2.1 lakh = 21 lakh.
    b = H.changes(h, "B", sep, aug).set_index("instrument").loc["Trent"]
    assert b["status"] == "Added" and b["flow_cr"] == pytest.approx(0.21)


def test_buying_a_falling_stock_is_not_mistaken_for_a_bonus():
    aug, sep = MONTHS[0], MONTHS[1]
    # Exactly 1.5x shares, but the price fell only 5%: a purchase.
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE111A01011", "Real buy", 100, 3.0), holding("A", sep, "INE111A01011", "Real buy", 150, 2.85)])
    assert H.corporate_actions(h, aug, sep).empty
    assert H.changes(h, "A", sep, aug).set_index("instrument").loc["Real buy", "status"] == "Added"


def test_small_multiple_in_a_falling_market_needs_two_funds():
    aug, sep = MONTHS[0], MONTHS[1]
    # One fund adds exactly 10% in a month the stock falls 20%. Coincidence, not a 1:10 bonus.
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE111A01011", "Bank", 1000, 1.0), holding("A", sep, "INE111A01011", "Bank", 1100, 0.8)])
    assert H.corporate_actions(h, aug, sep).empty


def test_split_with_a_new_isin_is_one_stock_not_a_sale_and_a_purchase():
    aug, sep = MONTHS[0], MONTHS[1]
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE296A01024", "Bajaj Finance", 100, 10.0),
        holding("A", sep, "INE296A01032", "Bajaj Finance", 1000, 1.05)])
    c = H.changes(h, "A", sep, aug).set_index("isin")
    assert "INE296A01024" not in c.index                   # no phantom exit
    assert c.loc["INE296A01032", "status"] == "Split/bonus?" and c.loc["INE296A01032", "flow_cr"] == 0


def test_split_factor_comes_from_the_price_when_every_holder_also_traded():
    aug, sep = MONTHS[0], MONTHS[1]
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE262H01013", "Persistent", 100, 10.0),
        holding("A", sep, "INE262H01021", "Persistent", 230, 5.2)])     # 2x split plus 30 shares bought
    acts = H.corporate_actions(h, aug, sep).set_index("isin")
    assert acts.loc["INE262H01021", "factor"] == 2.0
    c = H.changes(h, "A", sep, aug).set_index("isin").loc["INE262H01021"]
    assert c["status"] == "Added" and c["flow_cr"] == pytest.approx(30 * 5.2 / 100)


def test_demerged_shares_are_a_spin_off_not_a_purchase():
    aug, sep = MONTHS[0], MONTHS[1]
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE154A01025", "ITC Limited", 1000, 4.0), holding("A", sep, "INE154A01025", "ITC Limited", 1000, 3.6),
        holding("B", aug, "INE154A01025", "ITC Limited", 3000, 4.0), holding("B", sep, "INE154A01025", "ITC Limited", 3000, 3.6),
        holding("A", sep, "INE379A01028", "ITC Hotels Limited", 100, 2.0),
        holding("B", sep, "INE379A01028", "ITC Hotels Limited", 300, 2.0)])
    spun = H.spin_offs(h, aug, sep)
    assert list(spun["isin"]) == ["INE379A01028"] and spun["ratio"].iloc[0] == pytest.approx(0.1)
    c = H.changes(h, "A", sep, aug).set_index("instrument").loc["ITC Hotels Limited"]
    assert c["status"] == "Spin-off" and c["flow_cr"] == 0
    assert "ITC Hotels Limited" not in set(H.smart_money_flow(h, ["A", "B"], sep)["instrument"])


def test_a_new_stock_matching_an_unrelated_holding_by_chance_is_a_real_purchase():
    aug, sep = MONTHS[0], MONTHS[1]
    # 0.75 of Dodla's share count, but one fund and a different company: nothing says demerger.
    h = pd.DataFrame(filler(aug) + filler(sep) + [
        holding("A", aug, "INE021O01019", "Dodla Dairy", 1000, 1.0), holding("A", sep, "INE021O01019", "Dodla Dairy", 1000, 1.0),
        holding("A", sep, "INE330T01021", "Happy Forgings", 750, 1.0)])
    assert H.spin_offs(h, aug, sep).empty
    assert H.changes(h, "A", sep, aug).set_index("instrument").loc["Happy Forgings", "status"] == "New"


# ---------------------------------------------------------------- following the money
def panel(n_months=8, inflow=1.0):
    """Two funds, four stocks. Each month W is bought and L sold; W then rises 5% and L falls 5%."""
    rows = []
    price = {"W": 1.0, "L": 1.0, "S1": 1.0, "S2": 1.0}
    qty = {"W": 1000.0, "L": 1000.0, "S1": 1000.0, "S2": 1000.0}
    extra = {f"X{i}": 1000.0 for i in range(8)}
    for t, m in enumerate(MONTHS[:n_months]):
        if t > 0:
            price["W"] *= 1.05
            price["L"] *= 0.95
            qty = {k: v * inflow for k, v in qty.items()}
            qty["W"] *= 1.10
            qty["L"] *= 0.90
            extra = {k: v * inflow for k, v in extra.items()}
        for code in ("A", "B"):
            for s in ("W", "L", "S1", "S2"):
                rows.append(holding(code, m, f"INE{s}", s, round(qty[s]), price[s]))
            rows += [holding(code, m, f"INE{k}", k, round(v), 1.0) for k, v in extra.items()]
    return pd.DataFrame(rows)


def many_stock_panel(n_months=14, signal=0.05, seed=3):
    """Stocks bought this month return `signal` more next month than stocks sold."""
    rng = np.random.default_rng(seed)
    names = [f"S{i:02d}" for i in range(30)]
    price = dict.fromkeys(names, 1.0)
    qty = dict.fromkeys(names, 10000.0)
    months = list(pd.date_range("2024-01-31", periods=n_months, freq="ME"))
    rows, direction = [], dict.fromkeys(names, 0)
    for m in months:
        for s in names:
            price[s] *= 1 + direction[s] * signal + rng.normal(0, 0.01)
        new_dir = {s: int(rng.choice([-1, 0, 1])) for s in names}
        for s in names:
            qty[s] = round(qty[s] * (1 + 0.1 * new_dir[s]))
        direction = new_dir
        for s in names:
            rows.append(holding("A", m, f"INE{s}", s, qty[s], price[s]))
    return pd.DataFrame(rows)


def test_trades_ignore_new_money_spread_across_the_whole_portfolio():
    """A fund with 5% inflows buys 5% more of everything; only W and L moved beyond that norm."""
    h = panel(4, inflow=1.05)
    ret, tr = flows.panels(h, ["A", "B"])
    assert set(tr["isin"]) == {"INEW", "INEL"}
    assert (tr.loc[tr["isin"] == "INEW", "direction"] == 1).all()
    assert (tr.loc[tr["isin"] == "INEL", "direction"] == -1).all()


def test_stock_returns_come_from_disclosed_prices():
    ret, _ = flows.panels(panel(4), ["A", "B"])
    assert ret["INEW"].dropna().to_numpy() == pytest.approx(0.05, abs=1e-6)
    assert ret["INEL"].dropna().to_numpy() == pytest.approx(-0.05, abs=1e-6)


def test_returns_are_adjusted_for_a_bonus():
    a, b = MONTHS[0], MONTHS[1]
    h = pd.DataFrame(filler(a) + filler(b) + [
        holding("A", a, "INE849A01020", "Trent", 100, 3.0), holding("A", b, "INE849A01020", "Trent", 150, 2.1)])
    ret, tr = flows.panels(h, ["A"])
    assert ret.loc[b, "INE849A01020"] == pytest.approx(0.05)    # 2.1 x 1.5 / 3 - 1, not a 30% fall
    assert "INE849A01020" not in set(tr["isin"])                 # and not a trade


def test_event_study_finds_a_signal_that_exists():
    h = many_stock_panel(signal=0.05)
    out = flows.follow_the_money(h, ["A"], skip=0)
    assert out["summary"]["spread"] == pytest.approx(0.10, abs=0.02)   # +5% vs -5%
    assert out["summary"]["spread_t"] > 5


def test_event_study_finds_nothing_in_noise():
    h = many_stock_panel(signal=0.0)
    out = flows.follow_the_money(h, ["A"], skip=0)
    assert abs(out["summary"]["spread_t"]) < 3


def test_waiting_a_month_uses_a_later_return():
    h = many_stock_panel(signal=0.05)
    out = flows.follow_the_money(h, ["A"], skip=1)
    assert out["summary"]["months"] > 0 and abs(out["summary"]["spread"]) < 0.03    # the signal lasts one month


def test_event_study_is_empty_without_enough_months():
    assert flows.follow_the_money(panel(2), ["A", "B"])["summary"] == {}


# ---------------------------------------------------------------- style estimate
def _segments(seed=11, n=600):
    idx = pd.bdate_range(end="2026-10-30", periods=n)
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.normal(0.0004, 0.01, (n, 3)), index=idx, columns=["Large Cap", "Mid Cap", "Small Cap"])
    r["Cash"] = 0.0002
    return r


def test_style_estimate_recovers_a_known_mix():
    r = _segments()
    fund = (r @ np.array([0.6, 0.3, 0.1, 0.0]))
    out = style.estimate(100 * (1 + fund).cumprod(), 100 * (1 + r).cumprod(), weeks=104)
    assert out.weights["Large Cap"] == pytest.approx(60, abs=2)
    assert out.weights["Mid Cap"] == pytest.approx(30, abs=2)
    assert out.weights["Small Cap"] == pytest.approx(10, abs=2)
    assert out.weights.sum() == pytest.approx(100) and out.r_squared > 0.99


def test_style_estimate_refuses_short_histories():
    r = _segments(n=60)
    fund = r["Large Cap"]
    assert style.estimate(100 * (1 + fund).cumprod(), 100 * (1 + r).cumprod()) is None


def test_validation_measures_error_against_real_holdings():
    r = _segments()
    factors = 100 * (1 + r).cumprod()
    nav = pd.DataFrame({"F": 100 * (1 + r @ np.array([0.8, 0.2, 0.0, 0.0])).cumprod()})
    m = pd.Timestamp("2026-09-30")
    h = pd.DataFrame([holding("F", m, "I1", "a", 1, 1, cap="Large Cap", pct=70.0),
                      holding("F", m, "I2", "b", 1, 1, cap="Mid Cap", pct=30.0)])
    t = style.validate(nav, factors, h, ["F"])
    errs = style.validation_errors(t)
    assert t.loc[0, "actual Large Cap"] == 70 and errs["Large Cap"] == pytest.approx(10, abs=2.5)
    assert errs["Mid Cap"] == pytest.approx(10, abs=2.5) and errs["overall"] < 10


# ---------------------------------------------------------------- walk-forward
def test_walk_forward_never_uses_future_returns():
    """Rewriting the last year must not change a single earlier out-of-sample return."""
    navs = pd.DataFrame({f"F{i}": make_nav(i, years=8) for i in range(4)})
    full = pf.walk_forward(navs)
    cut = navs.index[-252]
    changed = navs.copy()
    changed.loc[cut:] *= np.linspace(1, 3, len(changed.loc[cut:]))[:, None]
    other = pf.walk_forward(changed)
    last_clean = cut - pd.DateOffset(months=7)
    a = full["weekly"].loc[:last_clean]
    b = other["weekly"].loc[:last_clean]
    assert len(a) > 100
    pd.testing.assert_frame_equal(a, b)


def test_equal_weight_leg_is_the_plain_average():
    navs = pd.DataFrame({f"F{i}": make_nav(i, years=6) for i in range(3)})
    out = pf.walk_forward(navs)
    weekly = navs.resample("W-FRI").last().pct_change().dropna()
    idx = out["weekly"].index
    assert out["weekly"]["Equal weight"].to_numpy() == pytest.approx(weekly.loc[idx].mean(axis=1).to_numpy())
    assert list(out["summary"].index) == ["Max Sharpe", "Min volatility", "Equal weight"]


def test_walk_forward_needs_more_history_than_it_trains_on():
    with pytest.raises(ValueError):
        pf.walk_forward(pd.DataFrame({"A": make_nav(1, years=2), "B": make_nav(2, years=2)}))


def test_optimiser_failure_is_an_error_not_an_arbitrary_answer():
    with pytest.raises(ValueError, match="converge"):
        pf._solve(lambda w: float("nan"), 3)


# ---------------------------------------------------------------- index comparison and wording
def _funds():
    return pd.DataFrame({
        "category": ["Equity Scheme - Small Cap Fund", "Equity Scheme - Large Cap Fund", "Other Scheme - Index Funds",
                     "Other Scheme - Index Funds", "Other Scheme - Index Funds"],
        "is_benchmark": [0, 0, 1, 1, 1], "label": ["Small fund", "Large fund", "Smallcap", "Nifty 100", "Nifty 50"]},
        index=["S", "L", universe.SMALLCAP_250, universe.NIFTY_100, universe.BENCHMARK_CODE])


def test_funds_are_compared_with_their_own_category_index():
    nav = pd.DataFrame({c: make_nav(i, years=7) for i, c in enumerate(_funds().index)})
    t, s = evidence.funds_vs_index(_funds(), nav)
    row = t.set_index("scheme_code").loc["S"]
    assert row["index"] == universe.by_code(universe.SMALLCAP_250).label
    assert row["active_3y"] == pytest.approx(perf.period_return(nav["S"], 3) - perf.period_return(nav[universe.SMALLCAP_250], 3))
    assert s["funds_3y"] == 2


def test_flow_sentence_says_what_the_numbers_support():
    strong = {"spread": 0.01, "spread_t": 3.0, "months": 34}
    weak = {"spread": 0.002, "spread_t": 0.5, "months": 34}
    assert "statistically clear" in report.flow_sentence(strong, weak)
    assert "little left to capture" in report.flow_sentence(strong, weak)
    assert "short of" in report.flow_sentence(weak, weak)
    assert "long enough" in report.flow_sentence(strong, strong)


def test_optimiser_sentence_says_what_the_numbers_support():
    s = pd.DataFrame({"return": [0.15, 0.16]}, index=["Max Sharpe", "Equal weight"])
    base = {"summary": s, "funds": ["a", "b"]}
    assert "No evidence" in report.optimiser_sentence({**base, "gap_t": -0.2, "gap_vs_equal": -0.01})
    assert "lost to" in report.optimiser_sentence({**base, "gap_t": -3.0, "gap_vs_equal": -0.01})
    assert "It beat" in report.optimiser_sentence({**base, "gap_t": 3.0, "gap_vs_equal": 0.01})
