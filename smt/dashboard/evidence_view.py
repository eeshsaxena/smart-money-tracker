"""The Evidence tab: four claims tested on the app's own data. Layout only; the numbers come from analytics/evidence."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from smt import report
from smt.dashboard import figures as fig
from smt.dashboard.components import NUM1, NUM2, PCT, PCT0, PCT2, finding, kpi, note, pct, section, table


def _index_section(ev: dict) -> html.Div:
    t, s = ev["index_table"], ev["index_summary"]
    if t.empty or not s:
        return section("Do these funds beat the right index?", note("Needs 3 years of history."))
    view = t[["fund", "index", "return_3y", "index_3y", "active_3y", "hit_rate", "information_ratio"]].rename(columns={
        "fund": "Fund", "index": "Compared with", "return_3y": "3Y return", "index_3y": "Index 3Y",
        "active_3y": "Ahead of index", "hit_rate": "Beat index (3Y windows)", "information_ratio": "Info ratio"})
    view["Compared with"] = view["Compared with"].str.split(" (", regex=False).str[0]
    return section(
        "Do these funds beat the right index?",
        finding(report.index_sentence(s)),
        dbc.Row([
            kpi("Beat their index over 3Y", f"{s['beat_3y']} of {s['funds_3y']}",
                f"median {pct(s['median_active_3y'], signed=True)} a year"),
            kpi("Beat their index over 5Y", f"{s['beat_5y']} of {s['funds_5y']}",
                f"median {pct(s['median_active_5y'], signed=True)} a year"),
            kpi("Median alpha vs Nifty 50", pct(s["median_alpha_vs_nifty50"]), "the flattering comparison"),
            kpi("Median alpha vs right index", pct(s["median_alpha"]), "the fair comparison"),
        ]),
        dcc.Graph(figure=fig.diverging_bars(t.dropna(subset=["active_3y"]).set_index("fund")["active_3y"],
                                            "Annual return ahead of the category index, last 3 years")),
        table(view, "ev-index", page_size=25, formats={"3Y return": PCT, "Index 3Y": PCT, "Ahead of index": PCT,
                                                       "Beat index (3Y windows)": PCT0, "Info ratio": NUM2}),
        note(report.INDEX_CAVEAT),
    )


def _flow_section(ev: dict) -> html.Div:
    next_month, later = ev["flows"]["next_month"], ev["flows"]["after_a_month"]
    s, s2 = next_month.get("summary") or {}, later.get("summary") or {}
    if not s:
        return section("Does following the money work?", note("Needs more months of holdings."))
    monthly = next_month["monthly"].set_index("month")[["bought", "sold", "held"]].rename(columns={
        "bought": "Stocks bought", "sold": "Stocks sold", "held": "All stocks held"})
    growth = (1 + monthly).cumprod()
    return section(
        "Does following the money work?",
        finding(report.flow_sentence(s, s2)),
        dbc.Row([
            kpi("Bought minus sold, next month", pct(s["spread"], 2, signed=True),
                f"t = {s['spread_t']:.2f}, {s['months']} months"),
            kpi("Same, one month later", pct(s2.get("spread"), 2, signed=True),
                f"t = {s2.get('spread_t', float('nan')):.2f}"),
            kpi("Months bought beat sold", pct(s["spread_hit_rate"], 0), ""),
            kpi("Sold stocks with a price", pct(s["coverage_sold"], 0), "the rest left every tracked fund"),
        ]),
        dcc.Graph(figure=fig.lines(growth, "Growth of 1 in each basket, held for the month after each disclosure",
                                   pct=False)),
        *_robustness(ev["flows"].get("robustness")),
        note(report.FLOW_CAVEAT),
    )


def _robustness(r) -> list:
    if r is None or r.empty:
        return []
    view = r[["variant", "months", "spread", "t", "ci_low", "ci_high", "note"]].rename(columns={
        "variant": "Variant", "months": "Months", "spread": "Bought minus sold", "ci_low": "95% low",
        "ci_high": "95% high", "note": "Note"})
    return [html.H6("Does it hold up?", className="mt-3"), finding(report.robustness_sentence(r)),
            table(view, "ev-robust", formats={"Bought minus sold": PCT2, "95% low": PCT2, "95% high": PCT2,
                                               "t": NUM2})]


def _style_section(ev: dict) -> html.Div:
    st = ev["style"]
    if st["validation"].empty:
        return section("Can NAV alone reveal a fund's style?", note("Needs funds with both NAV and holdings."))
    v = st["validation"].drop(columns=["scheme_code", "weeks"]).rename(columns={"fund": "Fund", "r_squared": "R²"})
    formats = {c: NUM1 for c in v.columns if c.startswith(("est", "actual"))} | {"R²": NUM2}
    children = [
        finding(report.style_sentence(st)),
        table(v, "ev-style", page_size=15, formats=formats),
    ]
    if not st["nav_only"].empty:
        only = st["nav_only"].rename(columns={"fund": "Fund", "r_squared": "R²"})
        children += [html.H6("Estimates for funds with no usable disclosure", className="mt-3"),
                     table(only, "ev-style-nav", page_size=15,
                           formats={c: NUM1 for c in only.columns if c not in ("Fund", "R²")} | {"R²": NUM2})]
    children.append(note(report.STYLE_CAVEAT))
    return section("Can NAV alone reveal a fund's style?", *children)


def _optimiser_section(ev: dict) -> html.Div:
    o = ev["optimiser"]
    if not o:
        return section("Does the optimiser beat equal weight?", note("Needs more history than the training window."))
    s = o["summary"].reset_index().rename(columns={"index": "Portfolio", "return": "Return", "volatility": "Volatility",
                                                   "sharpe": "Sharpe", "max_drawdown": "Worst fall"})
    growth = (1 + o["weekly"]).cumprod()
    return section(
        "Does the optimiser beat equal weight?",
        finding(report.optimiser_sentence(o)),
        table(s, "ev-opt", formats={"Return": PCT, "Volatility": PCT, "Sharpe": NUM2, "Worst fall": PCT}),
        dcc.Graph(figure=fig.lines(growth, "Growth of 1, with weights refitted every 6 months on past data only",
                                   pct=False)),
        note(report.OPTIMISER_CAVEAT),
    )


def build(evidence: dict) -> list:
    return [_index_section(evidence), _flow_section(evidence), _style_section(evidence),
            _optimiser_section(evidence)]

