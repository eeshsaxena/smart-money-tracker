"""Dash app factory. Run with `python -m smt serve`, or `gunicorn smt.dashboard.app:server`."""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from smt import universe
from smt.analytics import holdings as H
from smt.analytics import performance as perf
from smt.analytics import portfolio as pf
from smt.analytics import sip as sip_mod
from smt.dashboard import figures as fig
from smt.dashboard.data import Data

PCT = Format(precision=1, scheme=Scheme.percentage)
NUM2 = Format(precision=2, scheme=Scheme.fixed)
NUM1 = Format(precision=1, scheme=Scheme.fixed)
NUM0 = Format(precision=0, scheme=Scheme.fixed).group(True)
PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "Max": None}


# --------------------------------------------------------------------------- #
# Small layout helpers
# --------------------------------------------------------------------------- #
def kpi(label: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="text-muted small"),
        html.Div(value, className="fs-4 fw-semibold"),
        html.Div(sub, className="small text-muted"),
    ])), xs=6, md=3, className="mb-3")


def pct(v, digits: int = 1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


def table(df: pd.DataFrame, id_: str | None = None, formats: dict | None = None, page_size: int = 15,
          **kw) -> dash_table.DataTable:
    formats = formats or {}
    cols = [{"name": c, "id": c, "type": "numeric" if c in formats else "text",
             **({"format": formats[c]} if c in formats else {})} for c in df.columns]
    return dash_table.DataTable(
        id=id_ or f"t{id(df)}", data=df.to_dict("records"), columns=cols, page_size=page_size,
        sort_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif", "fontSize": 13, "padding": "6px 10px", "textAlign": "left",
                    "maxWidth": 320, "whiteSpace": "normal"},
        style_header={"fontWeight": 600, "backgroundColor": "#f1f5f9"}, **kw)


def note(text: str) -> html.P:
    return html.P(text, className="text-muted small mb-2")


def section(title: str, *children) -> html.Div:
    return html.Div([html.H5(title, className="mt-4 mb-2"), *children])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(db_path: Path | str | None = None) -> Dash:
    data = Data.load(db_path)
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Smart Money Tracker",
               suppress_callback_exceptions=True)

    if data.empty:
        app.layout = dbc.Container([html.H2("Smart Money Tracker", className="mt-4"),
                                    dbc.Alert("No data yet. Run `python -m smt refresh` first.", color="warning")])
        return app

    funds = data.funds
    opts = data.fund_options()
    opts_b = data.fund_options(include_benchmark=True)
    hcodes = data.holdings_codes()
    if "122639" in hcodes:                       # open on PPFAS Flexi Cap, the best-known fund here
        hcodes = ["122639"] + [c for c in hcodes if c != "122639"]
    hopts = [{"label": data.label(c), "value": c} for c in hcodes]
    score = data.scorecard()
    nav_date = data.nav.index.max().date()
    latest_holdings = data.holdings["as_of"].max().date() if not data.holdings.empty else None

    header = dbc.Row([
        dbc.Col([html.H2("Smart Money Tracker", className="mb-0"),
                 html.Div(f"{len(funds) - 1} Indian equity mutual funds, verified against AMFI. "
                          f"NAV as of {nav_date}"
                          + (f"; portfolios as of {latest_holdings}" if latest_holdings else "") + ".",
                          className="text-muted")], md=9),
    ], className="mt-4 mb-3")

    # ---------------------------------------------------------------- tabs
    t_score = dbc.Tab(label="Scorecard", tab_id="score", children=[
        note("Returns over 1 year are annualised. Risk figures use the last 3 years of daily NAVs; "
             f"benchmark is the {universe.BENCHMARK_NAME}. Rank is within this app's funds of the same SEBI category."),
        dcc.Dropdown(id="score-cat", multi=True, placeholder="Filter by category",
                     options=sorted(score["Category"].unique())),
        html.Div(id="score-table", className="mt-3"),
    ])

    t_fund = dbc.Tab(label="Fund", tab_id="fund", children=[
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="fund", options=opts, value="122639", clearable=False), md=6),
            dbc.Col(dbc.RadioItems(id="fund-period", options=list(PERIODS), value="5Y", inline=True), md=6),
        ], className="mt-3"),
        html.Div(id="fund-kpis", className="mt-3"),
        dbc.Row([dbc.Col(dcc.Graph(id="fund-growth"), lg=6), dbc.Col(dcc.Graph(id="fund-rolling"), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id="fund-dd"), lg=6), dbc.Col(dcc.Graph(id="fund-cal"), lg=6)]),
    ])

    years = sorted({d.year for d in data.nav.index})
    t_sip = dbc.Tab(label="SIP", tab_id="sip", children=[
        note("A monthly SIP on the first business day of each month, valued at the latest NAV. "
             "XIRR is the annualised return that accounts for when each instalment went in."),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="sip-funds", options=opts_b, multi=True,
                                 value=["122639", "119076", "118955", universe.BENCHMARK_CODE]), md=6),
            dbc.Col(dbc.InputGroup([dbc.InputGroupText("₹ / month"),
                                    dbc.Input(id="sip-amount", type="number", value=10000, min=500, step=500)]), md=3),
            dbc.Col(dcc.Dropdown(id="sip-start", options=years, value=max(years[0], years[-1] - 5),
                                 clearable=False), md=3),
        ], className="mt-3"),
        html.Div(id="sip-table", className="mt-3"),
        dcc.Graph(id="sip-chart"),
    ])

    t_port = dbc.Tab(label="Portfolio builder", tab_id="port", children=[
        note("Pick 2-8 funds. Uses 3 years of weekly returns. Weights are long-only and add up to 100%. "
             "Past returns drive the optimiser, so treat the weights as a description of the past, not advice."),
        dcc.Dropdown(id="port-funds", options=opts_b, multi=True, className="mt-2",
                     value=["122639", "119076", "118989", "119212", universe.BENCHMARK_CODE]),
        dbc.Row([dbc.Col(dcc.Graph(id="port-frontier"), lg=7), dbc.Col(html.Div(id="port-weights"), lg=5)]),
        dcc.Graph(id="port-corr"),
    ])

    t_hold = dbc.Tab(label="Holdings", tab_id="hold", children=[
        note("From each AMC's official month-end portfolio disclosure. 'Equity' means Indian stocks held outright; "
             "arbitrage positions (hedged with futures) are shown separately. Market-cap classes are AMFI's list"
             f" for the six months ended {data.market_cap_label or 'n/a'}."),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="hold-fund", options=hopts, value=hcodes[0] if hcodes else None,
                                 clearable=False), md=6),
            dbc.Col(dcc.Dropdown(id="hold-month", clearable=False), md=3),
        ], className="mt-2"),
        html.Div(id="hold-kpis", className="mt-3"),
        dbc.Row([dbc.Col(dcc.Graph(id="hold-assets"), lg=4), dbc.Col(dcc.Graph(id="hold-caps"), lg=4),
                 dbc.Col(dcc.Graph(id="hold-sectors"), lg=4)]),
        section("Top holdings", html.Div(id="hold-top")),
        section("What changed since the previous month",
                note("Classified by number of shares, so price moves alone never count as buying or selling. "
                     "A share count that doubles while the weight stays flat is flagged as a likely split or bonus."),
                html.Div(id="hold-changes")),
        section("Style drift", dcc.Graph(id="hold-drift"), html.Div(id="hold-drift-note")),
    ])

    t_flow = dbc.Tab(label="Smart money", tab_id="flow", children=[
        note(f"Net buying and selling across the {len(hcodes)} funds with portfolio data, in ₹ crore "
             "(change in shares × share price). Funds buying = new positions plus additions."),
        dcc.Dropdown(id="flow-month", clearable=False, className="mt-2", style={"maxWidth": 260},
                     options=[d.date().isoformat() for d in sorted(data.holdings["as_of"].unique(), reverse=True)][:-1]
                     if not data.holdings.empty else [],
                     value=latest_holdings.isoformat() if latest_holdings else None),
        dbc.Row([dbc.Col(dcc.Graph(id="flow-buy"), lg=6), dbc.Col(dcc.Graph(id="flow-sell"), lg=6)]),
        html.Div(id="flow-table"),
    ])

    t_overlap = dbc.Tab(label="Overlap", tab_id="overlap", children=[
        note("Overlap = sum, over stocks both funds hold, of the smaller of the two weights. "
             "50% means half of each portfolio is effectively the same."),
        dbc.Row([dbc.Col(dcc.Dropdown(id="ov-a", options=hopts, value=hcodes[0] if hcodes else None,
                                      clearable=False), md=6),
                 dbc.Col(dcc.Dropdown(id="ov-b", options=hopts, value=hcodes[1] if len(hcodes) > 1 else None,
                                      clearable=False), md=6)], className="mt-2"),
        html.Div(id="ov-kpis", className="mt-3"),
        html.Div(id="ov-table"),
        dcc.Graph(id="ov-matrix", figure=fig.heatmap(
            H.overlap_matrix(data.holdings, hcodes).rename(index=data.label, columns=data.label),
            "Overlap between every pair of funds (% of portfolio)", fmt=".0f", zmin=0, zmax=100)
            if len(hcodes) > 1 else fig.empty("Need two funds with holdings")),
    ])

    status_rows = data.checks.rename(columns=str.title)
    verify = funds.reset_index()[["scheme_code", "scheme_name", "expected_name", "category", "holdings_source",
                                  "verified", "verified_at"]]
    files = data.files[["connector", "as_of", "schemes_matched", "bytes", "status", "url"]] if not data.files.empty \
        else pd.DataFrame()
    t_status = dbc.Tab(label="Data status", tab_id="status", children=[
        section("Checks", table(status_rows, "status-checks", page_size=20,
                                style_data_conditional=[
                                    {"if": {"filter_query": '{Status} = "fail"'}, "backgroundColor": "#fee2e2"},
                                    {"if": {"filter_query": '{Status} = "warn"'}, "backgroundColor": "#fef9c3"}])),
        section("Funds and AMFI verification", table(verify, "status-funds", page_size=25)),
        section("Refresh log", table(data.log, "status-log", page_size=10)),
        section("Disclosure files used", table(files, "status-files", page_size=10)),
    ])

    app.layout = dbc.Container([
        header,
        dbc.Tabs([t_score, t_fund, t_sip, t_port, t_hold, t_flow, t_overlap, t_status], active_tab="score"),
        html.Footer(note("Data: AMFI (scheme master, market-cap list), mfapi.in (NAV history), "
                         "PPFAS and DSP portfolio disclosures. Not investment advice."), className="my-4"),
    ], fluid=True, style={"maxWidth": 1400})

    register_callbacks(app, data, score)
    return app


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def register_callbacks(app: Dash, data: Data, score: pd.DataFrame) -> None:
    funds, nav, h = data.funds, data.nav, data.holdings

    @app.callback(Output("score-table", "children"), Input("score-cat", "value"))
    def score_table(cats):
        df = score if not cats else score[score["Category"].isin(cats)]
        df = df.drop(columns=["code"])
        return table(df, "score-dt", page_size=30, formats={
            "1Y": PCT, "3Y": PCT, "5Y": PCT, "Volatility": PCT, "Max drawdown": PCT, "Alpha": PCT,
            "Sharpe": NUM2, "Beta": NUM2},
            style_data_conditional=[{"if": {"filter_query": f'{{Fund}} = "{universe.BENCHMARK_NAME}"'},
                                     "fontStyle": "italic", "backgroundColor": "#f8fafc"}])

    @app.callback(Output("fund-kpis", "children"), Output("fund-growth", "figure"), Output("fund-rolling", "figure"),
                  Output("fund-dd", "figure"), Output("fund-cal", "figure"),
                  Input("fund", "value"), Input("fund-period", "value"))
    def fund_view(code, period):
        s, b = nav[code].dropna(), data.bench.dropna()
        name, bname = data.label(code), universe.BENCHMARK_NAME
        yrs = PERIODS[period]
        start = s.index[0] if yrs is None else max(s.index[0], s.index[-1] - pd.DateOffset(years=yrs))
        both = pd.concat({name: s, bname: b}, axis=1).loc[start:].dropna()
        tr, rm = perf.trailing_returns(s), perf.risk_metrics(s, b)
        kpis = dbc.Row([
            kpi("Category", funds.at[code, "category_short"], funds.at[code, "amc"]),
            kpi("3Y return (annualised)", pct(tr["3Y"]), f"benchmark {pct(perf.period_return(b, 3))}"),
            kpi("3Y Sharpe / beta", "–" if rm.get("sharpe") is None else f"{rm['sharpe']:.2f} / {rm.get('beta', 0):.2f}",
                f"alpha {pct(rm.get('alpha'))}"),
            kpi("Worst fall (3Y)", pct(rm.get("max_drawdown")), f"volatility {pct(rm.get('volatility'))}"),
        ])
        g = fig.growth(perf.growth_of(both, both.index[0]), f"Growth of ₹10,000 since {both.index[0].date()}") \
            if len(both) > 1 else fig.empty("Not enough history")
        roll = pd.concat({name: perf.rolling_returns(s, 3), bname: perf.rolling_returns(b, 3)}, axis=1).dropna()
        r = fig.lines(roll, "Rolling 3-year return (annualised)", colors=[fig.ACCENT, fig.BENCH]) \
            if len(roll) else fig.empty("Needs 3 years of history")
        dd = fig.drawdown(pd.concat({name: perf.drawdown(s.loc[start:]), bname: perf.drawdown(b.loc[start:])},
                                    axis=1), "Fall from previous peak")
        cal = pd.concat({name: perf.calendar_year_returns(s), bname: perf.calendar_year_returns(b)}, axis=1).dropna()
        c = fig.calendar_bars(cal.tail(10), "Calendar-year returns (current year to date)") \
            if len(cal) else fig.empty("Needs a full calendar year")
        return kpis, g, r, dd, c

    @app.callback(Output("sip-table", "children"), Output("sip-chart", "figure"),
                  Input("sip-funds", "value"), Input("sip-amount", "value"), Input("sip-start", "value"))
    def sip_view(codes, amount, start_year):
        if not codes or not amount:
            return note("Pick at least one fund."), fig.empty("Pick a fund")
        rows, paths = [], {}
        for code in codes:
            s = nav[code].dropna()
            res = sip_mod.simulate(s, float(amount), pd.Timestamp(int(start_year), 1, 1))
            if res.installments.empty:
                continue
            rows.append({"Fund": data.label(code), "First instalment": res.installments["date"].iloc[0].date().isoformat(),
                         "Instalments": len(res.installments), "Invested (₹)": res.invested,
                         "Value (₹)": res.value, "Gain (₹)": res.value - res.invested, "XIRR": res.xirr})
            paths[data.label(code)] = sip_mod.value_path(s, res)["value"]
        if not rows:
            return note("No NAV history in that period."), fig.empty("No data")
        df = pd.DataFrame(rows).sort_values("XIRR", ascending=False)
        chart = fig.lines(pd.DataFrame(paths).ffill(), "Value of the SIP over time", pct=False)
        chart.update_yaxes(tickprefix="₹", tickformat=",.0f")
        return table(df, "sip-dt", formats={"Invested (₹)": NUM0, "Value (₹)": NUM0, "Gain (₹)": NUM0, "XIRR": PCT}), chart

    @app.callback(Output("port-frontier", "figure"), Output("port-weights", "children"),
                  Output("port-corr", "figure"), Input("port-funds", "value"))
    def port_view(codes):
        codes = (codes or [])[:8]
        if len(codes) < 2:
            msg = "Pick at least two funds"
            return fig.empty(msg), note(msg), fig.empty(msg)
        sub = nav[codes].dropna(how="any")
        labels = {c: data.label(c) for c in codes}
        try:
            opt = pf.optimise(sub)
        except ValueError as e:
            return fig.empty(str(e)), note(str(e)), fig.empty(str(e))
        w = pd.DataFrame({"Fund": [labels[c] for c in codes],
                          "Max Sharpe": opt["max_sharpe"]["weights"].reindex(codes).values,
                          "Min volatility": opt["min_volatility"]["weights"].reindex(codes).values})
        summary = pd.DataFrame({
            "Fund": ["Expected return", "Volatility", "Sharpe"],
            "Max Sharpe": [opt["max_sharpe"]["return"], opt["max_sharpe"]["volatility"], opt["max_sharpe"]["sharpe"]],
            "Min volatility": [opt["min_volatility"]["return"], opt["min_volatility"]["volatility"],
                               opt["min_volatility"]["sharpe"]]})
        weights = html.Div([html.H6("Suggested weights", className="mt-4"),
                            table(w, "port-w", formats={"Max Sharpe": PCT, "Min volatility": PCT}),
                            table(summary, "port-s", formats={"Max Sharpe": NUM2, "Min volatility": NUM2}),
                            note(f"Based on {opt['weeks']} weeks of common history.")])
        corr = pf.correlation(sub).rename(index=labels, columns=labels)
        return fig.frontier(opt, labels), weights, fig.heatmap(corr, "Correlation of weekly returns", zmin=0, zmax=1)

    @app.callback(Output("hold-month", "options"), Output("hold-month", "value"), Input("hold-fund", "value"))
    def hold_months(code):
        ms = [m.date().isoformat() for m in H.months(h, code)] if code else []
        return ms, (ms[0] if ms else None)

    @app.callback(Output("hold-kpis", "children"), Output("hold-assets", "figure"), Output("hold-caps", "figure"),
                  Output("hold-sectors", "figure"), Output("hold-top", "children"), Output("hold-changes", "children"),
                  Output("hold-drift", "figure"), Output("hold-drift-note", "children"),
                  Input("hold-fund", "value"), Input("hold-month", "value"))
    def hold_view(code, month):
        if not code or not month:
            e = fig.empty("No holdings")
            return None, e, e, e, None, None, e, None
        as_of = pd.Timestamp(month)
        snap = H.snapshot(h, code, as_of)
        conc = H.concentration(snap)
        mix = H.asset_mix(snap)
        kpis = dbc.Row([
            kpi("Stocks held", f"{conc['stocks']}", f"≈{conc['effective_stocks']:.0f} equal-weight equivalent"),
            kpi("Top 10 stocks", f"{conc['top10_pct']:.1f}%", "of net assets"),
            kpi("Indian equity", f"{mix.get('Equity', 0):.1f}%", f"foreign {mix.get('Foreign Equity', 0):.1f}%"),
            kpi("Portfolio date", month, funds.at[code, "category_short"]),
        ])
        top = snap[snap["asset_class"].isin(H.STOCKS)].nlargest(15, "pct_nav")[
            ["instrument", "industry", "cap_class", "pct_nav", "market_value_lakh"]].rename(columns={
                "instrument": "Stock", "industry": "Industry", "cap_class": "Cap", "pct_nav": "% of NAV",
                "market_value_lakh": "Value (₹ lakh)"})
        ch = H.changes(h, code, as_of)
        if ch.empty:
            changes_view = note("Needs an earlier month to compare with.")
        else:
            counts = ch["status"].value_counts()
            moved = ch[ch["status"] != "Unchanged"].assign(flow=lambda d: d["flow_cr"])[
                ["instrument", "status", "quantity_prev", "quantity", "pct_nav_prev", "pct_nav", "flow"]].rename(
                columns={"instrument": "Stock", "status": "Change", "quantity_prev": "Shares before",
                         "quantity": "Shares now", "pct_nav_prev": "Weight before %", "pct_nav": "Weight now %",
                         "flow": "Bought (+) / sold (−), ₹ cr"})
            changes_view = html.Div([
                note(" · ".join(f"{k}: {v}" for k, v in counts.items()) + f"  (vs {ch['prev'].iloc[0].date()})"),
                table(moved, "hold-ch", formats={"Shares before": NUM0, "Shares now": NUM0, "Weight before %": NUM2,
                                                 "Weight now %": NUM2, "Bought (+) / sold (−), ₹ cr": NUM1})])
        drift = H.style_drift(h, code, funds.at[code, "category"])
        rule = drift["rule"].iloc[0] if len(drift) else ""
        fails = drift[drift["meets_rule"] == False]  # noqa: E712 - None means "no rule"
        drift_note = note("No SEBI market-cap rule for this category." if not rule else
                          (f"Met the rule ({rule}) in all {len(drift)} months." if fails.empty else
                           f"Below the rule ({rule}) in: {', '.join(d.date().isoformat() for d in fails['as_of'])}. "
                           "AMFI's list is updated twice a year and funds get time to rebalance, "
                           "so a small shortfall is a flag to look at, not proof of a breach."))
        return (kpis, fig.hbar(mix, "Asset mix (% of NAV)"),
                fig.donut(H.cap_mix(snap), "Market-cap mix (% of NAV)"),
                fig.hbar(H.sector_mix(snap).head(12), "Top sectors (% of NAV)"),
                table(top, "hold-top-dt", formats={"% of NAV": NUM2, "Value (₹ lakh)": NUM0}),
                changes_view, fig.drift(drift, rule), drift_note)

    @app.callback(Output("flow-buy", "figure"), Output("flow-sell", "figure"), Output("flow-table", "children"),
                  Input("flow-month", "value"))
    def flow_view(month):
        df = H.smart_money_flow(h, data.holdings_codes(), pd.Timestamp(month) if month else None)
        if df.empty:
            e = fig.empty("Needs two months of holdings")
            return e, e, None
        buys = df[df["net_flow_cr"] > 0].head(15)
        sells = df[df["net_flow_cr"] < 0].tail(15)
        t = df[["instrument", "cap_class", "industry", "net_flow_cr", "funds_buying", "funds_selling",
                "new_positions", "full_exits", "funds_holding"]].rename(columns={
                    "instrument": "Stock", "cap_class": "Cap", "industry": "Industry",
                    "net_flow_cr": "Net ₹ cr", "funds_buying": "Buying", "funds_selling": "Selling",
                    "new_positions": "New", "full_exits": "Exits", "funds_holding": "Holding"})
        return (fig.flows(buys, "Most bought"), fig.flows(sells, "Most sold"),
                section("All stocks", table(t, "flow-dt", formats={"Net ₹ cr": NUM1}, page_size=20,
                                            filter_action="native")))

    @app.callback(Output("ov-kpis", "children"), Output("ov-table", "children"),
                  Input("ov-a", "value"), Input("ov-b", "value"))
    def overlap_view(a, b):
        if not a or not b:
            return None, None
        o = H.overlap(H.snapshot(h, a), H.snapshot(h, b))
        kpis = dbc.Row([kpi("Weighted overlap", f"{o['weighted_pct']:.1f}%"),
                        kpi("Stocks in common", str(o["common_stocks"])),
                        kpi("Share of distinct stocks", f"{o['jaccard'] * 100:.0f}%", "common / all held")])
        d = o["detail"][["instrument", "weight_a", "weight_b", "overlap"]].rename(columns={
            "instrument": "Stock", "weight_a": f"{data.label(a)} %", "weight_b": f"{data.label(b)} %",
            "overlap": "Overlap %"})
        return kpis, table(d, "ov-dt", formats={c: NUM2 for c in d.columns[1:]})


def _server():
    """WSGI entry point for gunicorn: `gunicorn smt.dashboard.app:server`."""
    return create_app().server


def __getattr__(name):     # build the app lazily so importing this module stays cheap
    if name == "server":
        return _server()
    raise AttributeError(name)
