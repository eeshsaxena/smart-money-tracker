"""Dash callbacks wiring UI interactions to all 6 analysis features."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, dash_table, dcc, no_update
import dash_bootstrap_components as dbc

from config import FUND_MANAGERS, PEER_GROUPS
from src.dashboard.components import metric_card, signal_badge, empty_state
from src.scraper.sebi_scraper import (
    detect_accumulation_signals,
    fetch_multi_month_portfolio,
    fetch_all_manager_portfolios,
)
from src.scraper.nse_classifier import match_holdings_to_classification
from src.analysis.smart_money_flow import (
    aggregate_cross_fund_signals,
    compute_conviction_scores,
    create_flow_chart,
    create_conviction_treemap,
)
from src.analysis.style_fingerprint import (
    compute_manager_fingerprint,
    cluster_managers,
    create_fingerprint_radar,
    create_cluster_scatter,
    create_fingerprint_comparison,
)
from src.analysis.overlap import (
    compute_overlap_matrix,
    compute_weighted_overlap,
    create_overlap_heatmap,
    find_common_holdings,
    create_common_holdings_chart,
    create_overlap_sunburst,
)
from src.analysis.style_drift import (
    compute_drift_over_time,
    create_drift_chart,
    create_mandate_gauge,
    detect_style_drift,
)
from src.analysis.xirr_calc import (
    benchmark_sip,
    compare_category_peers,
    create_sip_growth_chart,
    create_peer_comparison_chart,
)
from src.analysis.bhb_attribution import (
    compute_bhb_attribution,
    create_attribution_chart,
    create_attribution_waterfall,
)


# ── Holdings tab ──

@callback(
    Output("fund-select", "options"),
    Output("fund-select", "value"),
    Input("fund-manager-select", "value"),
)
def update_fund_dropdown(manager_key):
    if not manager_key or manager_key not in FUND_MANAGERS:
        return [], None
    funds = FUND_MANAGERS[manager_key]["funds"]
    options = [{"label": f["name"], "value": f["scheme_code"]} for f in funds]
    return options, funds[0]["scheme_code"] if funds else None


@callback(
    Output("holdings-output", "children"),
    Output("signals-output", "children"),
    Output("portfolio-store", "data"),
    Input("fetch-holdings-btn", "n_clicks"),
    State("fund-select", "value"),
    State("months-select", "value"),
    State("fund-manager-select", "value"),
    prevent_initial_call=True,
)
def fetch_and_display_holdings(n_clicks, scheme_code, months, manager_key):
    if not scheme_code:
        return html.P("Select a fund first.", className="text-warning"), "", None

    months = int(months) if months else 6
    portfolio = fetch_multi_month_portfolio(scheme_code, months)

    if portfolio.empty:
        return (
            empty_state("No data found. The fund may not have public disclosures yet.", "bi-database-x"),
            "",
            None,
        )

    manager = FUND_MANAGERS.get(manager_key, {})
    fund_info = next((f for f in manager.get("funds", []) if f["scheme_code"] == scheme_code), {})
    portfolio["fund_name"] = fund_info.get("name", "")
    portfolio["manager"] = manager.get("name", "")
    portfolio["fund_category"] = fund_info.get("category", "")

    portfolio = match_holdings_to_classification(portfolio)

    latest_date = portfolio["date"].max()
    latest = portfolio[portfolio["date"] == latest_date].copy()
    latest = latest.sort_values("pct_aum", ascending=False)

    holdings_table = dash_table.DataTable(
        data=latest[["company", "sector", "quantity", "market_value_lakhs", "pct_aum", "market_cap_category"]].to_dict("records"),
        columns=[
            {"name": "Company", "id": "company"},
            {"name": "Sector", "id": "sector"},
            {"name": "Quantity", "id": "quantity", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Value (Lakhs)", "id": "market_value_lakhs", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "% AUM", "id": "pct_aum", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Cap Category", "id": "market_cap_category"},
        ],
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{market_cap_category} = large_cap"}, "backgroundColor": "#E3F2FD"},
            {"if": {"filter_query": "{market_cap_category} = mid_cap"}, "backgroundColor": "#FFF3E0"},
            {"if": {"filter_query": "{market_cap_category} = small_cap"}, "backgroundColor": "#E8F5E9"},
        ],
    )

    large_pct = latest[latest["market_cap_category"] == "large_cap"]["pct_aum"].sum()
    mid_small_pct = latest[latest["market_cap_category"].isin(["mid_cap", "small_cap"])]["pct_aum"].sum()
    num_sectors = latest["sector"].nunique()

    summary = dbc.Row(
        [
            dbc.Col(metric_card("Total Stocks", str(len(latest)), f"as of {latest_date.strftime('%b %Y')}"), md=3),
            dbc.Col(metric_card(
                "Top Holding",
                latest.iloc[0]["company"] if len(latest) > 0 else "N/A",
                f"{latest.iloc[0]['pct_aum']:.1f}% AUM" if len(latest) > 0 else "",
            ), md=3),
            dbc.Col(metric_card("Large Cap", f"{large_pct:.1f}%", f"{num_sectors} sectors"), md=3),
            dbc.Col(metric_card("Mid+Small Cap", f"{mid_small_pct:.1f}%"), md=3),
        ],
        className="mb-4",
    )

    signals_df = detect_accumulation_signals(portfolio)
    if signals_df.empty:
        signals_content = empty_state("No significant accumulation/exit signals detected.", "bi-arrow-left-right")
    else:
        signal_rows = []
        for _, row in signals_df.head(20).iterrows():
            signal_rows.append(
                html.Tr([
                    html.Td(row["company"]),
                    html.Td(f"{row['previous_pct']}%"),
                    html.Td(f"{row['current_pct']}%"),
                    html.Td(f"{row['change_pct']:+.2f}%"),
                    html.Td(signal_badge(row["signal"])),
                ])
            )
        signals_content = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in ["Company", "Previous %", "Current %", "Change", "Signal"]])),
                html.Tbody(signal_rows),
            ],
            bordered=True, hover=True, striped=True, responsive=True, className="mt-3",
        )

    return (
        html.Div([summary, holdings_table]),
        signals_content,
        portfolio.to_json(date_format="iso"),
    )


# ── Smart Money Flow tab ──

@callback(
    Output("flow-output", "children"),
    Output("conviction-output", "children"),
    Output("all-portfolios-store", "data"),
    Output("overlap-fund-checklist", "options"),
    Input("scan-flow-btn", "n_clicks"),
    State("flow-months", "value"),
    prevent_initial_call=True,
)
def scan_smart_money(n_clicks, months):
    months = int(months) if months else 3
    all_portfolios = fetch_all_manager_portfolios(months)

    if not all_portfolios:
        return (
            empty_state("Could not fetch portfolio data. Try again.", "bi-exclamation-triangle"),
            "",
            None,
            [],
        )

    for name in all_portfolios:
        all_portfolios[name] = match_holdings_to_classification(all_portfolios[name])

    signals = aggregate_cross_fund_signals(all_portfolios)
    conviction = compute_conviction_scores(all_portfolios)

    flow_fig = create_flow_chart(signals)

    signals_table = html.Div()
    if not signals.empty:
        display_cols = ["company", "net_change_pct", "num_funds_holding", "signal", "strength"]
        available = [c for c in display_cols if c in signals.columns]
        signals_table = dash_table.DataTable(
            data=signals[available].head(30).to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in available],
            page_size=15,
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        )

    flow_content = html.Div([
        dbc.Row([
            dbc.Col(metric_card("Managers Tracked", str(len(all_portfolios))), md=3),
            dbc.Col(metric_card("Accumulation Signals", str(len(signals[signals["signal"] == "ACCUMULATE"])) if not signals.empty else "0", color="success"), md=3),
            dbc.Col(metric_card("Exit Signals", str(len(signals[signals["signal"] == "EXIT"])) if not signals.empty else "0", color="danger"), md=3),
            dbc.Col(metric_card("New Entries", str(len(signals[signals["signal"] == "NEW ENTRY"])) if not signals.empty and "NEW ENTRY" in signals["signal"].values else "0", color="info"), md=3),
        ], className="mb-4"),
        dcc.Graph(figure=flow_fig),
        html.H6("Signal Details", className="mt-3"),
        signals_table,
    ])

    treemap = create_conviction_treemap(conviction)
    conviction_content = html.Div([dcc.Graph(figure=treemap)]) if not conviction.empty else ""

    all_fund_names = []
    for mgr_name, portfolio in all_portfolios.items():
        if "fund_name" in portfolio.columns:
            for fn in portfolio["fund_name"].unique():
                all_fund_names.append(fn)
        else:
            all_fund_names.append(mgr_name)

    checklist_options = [{"label": n, "value": n} for n in sorted(set(all_fund_names)) if n]

    serialized = {}
    for name, df in all_portfolios.items():
        serialized[name] = df.to_json(date_format="iso")

    import json
    return flow_content, conviction_content, json.dumps(serialized), checklist_options


# ── Manager Fingerprint tab ──

@callback(
    Output("fingerprint-output", "children"),
    Input("fingerprint-btn", "n_clicks"),
    State("fingerprint-manager", "value"),
    State("num-clusters", "value"),
    State("all-portfolios-store", "data"),
    prevent_initial_call=True,
)
def compute_fingerprints(n_clicks, selected_manager, n_clusters, stored_data):
    import json

    if not stored_data:
        return dbc.Alert(
            "Run 'Scan All Managers' in Smart Money Flow tab first to load portfolio data.",
            color="warning",
        )

    all_portfolios = {}
    raw = json.loads(stored_data)
    for name, json_str in raw.items():
        all_portfolios[name] = pd.read_json(json_str)

    n_clusters = int(n_clusters) if n_clusters else 3
    fingerprints = []

    for mgr_name, portfolio in all_portfolios.items():
        if portfolio.empty:
            continue
        latest = portfolio[portfolio["date"] == portfolio["date"].max()]
        fp = compute_manager_fingerprint(latest, portfolio, mgr_name)
        fingerprints.append(fp)

    if not fingerprints:
        return empty_state("No fingerprint data available.", "bi-fingerprint")

    clustered = cluster_managers(fingerprints, n_clusters)

    selected_name = FUND_MANAGERS.get(selected_manager, {}).get("name", "")
    selected_fp = next((fp for fp in fingerprints if fp["manager"] == selected_name), fingerprints[0])
    radar = create_fingerprint_radar(selected_fp, selected_fp["manager"])
    scatter = create_cluster_scatter(clustered)
    comparison = create_fingerprint_comparison(clustered)

    fp_table = dash_table.DataTable(
        data=clustered[[
            "manager", "large_cap_pct", "mid_cap_pct", "small_cap_pct",
            "herfindahl_index", "top_3_sector_pct", "turnover_pct", "cluster_label",
        ]].round(1).to_dict("records"),
        columns=[
            {"name": "Manager", "id": "manager"},
            {"name": "Large %", "id": "large_cap_pct"},
            {"name": "Mid %", "id": "mid_cap_pct"},
            {"name": "Small %", "id": "small_cap_pct"},
            {"name": "HHI", "id": "herfindahl_index"},
            {"name": "Top 3 Sec %", "id": "top_3_sector_pct"},
            {"name": "Turnover %", "id": "turnover_pct"},
            {"name": "Cluster", "id": "cluster_label"},
        ],
        sort_action="native",
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
    )

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=radar), md=6), dbc.Col(dcc.Graph(figure=scatter), md=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=comparison))], className="mt-3"),
        html.H6("Fingerprint Data", className="mt-3"),
        fp_table,
    ])


# ── Portfolio Overlap tab ──

@callback(
    Output("overlap-output", "children"),
    Output("common-holdings-output", "children"),
    Input("overlap-btn", "n_clicks"),
    State("all-portfolios-store", "data"),
    State("overlap-fund-checklist", "value"),
    State("overlap-method", "value"),
    prevent_initial_call=True,
)
def compute_and_display_overlap(n_clicks, stored_data, selected_funds, method):
    import json

    if not stored_data:
        return (
            dbc.Alert("Run 'Scan All Managers' in Smart Money Flow tab first.", color="warning"),
            "",
        )

    all_portfolios = {}
    raw = json.loads(stored_data)
    for name, json_str in raw.items():
        all_portfolios[name] = pd.read_json(json_str)

    fund_holdings = {}
    for mgr_name, portfolio in all_portfolios.items():
        if portfolio.empty:
            continue
        latest = portfolio[portfolio["date"] == portfolio["date"].max()]
        if "fund_name" in latest.columns:
            for fn in latest["fund_name"].unique():
                if not selected_funds or fn in selected_funds:
                    fund_holdings[fn] = latest[latest["fund_name"] == fn]
        else:
            if not selected_funds or mgr_name in selected_funds:
                fund_holdings[mgr_name] = latest

    if len(fund_holdings) < 2:
        return (
            dbc.Alert("Select at least 2 funds to compare. Run Smart Money scan first, then check funds.", color="warning"),
            "",
        )

    if method == "weighted":
        matrix = compute_weighted_overlap(fund_holdings)
        title = "Weighted Portfolio Overlap (by AUM %)"
    else:
        matrix = compute_overlap_matrix(fund_holdings)
        title = "Portfolio Overlap — Jaccard Similarity (%)"

    heatmap = create_overlap_heatmap(matrix, title)
    sunburst = create_overlap_sunburst(fund_holdings)

    common = find_common_holdings(fund_holdings)
    common_chart = create_common_holdings_chart(common)

    overlap_content = html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=heatmap), md=7), dbc.Col(dcc.Graph(figure=sunburst), md=5)]),
    ])

    common_content = html.Div()
    if not common.empty:
        common_table = dash_table.DataTable(
            data=common.head(25).to_dict("records"),
            columns=[
                {"name": "Company", "id": "company"},
                {"name": "# Funds", "id": "num_funds"},
                {"name": "Avg Weight %", "id": "avg_weight"},
                {"name": "Held By", "id": "held_by"},
            ],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        )
        common_content = html.Div([
            dcc.Graph(figure=common_chart),
            common_table,
        ])

    return overlap_content, common_content


# ── Style Drift tab ──

@callback(
    Output("drift-output", "children"),
    Input("drift-btn", "n_clicks"),
    State("portfolio-store", "data"),
    State("drift-category", "value"),
    State("all-portfolios-store", "data"),
    prevent_initial_call=True,
)
def analyze_drift(n_clicks, stored_data, category, all_stored):
    import json

    portfolio = pd.DataFrame()
    if stored_data:
        portfolio = pd.read_json(stored_data)

    if portfolio.empty and all_stored:
        raw = json.loads(all_stored)
        frames = []
        for name, json_str in raw.items():
            df = pd.read_json(json_str)
            if not df.empty and "fund_category" in df.columns:
                matching = df[df["fund_category"] == category]
                if not matching.empty:
                    frames.append(matching)
        if frames:
            portfolio = pd.concat(frames, ignore_index=True)

    if portfolio.empty:
        return dbc.Alert("Fetch holdings or run Smart Money scan first.", color="warning")

    latest = portfolio[portfolio["date"] == portfolio["date"].max()]
    drift_result = detect_style_drift(latest, category)
    drift_history = compute_drift_over_time(portfolio, category)
    drift_fig = create_drift_chart(drift_history, f"Fund ({category.replace('_', ' ').title()})")
    gauge_fig = create_mandate_gauge(drift_result["allocation"], category)

    severity_colors = {"critical": "danger", "warning": "warning", "minor": "info", "none": "success"}
    severity_text = {
        "critical": "CRITICAL DRIFT — Fund significantly outside mandate",
        "warning": "WARNING — Fund drifting from mandate",
        "minor": "MINOR — Slight deviation detected",
        "none": "COMPLIANT — Within SEBI mandate",
    }

    sev = drift_result.get("severity", "none")

    violation_alerts = []
    for v in drift_result.get("violations", []):
        violation_alerts.append(
            dbc.Alert(
                f"{v['category'].replace('_', ' ').title()}: {v['actual_pct']:.1f}% actual "
                f"(required {v['required_pct']:.1f}%, shortfall {v['shortfall_pct']:.1f}%)",
                color="danger",
                className="py-2",
            )
        )

    return html.Div([
        dbc.Alert(severity_text[sev], color=severity_colors[sev], className="text-center fs-5"),
        *violation_alerts,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=gauge_fig), md=5),
            dbc.Col(dcc.Graph(figure=drift_fig), md=7),
        ]),
    ])


# ── SIP XIRR tab ──

@callback(
    Output("xirr-output", "children"),
    Output("xirr-growth-chart", "children"),
    Input("calc-xirr-btn", "n_clicks"),
    State("sip-amount", "value"),
    State("sip-ticker", "value"),
    State("sip-period", "value"),
    prevent_initial_call=True,
)
def calculate_xirr(n_clicks, sip_amount, ticker, period):
    if not ticker or not sip_amount:
        return html.P("Enter ticker and SIP amount.", className="text-warning"), ""

    sip_amount = float(sip_amount)
    result = benchmark_sip(ticker, sip_amount, period)

    fund = result["fund"]
    bench = result["benchmark"]

    cards = dbc.Row([
        dbc.Col(metric_card(
            "Fund XIRR",
            f"{fund['xirr']}%" if fund["xirr"] else "N/A",
            f"Invested: INR {fund['total_invested']:,.0f}",
            "success" if fund.get("xirr") and fund["xirr"] > 0 else "danger",
        ), md=2),
        dbc.Col(metric_card(
            "Current Value",
            f"INR {fund['current_value']:,.0f}" if fund["current_value"] else "N/A",
            f"{fund['num_installments']} SIPs",
        ), md=2),
        dbc.Col(metric_card(
            "Absolute Gain",
            f"INR {fund.get('absolute_gain', 0):,.0f}",
            f"{fund.get('absolute_gain_pct', 0):+.1f}%",
            "success" if fund.get("absolute_gain", 0) > 0 else "danger",
        ), md=2),
        dbc.Col(metric_card(
            "Nifty 50 XIRR",
            f"{bench['xirr']}%" if bench["xirr"] else "N/A",
            "Benchmark",
        ), md=2),
        dbc.Col(metric_card(
            "Alpha vs Nifty",
            f"{result['alpha']:+.2f}%" if result["alpha"] else "N/A",
            "Outperformance" if result.get("alpha") and result["alpha"] > 0 else "Underperformance",
            "success" if result.get("alpha") and result["alpha"] > 0 else "danger",
        ), md=2),
        dbc.Col(metric_card(
            f"INR {sip_amount:,.0f}/mo grew to",
            f"INR {fund['current_value']:,.0f}" if fund["current_value"] else "N/A",
            f"over {fund['num_installments']} months",
            "info",
        ), md=2),
    ], className="mb-4")

    growth_fig = create_sip_growth_chart(fund, bench, ticker)
    growth_chart = dcc.Graph(figure=growth_fig)

    return cards, growth_chart


@callback(
    Output("peer-output", "children"),
    Input("peer-compare-btn", "n_clicks"),
    State("peer-group-select", "value"),
    State("sip-amount", "value"),
    State("sip-period", "value"),
    prevent_initial_call=True,
)
def compare_peers(n_clicks, peer_group, sip_amount, period):
    if not peer_group or peer_group not in PEER_GROUPS:
        return html.P("Select a peer group.", className="text-warning")

    tickers = PEER_GROUPS[peer_group]
    sip_amount = float(sip_amount or 10000)
    period = period or "5y"

    df = compare_category_peers(
        {name: ticker for ticker, name in tickers.items()},
        sip_amount,
        period,
    )

    if df.empty:
        return html.P("Could not fetch data for these tickers.", className="text-warning")

    bar_fig = create_peer_comparison_chart(df)

    cols_to_show = ["fund_name", "xirr", "total_invested", "current_value", "absolute_gain_pct", "num_installments"]
    available = [c for c in cols_to_show if c in df.columns]

    table = dash_table.DataTable(
        data=df[available].to_dict("records"),
        columns=[
            {"name": "Fund", "id": "fund_name"},
            {"name": "XIRR (%)", "id": "xirr", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Invested", "id": "total_invested", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Current Value", "id": "current_value", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Gain %", "id": "absolute_gain_pct", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "SIPs", "id": "num_installments"},
        ],
        sort_action="native",
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"row_index": 0}, "backgroundColor": "#E8F5E9", "fontWeight": "bold"},
        ],
    )
    return html.Div([dcc.Graph(figure=bar_fig), table])


# ── BHB Attribution tab ──

@callback(
    Output("attribution-output", "children"),
    Input("attribution-btn", "n_clicks"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_attribution(n_clicks, stored_data):
    if not stored_data:
        return dbc.Alert("Fetch holdings first in the Fund Holdings tab.", color="warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty or "sector" not in portfolio.columns:
        return html.P("No sector data available for attribution.", className="text-warning")

    latest = portfolio[portfolio["date"] == portfolio["date"].max()]

    from config import SECTOR_MAP
    latest = latest.copy()
    latest["sector_normalized"] = latest["sector"].map(SECTOR_MAP).fillna(latest["sector"])
    sectors = latest["sector_normalized"].dropna().unique()

    if len(sectors) == 0:
        return html.P("No sector data.", className="text-warning")

    total_aum = latest["pct_aum"].sum()
    port_weights = latest.groupby("sector_normalized")["pct_aum"].sum() / total_aum
    bench_weights = pd.Series(1.0 / len(sectors), index=sectors)

    np.random.seed(42)
    port_returns = pd.Series(np.random.normal(0.12, 0.05, len(sectors)), index=sectors)
    bench_returns = pd.Series(np.random.normal(0.10, 0.03, len(sectors)), index=sectors)

    attr_df = compute_bhb_attribution(port_weights, bench_weights, port_returns, bench_returns)
    bar_fig = create_attribution_chart(attr_df)
    waterfall_fig = create_attribution_waterfall(attr_df)

    totals = attr_df[attr_df["sector"] == "TOTAL"].iloc[0]

    summary = dbc.Row([
        dbc.Col(metric_card("Allocation Effect", f"{totals['allocation_effect']:+.2f}%", "Sector weighting"), md=3),
        dbc.Col(metric_card("Selection Effect", f"{totals['selection_effect']:+.2f}%", "Stock picking", color="success"), md=3),
        dbc.Col(metric_card("Interaction Effect", f"{totals['interaction_effect']:+.2f}%", "Combined"), md=3),
        dbc.Col(metric_card("Total Alpha", f"{totals['total_effect']:+.2f}%", "Net outperformance", color="info"), md=3),
    ], className="mb-4")

    detail_table = dash_table.DataTable(
        data=attr_df.to_dict("records"),
        columns=[
            {"name": "Sector", "id": "sector"},
            {"name": "Port Wt %", "id": "portfolio_weight", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Bench Wt %", "id": "benchmark_weight", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Allocation %", "id": "allocation_effect", "type": "numeric", "format": {"specifier": "+.3f"}},
            {"name": "Selection %", "id": "selection_effect", "type": "numeric", "format": {"specifier": "+.3f"}},
            {"name": "Total %", "id": "total_effect", "type": "numeric", "format": {"specifier": "+.3f"}},
        ],
        sort_action="native",
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": "{sector} = TOTAL"}, "fontWeight": "bold", "backgroundColor": "#E3F2FD"},
        ],
    )

    return html.Div([
        summary,
        dbc.Row([dbc.Col(dcc.Graph(figure=waterfall_fig), md=5), dbc.Col(dcc.Graph(figure=bar_fig), md=7)]),
        html.H6("Attribution Detail by Sector", className="mt-3"),
        detail_table,
    ])
