"""Dash callbacks — all 9 features wired to UI."""

import json

import numpy as np
import pandas as pd
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
    create_pca_scatter,
)
from src.analysis.momentum import (
    build_signal_features,
    train_signal_predictor,
    predict_signals,
    create_feature_importance_chart,
)
from src.analysis.overlap import (
    compute_overlap_matrix,
    compute_weighted_overlap,
    create_overlap_heatmap,
    find_common_holdings,
    create_common_holdings_chart,
    create_overlap_sunburst,
    create_venn_diagram,
)
from src.analysis.backtest import (
    backtest_smart_money_signals,
    create_backtest_chart,
)
from src.analysis.efficient_frontier import (
    fetch_stock_prices,
    compute_efficient_frontier,
    optimize_portfolio,
    create_efficient_frontier_chart,
    create_allocation_pie,
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
    fetch_nav_series,
)
from src.analysis.bhb_attribution import (
    compute_bhb_attribution,
    create_attribution_chart,
    create_attribution_waterfall,
)
from src.analysis.risk_metrics import (
    compute_risk_metrics,
    create_risk_dashboard,
    create_risk_return_scatter,
)
from src.analysis.sector_breakdown import (
    create_sector_pie,
    create_sector_treemap,
    compute_sector_evolution,
    create_sector_evolution_chart,
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
        return empty_state("No data found. Try a different fund or time range.", "bi-database-x"), "", None

    manager = FUND_MANAGERS.get(manager_key, {})
    fund_info = next((f for f in manager.get("funds", []) if f["scheme_code"] == scheme_code), {})
    portfolio["fund_name"] = fund_info.get("name", "")
    portfolio["manager"] = manager.get("name", "")
    portfolio["fund_category"] = fund_info.get("category", "")

    portfolio = match_holdings_to_classification(portfolio)

    latest_date = portfolio["date"].max()
    latest = portfolio[portfolio["date"] == latest_date].copy().sort_values("pct_aum", ascending=False)

    holdings_table = dash_table.DataTable(
        data=latest[["company", "sector", "quantity", "market_value_lakhs", "pct_aum", "market_cap_category"]].to_dict("records"),
        columns=[
            {"name": "Company", "id": "company"},
            {"name": "Sector", "id": "sector"},
            {"name": "Qty", "id": "quantity", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Value (L)", "id": "market_value_lakhs", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "% AUM", "id": "pct_aum", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Cap", "id": "market_cap_category"},
        ],
        page_size=25, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold"},
    )

    large_pct = latest[latest["market_cap_category"] == "large_cap"]["pct_aum"].sum()
    mid_small_pct = latest[latest["market_cap_category"].isin(["mid_cap", "small_cap"])]["pct_aum"].sum()

    summary = dbc.Row([
        dbc.Col(metric_card("Total Stocks", str(len(latest)), f"as of {latest_date.strftime('%b %Y')}"), md=3),
        dbc.Col(metric_card("Top Holding", latest.iloc[0]["company"] if len(latest) > 0 else "N/A", f"{latest.iloc[0]['pct_aum']:.1f}% AUM" if len(latest) > 0 else ""), md=3),
        dbc.Col(metric_card("Large Cap", f"{large_pct:.1f}%", f"{latest['sector'].nunique()} sectors"), md=3),
        dbc.Col(metric_card("Mid+Small", f"{mid_small_pct:.1f}%"), md=3),
    ], className="mb-4")

    signals_df = detect_accumulation_signals(portfolio)
    if signals_df.empty:
        signals_content = empty_state("No significant signals detected.", "bi-arrow-left-right")
    else:
        signal_rows = [
            html.Tr([
                html.Td(row["company"]),
                html.Td(f"{row['previous_pct']}%"),
                html.Td(f"{row['current_pct']}%"),
                html.Td(f"{row['change_pct']:+.2f}%"),
                html.Td(signal_badge(row["signal"])),
            ])
            for _, row in signals_df.head(20).iterrows()
        ]
        signals_content = dbc.Table(
            [html.Thead(html.Tr([html.Th(c) for c in ["Company", "Prev %", "Curr %", "Change", "Signal"]])), html.Tbody(signal_rows)],
            bordered=True, hover=True, striped=True, responsive=True, className="mt-3",
        )

    return html.Div([summary, holdings_table]), signals_content, portfolio.to_json(date_format="iso")


# ── Smart Money Flow ──

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
        return empty_state("Could not fetch data. Try again.", "bi-exclamation-triangle"), "", None, []

    for name in all_portfolios:
        all_portfolios[name] = match_holdings_to_classification(all_portfolios[name])

    signals = aggregate_cross_fund_signals(all_portfolios)
    conviction = compute_conviction_scores(all_portfolios)

    flow_fig = create_flow_chart(signals)
    sig_count = lambda s: str(len(signals[signals["signal"] == s])) if not signals.empty and s in signals["signal"].values else "0"

    flow_content = html.Div([
        dbc.Row([
            dbc.Col(metric_card("Managers", str(len(all_portfolios))), md=3),
            dbc.Col(metric_card("Accumulate", sig_count("ACCUMULATE"), color="success"), md=3),
            dbc.Col(metric_card("Exit", sig_count("EXIT"), color="danger"), md=3),
            dbc.Col(metric_card("New Entry", sig_count("NEW ENTRY"), color="info"), md=3),
        ], className="mb-4"),
        dcc.Graph(figure=flow_fig),
        _signals_table(signals),
    ])

    treemap = create_conviction_treemap(conviction)

    ml_section = html.Div()
    all_history = pd.concat([p for p in all_portfolios.values() if not p.empty], ignore_index=True)
    if not all_history.empty and len(all_history["date"].unique()) >= 3:
        features = build_signal_features(all_history)
        if not features.empty and len(features) >= 20:
            model_result = train_signal_predictor(features)
            if model_result.get("model"):
                latest_features = features[features["date"] == features["date"].max()]
                predictions = predict_signals(model_result, latest_features)
                importance_fig = create_feature_importance_chart(model_result["feature_importance"])

                pred_table = html.Div()
                if not predictions.empty:
                    pred_table = dash_table.DataTable(
                        data=predictions.head(20).to_dict("records"),
                        columns=[
                            {"name": "Company", "id": "company"},
                            {"name": "ML Confidence %", "id": "predicted_continue"},
                            {"name": "Prediction", "id": "prediction"},
                        ],
                        sort_action="native", page_size=10,
                        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                        style_header={"fontWeight": "bold"},
                    )

                ml_section = html.Div([
                    html.Hr(),
                    html.H5("ML Signal Predictor (GBM)", className="mt-3"),
                    dbc.Row([
                        dbc.Col(metric_card("Model Accuracy", f"{model_result['accuracy']}%", f"CV std: {model_result['cv_std']}%"), md=3),
                        dbc.Col(metric_card("Training Samples", str(model_result["n_samples"])), md=3),
                        dbc.Col(metric_card("Positive Rate", f"{model_result['positive_rate']}%", "Stocks that continued increasing"), md=3),
                        dbc.Col(metric_card("Features", str(len(model_result["features_used"]))), md=3),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=importance_fig), md=5),
                        dbc.Col(pred_table, md=7),
                    ]),
                ])

    conviction_content = html.Div([dcc.Graph(figure=treemap), ml_section]) if not conviction.empty else ml_section

    all_fund_names = []
    for mgr_name, portfolio in all_portfolios.items():
        if "fund_name" in portfolio.columns:
            all_fund_names.extend(portfolio["fund_name"].unique())
        else:
            all_fund_names.append(mgr_name)

    checklist = [{"label": n, "value": n} for n in sorted(set(all_fund_names)) if n]

    serialized = {name: df.to_json(date_format="iso") for name, df in all_portfolios.items()}
    return flow_content, conviction_content, json.dumps(serialized), checklist


def _signals_table(signals):
    if signals.empty:
        return html.Div()
    display_cols = [c for c in ["company", "net_change_pct", "num_funds_holding", "signal", "strength"] if c in signals.columns]
    return dash_table.DataTable(
        data=signals[display_cols].head(30).to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
        page_size=15, sort_action="native",
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
        style_header={"fontWeight": "bold"},
    )


# ── Manager Fingerprint ──

@callback(
    Output("fingerprint-output", "children"),
    Input("fingerprint-btn", "n_clicks"),
    State("fingerprint-manager", "value"),
    State("num-clusters", "value"),
    State("all-portfolios-store", "data"),
    prevent_initial_call=True,
)
def compute_fingerprints(n_clicks, selected_manager, n_clusters, stored_data):
    if not stored_data:
        return dbc.Alert("Run Smart Money scan first to load data.", color="warning")

    all_portfolios = {name: pd.read_json(j) for name, j in json.loads(stored_data).items()}
    n_clusters = int(n_clusters or 3)

    fingerprints = []
    for mgr_name, portfolio in all_portfolios.items():
        if portfolio.empty:
            continue
        latest = portfolio[portfolio["date"] == portfolio["date"].max()]
        fingerprints.append(compute_manager_fingerprint(latest, portfolio, mgr_name))

    if not fingerprints:
        return empty_state("No fingerprint data.", "bi-fingerprint")

    clustered = cluster_managers(fingerprints, n_clusters)

    selected_name = FUND_MANAGERS.get(selected_manager, {}).get("name", "")
    selected_fp = next((fp for fp in fingerprints if fp["manager"] == selected_name), fingerprints[0])

    pca_chart = create_pca_scatter(clustered)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=create_fingerprint_radar(selected_fp, selected_fp["manager"])), md=6),
            dbc.Col(dcc.Graph(figure=create_cluster_scatter(clustered)), md=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=pca_chart), md=6) if not pca_chart.data == () else html.Div(),
            dbc.Col(dcc.Graph(figure=create_fingerprint_comparison(clustered)), md=6),
        ], className="mt-3"),
        html.H6("Data", className="mt-3"),
        dash_table.DataTable(
            data=clustered[["manager", "large_cap_pct", "mid_cap_pct", "small_cap_pct", "herfindahl_index", "top_3_sector_pct", "turnover_pct", "cluster_label"]].round(1).to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in ["manager", "large_cap_pct", "mid_cap_pct", "small_cap_pct", "herfindahl_index", "top_3_sector_pct", "turnover_pct", "cluster_label"]],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold"},
        ),
    ])


# ── Portfolio Overlap ──

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
    if not stored_data:
        return dbc.Alert("Run Smart Money scan first.", color="warning"), ""

    all_portfolios = {name: pd.read_json(j) for name, j in json.loads(stored_data).items()}

    fund_holdings = {}
    for mgr_name, portfolio in all_portfolios.items():
        if portfolio.empty:
            continue
        latest = portfolio[portfolio["date"] == portfolio["date"].max()]
        if "fund_name" in latest.columns:
            for fn in latest["fund_name"].unique():
                if not selected_funds or fn in selected_funds:
                    fund_holdings[fn] = latest[latest["fund_name"] == fn]
        elif not selected_funds or mgr_name in selected_funds:
            fund_holdings[mgr_name] = latest

    if len(fund_holdings) < 2:
        return dbc.Alert("Select 2+ funds. Run Smart Money scan first, then check funds.", color="warning"), ""

    matrix_fn = compute_weighted_overlap if method == "weighted" else compute_overlap_matrix
    title = "Weighted Overlap (AUM %)" if method == "weighted" else "Jaccard Similarity (%)"
    matrix = matrix_fn(fund_holdings)

    common = find_common_holdings(fund_holdings)

    venn_fig = create_venn_diagram(fund_holdings) if 2 <= len(fund_holdings) <= 3 else None

    overlap_content = html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=create_overlap_heatmap(matrix, title)), md=7),
            dbc.Col(dcc.Graph(figure=venn_fig) if venn_fig else dcc.Graph(figure=create_overlap_sunburst(fund_holdings)), md=5),
        ]),
        dbc.Row([dbc.Col(dcc.Graph(figure=create_overlap_sunburst(fund_holdings)))], className="mt-2") if venn_fig else html.Div(),
    ])

    common_content = html.Div()
    if not common.empty:
        common_content = html.Div([
            dcc.Graph(figure=create_common_holdings_chart(common)),
            dash_table.DataTable(
                data=common.head(25).to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in ["company", "num_funds", "avg_weight", "held_by"]],
                sort_action="native",
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                style_header={"fontWeight": "bold"},
            ),
        ])

    return overlap_content, common_content


# ── Style Drift ──

@callback(
    Output("drift-output", "children"),
    Input("drift-btn", "n_clicks"),
    State("portfolio-store", "data"),
    State("drift-category", "value"),
    State("all-portfolios-store", "data"),
    prevent_initial_call=True,
)
def analyze_drift(n_clicks, stored_data, category, all_stored):
    portfolio = pd.DataFrame()
    if stored_data:
        portfolio = pd.read_json(stored_data)

    if portfolio.empty and all_stored:
        frames = []
        for name, j in json.loads(all_stored).items():
            df = pd.read_json(j)
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

    sev = drift_result.get("severity", "none")
    severity_map = {
        "critical": ("danger", "CRITICAL DRIFT"),
        "warning": ("warning", "WARNING — Drifting"),
        "minor": ("info", "MINOR deviation"),
        "none": ("success", "COMPLIANT"),
    }
    color, text = severity_map.get(sev, ("success", "OK"))

    violations = [
        dbc.Alert(
            f"{v['category'].replace('_', ' ').title()}: {v['actual_pct']:.1f}% (need {v['required_pct']:.1f}%, short {v['shortfall_pct']:.1f}%)",
            color="danger", className="py-2",
        )
        for v in drift_result.get("violations", [])
    ]

    return html.Div([
        dbc.Alert(text, color=color, className="text-center fs-5"),
        *violations,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=create_mandate_gauge(drift_result["allocation"], category)), md=5),
            dbc.Col(dcc.Graph(figure=create_drift_chart(drift_history, f"Fund ({category.replace('_', ' ').title()})")), md=7),
        ]),
    ])


# ── SIP XIRR ──

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

    result = benchmark_sip(ticker, float(sip_amount), period)
    fund, bench = result["fund"], result["benchmark"]

    cards = dbc.Row([
        dbc.Col(metric_card("Fund XIRR", f"{fund['xirr']}%" if fund["xirr"] else "N/A", f"Invested: INR {fund['total_invested']:,.0f}", "success" if fund.get("xirr") and fund["xirr"] > 0 else "danger"), md=2),
        dbc.Col(metric_card("Current Value", f"INR {fund['current_value']:,.0f}" if fund["current_value"] else "N/A", f"{fund['num_installments']} SIPs"), md=2),
        dbc.Col(metric_card("Absolute Gain", f"INR {fund.get('absolute_gain', 0):,.0f}", f"{fund.get('absolute_gain_pct', 0):+.1f}%", "success" if fund.get("absolute_gain", 0) > 0 else "danger"), md=2),
        dbc.Col(metric_card("Nifty 50 XIRR", f"{bench['xirr']}%" if bench["xirr"] else "N/A", "Benchmark"), md=2),
        dbc.Col(metric_card("Alpha", f"{result['alpha']:+.2f}%" if result["alpha"] else "N/A", "vs Nifty", "success" if result.get("alpha") and result["alpha"] > 0 else "danger"), md=2),
        dbc.Col(metric_card(f"INR {float(sip_amount):,.0f}/mo", f"INR {fund['current_value']:,.0f}" if fund["current_value"] else "N/A", f"{fund['num_installments']} months", "info"), md=2),
    ], className="mb-4")

    growth = dcc.Graph(figure=create_sip_growth_chart(fund, bench, ticker))
    return cards, growth


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
    df = compare_category_peers({name: ticker for ticker, name in tickers.items()}, float(sip_amount or 10000), period or "5y")

    if df.empty:
        return html.P("Could not fetch data.", className="text-warning")

    available = [c for c in ["fund_name", "xirr", "total_invested", "current_value", "absolute_gain_pct", "num_installments"] if c in df.columns]

    return html.Div([
        dcc.Graph(figure=create_peer_comparison_chart(df)),
        dash_table.DataTable(
            data=df[available].to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in available],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[{"if": {"row_index": 0}, "fontWeight": "bold"}],
        ),
    ])


# ── Risk Analytics ──

@callback(
    Output("risk-output", "children"),
    Input("risk-btn", "n_clicks"),
    Input("risk-compare-btn", "n_clicks"),
    State("risk-ticker", "value"),
    State("risk-period", "value"),
    prevent_initial_call=True,
)
def analyze_risk(single_click, compare_click, ticker, period):
    from dash import ctx
    triggered = ctx.triggered_id

    if triggered == "risk-compare-btn":
        return _risk_comparison(period)

    if not ticker:
        return html.P("Select a fund.", className="text-warning")

    nav = fetch_nav_series(ticker, period or "5y")
    if nav.empty:
        return html.P("Could not fetch NAV data.", className="text-warning")

    fund_name = ticker
    for mgr in FUND_MANAGERS.values():
        for f in mgr["funds"]:
            if f.get("ticker") == ticker:
                fund_name = f["name"]
                break

    metrics = compute_risk_metrics(nav)
    dashboard = create_risk_dashboard(metrics, nav, fund_name)

    m = dashboard["metrics"]

    return html.Div([
        dbc.Row([
            dbc.Col(metric_card("CAGR", f"{m['cagr_pct']}%" if m["cagr_pct"] else "N/A", f"{m.get('years', '?')}Y", "success" if m.get("cagr_pct") and m["cagr_pct"] > 0 else "danger"), md=2),
            dbc.Col(metric_card("Volatility", f"{m['volatility_pct']}%" if m["volatility_pct"] else "N/A", "Annualized"), md=2),
            dbc.Col(metric_card("Sharpe", f"{m['sharpe_ratio']}" if m["sharpe_ratio"] else "N/A", ">1 is good", "success" if m.get("sharpe_ratio") and m["sharpe_ratio"] > 1 else "warning"), md=2),
            dbc.Col(metric_card("Sortino", f"{m['sortino_ratio']}" if m["sortino_ratio"] else "N/A", ">2 is good", "success" if m.get("sortino_ratio") and m["sortino_ratio"] > 2 else "warning"), md=2),
            dbc.Col(metric_card("Max Drawdown", f"{m['max_drawdown_pct']}%" if m["max_drawdown_pct"] else "N/A", "Peak to trough", "danger"), md=2),
            dbc.Col(metric_card("VaR 95%", f"{m['var_95_pct']}%" if m["var_95_pct"] else "N/A", "Daily worst case", "danger"), md=2),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(metric_card("Calmar", f"{m['calmar_ratio']}" if m["calmar_ratio"] else "N/A"), md=3),
            dbc.Col(metric_card("Best Day", f"{m['best_day_pct']}%" if m["best_day_pct"] else "N/A", color="success"), md=3),
            dbc.Col(metric_card("Worst Day", f"{m['worst_day_pct']}%" if m["worst_day_pct"] else "N/A", color="danger"), md=3),
            dbc.Col(metric_card("Win Rate", f"{m['positive_days_pct']}%" if m["positive_days_pct"] else "N/A", "% positive days"), md=3),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=dashboard["rolling_chart"]))]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=dashboard["drawdown_chart"]), md=6),
            dbc.Col(dcc.Graph(figure=dashboard["monthly_heatmap"]), md=6),
        ]),
    ])


def _risk_comparison(period):
    all_metrics = []
    for mgr in FUND_MANAGERS.values():
        for f in mgr["funds"]:
            if f.get("ticker"):
                nav = fetch_nav_series(f["ticker"], period or "5y")
                if not nav.empty:
                    m = compute_risk_metrics(nav)
                    m["name"] = f["name"]
                    all_metrics.append(m)

    if not all_metrics:
        return html.P("Could not fetch data for comparison.", className="text-warning")

    scatter = create_risk_return_scatter(all_metrics)
    df = pd.DataFrame(all_metrics)
    available = [c for c in ["name", "cagr_pct", "volatility_pct", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio"] if c in df.columns]

    return html.Div([
        dcc.Graph(figure=scatter),
        html.H6("Risk Metrics Comparison", className="mt-3"),
        dash_table.DataTable(
            data=df[available].sort_values("sharpe_ratio", ascending=False).round(2).to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in available],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[{"if": {"row_index": 0}, "fontWeight": "bold"}],
        ),
    ])


# ── Sector Breakdown ──

@callback(
    Output("sector-output", "children"),
    Input("main-tabs", "active_tab"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def show_sector_breakdown(active_tab, stored_data):
    if active_tab != "tab-sector":
        return no_update

    if not stored_data:
        return empty_state("Fetch holdings in the Fund Holdings tab first.", "bi-pie-chart")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty:
        return empty_state("No data available.", "bi-pie-chart")

    latest = portfolio[portfolio["date"] == portfolio["date"].max()]
    fund_name = latest["fund_name"].iloc[0] if "fund_name" in latest.columns and not latest.empty else "Fund"

    pie = create_sector_pie(latest, fund_name)
    treemap = create_sector_treemap(latest, fund_name)
    evolution = compute_sector_evolution(portfolio)
    evolution_chart = create_sector_evolution_chart(evolution, fund_name)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=pie), md=5),
            dbc.Col(dcc.Graph(figure=treemap), md=7),
        ]),
        dbc.Row([dbc.Col(dcc.Graph(figure=evolution_chart))], className="mt-3"),
    ])


# ── BHB Attribution ──

@callback(
    Output("attribution-output", "children"),
    Input("attribution-btn", "n_clicks"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_attribution(n_clicks, stored_data):
    if not stored_data:
        return dbc.Alert("Fetch holdings first.", color="warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty or "sector" not in portfolio.columns:
        return html.P("No sector data.", className="text-warning")

    latest = portfolio[portfolio["date"] == portfolio["date"].max()].copy()

    from config import SECTOR_MAP
    latest["sector_normalized"] = latest["sector"].map(SECTOR_MAP).fillna(latest["sector"])
    sectors = latest["sector_normalized"].dropna().unique()

    if len(sectors) == 0:
        return html.P("No sector data.", className="text-warning")

    total_aum = latest["pct_aum"].sum()
    port_weights = latest.groupby("sector_normalized")["pct_aum"].sum() / total_aum
    bench_weights = pd.Series(1.0 / len(sectors), index=sectors)

    from src.analysis.bhb_attribution import fetch_sector_returns
    real_returns = fetch_sector_returns("1y")

    port_returns = pd.Series(index=sectors, dtype=float)
    bench_returns = pd.Series(index=sectors, dtype=float)
    for s in sectors:
        port_returns[s] = real_returns.get(s, 0.10) * (1 + (port_weights.get(s, 0) - bench_weights.get(s, 0)) * 0.5)
        bench_returns[s] = real_returns.get(s, 0.10)

    attr_df = compute_bhb_attribution(port_weights, bench_weights, port_returns, bench_returns)
    totals = attr_df[attr_df["sector"] == "TOTAL"].iloc[0]

    return html.Div([
        dbc.Row([
            dbc.Col(metric_card("Allocation", f"{totals['allocation_effect']:+.2f}%", "Sector weight"), md=3),
            dbc.Col(metric_card("Selection", f"{totals['selection_effect']:+.2f}%", "Stock picks", color="success"), md=3),
            dbc.Col(metric_card("Interaction", f"{totals['interaction_effect']:+.2f}%", "Combined"), md=3),
            dbc.Col(metric_card("Total Alpha", f"{totals['total_effect']:+.2f}%", "Net", color="info"), md=3),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=create_attribution_waterfall(attr_df)), md=5),
            dbc.Col(dcc.Graph(figure=create_attribution_chart(attr_df)), md=7),
        ]),
        html.H6("Detail", className="mt-3"),
        dash_table.DataTable(
            data=attr_df.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in ["sector", "portfolio_weight", "benchmark_weight", "allocation_effect", "selection_effect", "total_effect"]],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[{"if": {"filter_query": "{sector} = TOTAL"}, "fontWeight": "bold"}],
        ),
    ])


# ── CSV Export ──

@callback(
    Output("csv-download", "data"),
    Input("global-export-btn", "n_clicks"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, stored_data):
    if not stored_data:
        return no_update

    portfolio = pd.read_json(stored_data)
    if portfolio.empty:
        return no_update

    latest = portfolio[portfolio["date"] == portfolio["date"].max()]
    export_cols = [c for c in ["company", "sector", "quantity", "market_value_lakhs", "pct_aum", "market_cap_category"] if c in latest.columns]
    return dcc.send_data_frame(latest[export_cols].to_csv, "smart_money_holdings.csv", index=False)


# ── Fund Screener ──

@callback(
    Output("screener-output", "children"),
    Input("screen-btn", "n_clicks"),
    State("screen-min-sharpe", "value"),
    State("screen-max-dd", "value"),
    State("screen-min-cagr", "value"),
    State("screen-max-vol", "value"),
    State("screen-period", "value"),
    prevent_initial_call=True,
)
def screen_funds(n_clicks, min_sharpe, max_dd, min_cagr, max_vol, period):
    min_sharpe = float(min_sharpe or 0)
    max_dd = float(max_dd or -100)
    min_cagr = float(min_cagr or 0)
    max_vol = float(max_vol or 100)
    period = period or "5y"

    all_metrics = []
    for mgr in FUND_MANAGERS.values():
        for f in mgr["funds"]:
            if f.get("ticker"):
                nav = fetch_nav_series(f["ticker"], period)
                if not nav.empty and len(nav) > 20:
                    m = compute_risk_metrics(nav)
                    m["fund_name"] = f["name"]
                    m["manager"] = mgr["name"]
                    m["category"] = f.get("category", "")
                    all_metrics.append(m)

    if not all_metrics:
        return html.P("Could not fetch data.", className="text-warning")

    df = pd.DataFrame(all_metrics)
    original_count = len(df)

    filtered = df[
        (df["sharpe_ratio"].fillna(0) >= min_sharpe) &
        (df["max_drawdown_pct"].fillna(-100) >= max_dd) &
        (df["cagr_pct"].fillna(0) >= min_cagr) &
        (df["volatility_pct"].fillna(100) <= max_vol)
    ].sort_values("sharpe_ratio", ascending=False)

    summary = dbc.Row([
        dbc.Col(metric_card("Scanned", str(original_count), "funds"), md=3),
        dbc.Col(metric_card("Passed Filter", str(len(filtered)), f"of {original_count}", "success" if len(filtered) > 0 else "danger"), md=3),
        dbc.Col(metric_card("Best Sharpe", f"{filtered['sharpe_ratio'].max():.2f}" if not filtered.empty else "N/A", filtered.iloc[0]["fund_name"] if not filtered.empty else ""), md=3),
        dbc.Col(metric_card("Lowest DD", f"{filtered['max_drawdown_pct'].max():.1f}%" if not filtered.empty else "N/A"), md=3),
    ], className="mb-4")

    if filtered.empty:
        return html.Div([summary, dbc.Alert("No funds match your criteria. Try relaxing the filters.", color="warning")])

    show_cols = ["fund_name", "manager", "category", "cagr_pct", "volatility_pct", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio"]
    available = [c for c in show_cols if c in filtered.columns]

    scatter = create_risk_return_scatter([row.to_dict() | {"name": row["fund_name"]} for _, row in filtered.iterrows()])

    table = dash_table.DataTable(
        data=filtered[available].round(2).to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in available],
        sort_action="native", page_size=20,
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[{"if": {"row_index": 0}, "fontWeight": "bold"}],
    )

    return html.Div([summary, dcc.Graph(figure=scatter), table])


# ── Backtest ──

@callback(
    Output("backtest-output", "children"),
    Input("bt-btn", "n_clicks"),
    State("bt-capital", "value"),
    State("bt-top-n", "value"),
    State("all-portfolios-store", "data"),
    prevent_initial_call=True,
)
def run_backtest(n_clicks, capital, top_n, stored_data):
    if not stored_data:
        return dbc.Alert("Run Smart Money scan first to load data.", color="warning")

    capital = float(capital or 100000)
    top_n = int(top_n or 10)

    all_portfolios = {name: pd.read_json(j) for name, j in json.loads(stored_data).items()}
    combined = pd.concat([p for p in all_portfolios.values() if not p.empty], ignore_index=True)

    if combined.empty:
        return html.P("No data.", className="text-warning")

    result = backtest_smart_money_signals(combined, capital, top_n)
    eq_df = result.get("equity_curve", pd.DataFrame())
    metrics = result.get("metrics", {})
    trades = result.get("trades", pd.DataFrame())

    if eq_df.empty:
        return html.P("Not enough data for backtest (need 3+ months).", className="text-warning")

    chart = create_backtest_chart(eq_df, capital)

    m = metrics
    cards = dbc.Row([
        dbc.Col(metric_card("Strategy CAGR", f"{m.get('strategy_cagr_pct', 0):+.2f}%", f"{m.get('years', 0):.1f}Y",
                            "success" if m.get("strategy_cagr_pct", 0) > 0 else "danger"), md=2),
        dbc.Col(metric_card("Nifty CAGR", f"{m.get('benchmark_cagr_pct', 0):+.2f}%", "Buy & Hold"), md=2),
        dbc.Col(metric_card("Alpha", f"{m.get('alpha_pct', 0):+.2f}%", "vs Nifty",
                            "success" if m.get("alpha_pct", 0) > 0 else "danger"), md=2),
        dbc.Col(metric_card("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}"), md=2),
        dbc.Col(metric_card("Max DD", f"{m.get('max_drawdown_pct', 0):.1f}%", color="danger"), md=2),
        dbc.Col(metric_card("Rebalances", str(m.get("num_rebalances", 0))), md=2),
    ], className="mb-4")

    trades_section = html.Div()
    if not trades.empty:
        trades_section = html.Div([
            html.H6("Trade Log", className="mt-3"),
            dash_table.DataTable(
                data=trades.to_dict("records"),
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in trades.columns],
                page_size=10, sort_action="native",
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                style_header={"fontWeight": "bold"},
            ),
        ])

    return html.Div([cards, dcc.Graph(figure=chart), trades_section])


# ── Efficient Frontier ──

@callback(
    Output("frontier-output", "children"),
    Input("frontier-btn", "n_clicks"),
    State("portfolio-store", "data"),
    State("frontier-n", "value"),
    prevent_initial_call=True,
)
def compute_frontier(n_clicks, stored_data, max_n):
    if not stored_data:
        return dbc.Alert("Fetch holdings first in the Fund Holdings tab.", color="warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty:
        return html.P("No data.", className="text-warning")

    max_n = int(max_n or 10)
    latest = portfolio[portfolio["date"] == portfolio["date"].max()].sort_values("pct_aum", ascending=False)

    symbols = latest["Symbol"].dropna().unique().tolist()[:max_n] if "Symbol" in latest.columns else []
    if len(symbols) < 2:
        company_syms = latest["company"].head(max_n).tolist()
        return html.P(f"Need stock symbols to optimize. Found companies: {', '.join(company_syms[:5])}. "
                       "Symbols are matched from NSE data — ensure holdings tab ran with classification.", className="text-warning")

    prices = fetch_stock_prices(symbols)
    if prices.shape[1] < 2:
        return html.P("Could not fetch enough price data for optimization.", className="text-warning")

    frontier = compute_efficient_frontier(prices, num_portfolios=3000)
    optimal = optimize_portfolio(prices, "max_sharpe")
    min_vol = optimize_portfolio(prices, "min_vol")

    frontier_chart = create_efficient_frontier_chart(frontier, "Top Holdings")

    content = [dcc.Graph(figure=frontier_chart)]

    if optimal:
        content.append(dbc.Row([
            dbc.Col(metric_card("Optimal Return", f"{optimal['return_pct']:.1f}%", "Max Sharpe", "success"), md=3),
            dbc.Col(metric_card("Optimal Vol", f"{optimal['volatility_pct']:.1f}%"), md=3),
            dbc.Col(metric_card("Sharpe", f"{optimal['sharpe_ratio']:.2f}"), md=3),
            dbc.Col(metric_card("Stocks Used", str(len(optimal.get("allocation", {})))), md=3),
        ], className="mb-3"))

        alloc_pie = create_allocation_pie(optimal.get("allocation", {}), "Max Sharpe Allocation")
        content.append(dbc.Row([dbc.Col(dcc.Graph(figure=alloc_pie), md=6)]))

    if min_vol:
        content.append(dbc.Row([
            dbc.Col(metric_card("Min Vol Return", f"{min_vol['return_pct']:.1f}%"), md=4),
            dbc.Col(metric_card("Min Vol", f"{min_vol['volatility_pct']:.1f}%", "Lowest risk", "info"), md=4),
            dbc.Col(metric_card("Min Vol Sharpe", f"{min_vol['sharpe_ratio']:.2f}"), md=4),
        ], className="mt-3"))

    return html.Div(content)
