"""Dash callbacks wiring UI interactions to analysis logic."""

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, dash_table
import dash_bootstrap_components as dbc

from config import FUND_MANAGERS
from src.dashboard.components import metric_card, signal_badge
from src.scraper.sebi_scraper import (
    detect_accumulation_signals,
    fetch_multi_month_portfolio,
)
from src.scraper.nse_classifier import match_holdings_to_classification
from src.analysis.overlap import compute_overlap_matrix, create_overlap_heatmap
from src.analysis.style_drift import (
    compute_drift_over_time,
    create_drift_chart,
    detect_style_drift,
)
from src.analysis.xirr_calc import benchmark_sip, compare_category_peers, compute_sip_xirr, fetch_nav_series
from src.analysis.bhb_attribution import (
    compute_bhb_attribution,
    create_attribution_chart,
    create_attribution_waterfall,
)


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
    prevent_initial_call=True,
)
def fetch_and_display_holdings(n_clicks, scheme_code, months):
    if not scheme_code:
        return html.P("Select a fund first.", className="text-warning"), "", None

    months = int(months) if months else 6
    portfolio = fetch_multi_month_portfolio(scheme_code, months)

    if portfolio.empty:
        return (
            html.P("No data found. The fund may not have public disclosures yet.",
                   className="text-warning"),
            "",
            None,
        )

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
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
    )

    summary = dbc.Row(
        [
            dbc.Col(metric_card("Total Stocks", str(len(latest)), f"as of {latest_date.strftime('%b %Y')}"), md=3),
            dbc.Col(metric_card("Top Holding", latest.iloc[0]["company"] if len(latest) > 0 else "N/A", f"{latest.iloc[0]['pct_aum']:.1f}% AUM" if len(latest) > 0 else ""), md=3),
            dbc.Col(metric_card("Large Cap %", f"{latest[latest['market_cap_category'] == 'large_cap']['pct_aum'].sum():.1f}%"), md=3),
            dbc.Col(metric_card("Mid+Small Cap %", f"{latest[latest['market_cap_category'].isin(['mid_cap', 'small_cap'])]['pct_aum'].sum():.1f}%"), md=3),
        ],
        className="mb-4",
    )

    signals_df = detect_accumulation_signals(portfolio)
    if signals_df.empty:
        signals_content = html.P("No significant accumulation/exit signals detected.", className="text-muted")
    else:
        signal_rows = []
        for _, row in signals_df.iterrows():
            signal_rows.append(
                html.Tr(
                    [
                        html.Td(row["company"]),
                        html.Td(f"{row['previous_pct']}%"),
                        html.Td(f"{row['current_pct']}%"),
                        html.Td(f"{row['change_pct']:+.2f}%"),
                        html.Td(signal_badge(row["signal"])),
                    ]
                )
            )
        signals_content = dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th(c) for c in ["Company", "Previous %", "Current %", "Change", "Signal"]])
                ),
                html.Tbody(signal_rows),
            ],
            bordered=True,
            hover=True,
            striped=True,
            className="mt-3",
        )

    return (
        html.Div([summary, holdings_table]),
        signals_content,
        portfolio.to_json(date_format="iso"),
    )


@callback(
    Output("overlap-output", "children"),
    Input("overlap-btn", "n_clicks"),
    State("portfolio-store", "data"),
    State("overlap-tickers", "value"),
    prevent_initial_call=True,
)
def compute_and_display_overlap(n_clicks, stored_data, ticker_text):
    if not stored_data:
        return html.P("Fetch holdings first in the Holdings tab.", className="text-warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty:
        return html.P("No portfolio data available.", className="text-warning")

    fund_names = [t.strip() for t in (ticker_text or "").split("\n") if t.strip()]
    if len(fund_names) < 2:
        return html.P("Enter at least 2 fund names.", className="text-warning")

    # Demo: split stored portfolio by date as proxy for different funds
    dates = sorted(portfolio["date"].unique())
    fund_holdings = {}
    for i, name in enumerate(fund_names):
        idx = i % len(dates)
        fund_holdings[name] = portfolio[portfolio["date"] == dates[idx]]

    matrix = compute_overlap_matrix(fund_holdings)
    fig = create_overlap_heatmap(matrix)
    return dbc.Row([dbc.Col(dcc.Graph(figure=fig))])


@callback(
    Output("drift-output", "children"),
    Input("drift-btn", "n_clicks"),
    State("portfolio-store", "data"),
    State("drift-category", "value"),
    prevent_initial_call=True,
)
def analyze_drift(n_clicks, stored_data, category):
    if not stored_data:
        return html.P("Fetch holdings first in the Holdings tab.", className="text-warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty:
        return html.P("No portfolio data available.", className="text-warning")

    drift_result = detect_style_drift(portfolio[portfolio["date"] == portfolio["date"].max()], category)
    drift_history = compute_drift_over_time(portfolio, category)
    drift_fig = create_drift_chart(drift_history, f"Fund ({category})")

    status_color = "danger" if drift_result["drifted"] else "success"
    status_text = "DRIFT DETECTED" if drift_result["drifted"] else "WITHIN MANDATE"

    violation_cards = []
    for v in drift_result.get("violations", []):
        violation_cards.append(
            dbc.Alert(
                f"{v['category'].replace('_', ' ').title()}: {v['actual_pct']:.1f}% "
                f"(required {v['required_pct']:.1f}%, shortfall {v['shortfall_pct']:.1f}%)",
                color="danger",
            )
        )

    return html.Div(
        [
            dbc.Alert(status_text, color=status_color, className="text-center fs-5"),
            *violation_cards,
            dcc.Graph(figure=drift_fig),
        ]
    )


@callback(
    Output("xirr-output", "children"),
    Input("calc-xirr-btn", "n_clicks"),
    State("sip-amount", "value"),
    State("sip-ticker", "value"),
    State("sip-period", "value"),
    prevent_initial_call=True,
)
def calculate_xirr(n_clicks, sip_amount, ticker, period):
    if not ticker or not sip_amount:
        return html.P("Enter ticker and SIP amount.", className="text-warning")

    result = benchmark_sip(ticker, float(sip_amount), period)

    fund = result["fund"]
    bench = result["benchmark"]

    cards = dbc.Row(
        [
            dbc.Col(
                metric_card(
                    "Fund XIRR",
                    f"{fund['xirr']}%" if fund["xirr"] else "N/A",
                    f"Invested: INR {fund['total_invested']:,.0f}",
                    "success" if fund["xirr"] and fund["xirr"] > 0 else "danger",
                ),
                md=3,
            ),
            dbc.Col(
                metric_card(
                    "Current Value",
                    f"INR {fund['current_value']:,.0f}" if fund["current_value"] else "N/A",
                    f"{fund['num_installments']} installments",
                ),
                md=3,
            ),
            dbc.Col(
                metric_card(
                    "Nifty 50 XIRR",
                    f"{bench['xirr']}%" if bench["xirr"] else "N/A",
                    "Benchmark",
                ),
                md=3,
            ),
            dbc.Col(
                metric_card(
                    "Alpha vs Nifty",
                    f"{result['alpha']:+.2f}%" if result["alpha"] else "N/A",
                    "Outperformance" if result["alpha"] and result["alpha"] > 0 else "Underperformance",
                    "success" if result["alpha"] and result["alpha"] > 0 else "danger",
                ),
                md=3,
            ),
        ],
        className="mb-4",
    )
    return cards


@callback(
    Output("peer-output", "children"),
    Input("peer-compare-btn", "n_clicks"),
    State("peer-tickers", "value"),
    State("sip-amount", "value"),
    State("sip-period", "value"),
    prevent_initial_call=True,
)
def compare_peers(n_clicks, peer_text, sip_amount, period):
    if not peer_text:
        return html.P("Enter peer tickers.", className="text-warning")

    tickers = {}
    for line in peer_text.strip().split("\n"):
        if "=" in line:
            ticker, name = line.split("=", 1)
            tickers[name.strip()] = ticker.strip()

    if len(tickers) < 2:
        return html.P("Enter at least 2 peers in 'TICKER=Name' format.", className="text-warning")

    df = compare_category_peers(tickers, float(sip_amount or 10000), period or "5y")

    if df.empty:
        return html.P("Could not fetch data for these tickers.", className="text-warning")

    table = dash_table.DataTable(
        data=df[["fund_name", "xirr", "total_invested", "current_value", "num_installments"]].to_dict("records"),
        columns=[
            {"name": "Fund", "id": "fund_name"},
            {"name": "XIRR (%)", "id": "xirr", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Invested", "id": "total_invested", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Current Value", "id": "current_value", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Installments", "id": "num_installments"},
        ],
        sort_action="native",
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
    )
    return table


@callback(
    Output("attribution-output", "children"),
    Input("attribution-btn", "n_clicks"),
    State("portfolio-store", "data"),
    prevent_initial_call=True,
)
def run_attribution(n_clicks, stored_data):
    if not stored_data:
        return html.P("Fetch holdings first in the Holdings tab.", className="text-warning")

    portfolio = pd.read_json(stored_data)
    if portfolio.empty or "sector" not in portfolio.columns:
        return html.P("No sector data available for attribution.", className="text-warning")

    latest = portfolio[portfolio["date"] == portfolio["date"].max()]
    sectors = latest["sector"].dropna().unique()

    total_aum = latest["pct_aum"].sum()
    port_weights = latest.groupby("sector")["pct_aum"].sum() / total_aum

    # Simulate benchmark weights as equal-weight across sectors
    bench_weights = pd.Series(1.0 / len(sectors), index=sectors)

    # Simulate returns (in production, fetch from market data)
    import numpy as np
    np.random.seed(42)
    port_returns = pd.Series(np.random.normal(0.12, 0.05, len(sectors)), index=sectors)
    bench_returns = pd.Series(np.random.normal(0.10, 0.03, len(sectors)), index=sectors)

    attr_df = compute_bhb_attribution(port_weights, bench_weights, port_returns, bench_returns)
    bar_fig = create_attribution_chart(attr_df)
    waterfall_fig = create_attribution_waterfall(attr_df)

    totals = attr_df[attr_df["sector"] == "TOTAL"].iloc[0]

    summary = dbc.Row(
        [
            dbc.Col(metric_card("Allocation Effect", f"{totals['allocation_effect']:+.2f}%"), md=4),
            dbc.Col(metric_card("Selection Effect", f"{totals['selection_effect']:+.2f}%", color="success"), md=4),
            dbc.Col(metric_card("Total Alpha", f"{totals['total_effect']:+.2f}%", color="info"), md=4),
        ],
        className="mb-4",
    )

    return html.Div(
        [
            summary,
            dbc.Row([dbc.Col(dcc.Graph(figure=waterfall_fig))]),
            dbc.Row([dbc.Col(dcc.Graph(figure=bar_fig))]),
        ]
    )
