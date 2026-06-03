"""Dash app layout — 9-tab MF Intelligence Dashboard with dark theme."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import FUND_MANAGERS, PEER_GROUPS
from src.dashboard.components import (
    fund_manager_selector,
    loading_spinner,
    sip_input_form,
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.I(className="bi bi-graph-up-arrow me-2", style={"fontSize": "1.5rem"}),
                    html.Span("Smart Money Tracker", className="fs-4 fw-bold"),
                ],
                className="d-flex align-items-center",
            ),
            html.Div(
                [
                    html.Span(
                        "MF Intelligence Dashboard",
                        className="opacity-50 me-3 d-none d-lg-inline small",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-download me-1"), "Export CSV"],
                        id="global-export-btn",
                        color="outline-light",
                        size="sm",
                    ),
                ],
                className="d-flex align-items-center",
            ),
        ],
        fluid=True,
        className="d-flex justify-content-between",
    ),
    color="dark",
    dark=True,
    className="mb-4 border-bottom",
    style={"borderColor": "var(--border-color)"},
)


def create_layout():
    return html.Div(
        [
            navbar,
            dcc.Download(id="csv-download"),
            dbc.Container(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(_smart_money_tab(), label="Smart Money Flow", tab_id="tab-smart-money"),
                            dbc.Tab(_holdings_tab(), label="Fund Holdings", tab_id="tab-holdings"),
                            dbc.Tab(_fingerprint_tab(), label="Manager Fingerprint", tab_id="tab-fingerprint"),
                            dbc.Tab(_drift_tab(), label="Style Drift", tab_id="tab-drift"),
                            dbc.Tab(_xirr_tab(), label="SIP XIRR", tab_id="tab-xirr"),
                            dbc.Tab(_overlap_tab(), label="Portfolio Overlap", tab_id="tab-overlap"),
                            dbc.Tab(_risk_tab(), label="Risk Analytics", tab_id="tab-risk"),
                            dbc.Tab(_screener_tab(), label="Fund Screener", tab_id="tab-screener"),
                            dbc.Tab(_backtest_tab(), label="Backtest", tab_id="tab-backtest"),
                            dbc.Tab(_frontier_tab(), label="Efficient Frontier", tab_id="tab-frontier"),
                            dbc.Tab(_sector_tab(), label="Sector Breakdown", tab_id="tab-sector"),
                            dbc.Tab(_attribution_tab(), label="BHB Attribution", tab_id="tab-attribution"),
                        ],
                        id="main-tabs",
                        active_tab="tab-smart-money",
                        className="mb-4",
                    ),
                ],
                fluid=True,
            ),
            dcc.Store(id="portfolio-store"),
            dcc.Store(id="all-portfolios-store"),
            dcc.Store(id="classification-store"),
            html.Footer(
                dbc.Container(
                    html.P(
                        "Data: AMFI India, SEBI Disclosures, NSE/BSE, yfinance | Built with Plotly Dash + scikit-learn",
                        className="text-muted text-center small py-3 mb-0",
                    ),
                    fluid=True,
                ),
                className="mt-4 border-top",
            ),
        ]
    )


def _smart_money_tab():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Smart Money Flow Detector", className="mb-2"),
                            html.P(
                                "What are India's top fund managers collectively buying and selling? "
                                "Aggregated signals across all tracked managers.",
                                className="text-muted small",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Months"),
                            dbc.Select(
                                id="flow-months",
                                options=[{"label": f"{m}m", "value": str(m)} for m in [2, 3, 6]],
                                value="3",
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-search me-1"), "Scan All"],
                            id="scan-flow-btn",
                            color="success",
                            className="w-100 mt-4",
                        ),
                        md=2,
                    ),
                ],
                className="mb-4",
            ),
            loading_spinner("flow-output"),
            html.Hr(),
            html.H5("Conviction Scores", className="mt-3"),
            html.P("Stocks ranked by aggregate weight across all fund managers.", className="text-muted small"),
            loading_spinner("conviction-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _holdings_tab():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col([html.H6("Fund Manager", className="text-muted small mb-1"), fund_manager_selector(FUND_MANAGERS)], md=4),
                    dbc.Col([html.H6("Fund", className="text-muted small mb-1"), dbc.Select(id="fund-select", options=[], className="mb-3")], md=4),
                    dbc.Col(
                        [
                            html.H6("History", className="text-muted small mb-1"),
                            dbc.Select(
                                id="months-select",
                                options=[{"label": f"{m}m", "value": str(m)} for m in [3, 6, 9, 12, 24]],
                                value="6",
                                className="mb-3",
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-cloud-download me-1"), "Fetch"], id="fetch-holdings-btn", color="primary", className="w-100 mt-4"),
                        md=2,
                    ),
                ],
                className="mb-3",
            ),
            loading_spinner("holdings-output"),
            html.Hr(),
            html.H5("Accumulation / Exit Signals"),
            loading_spinner("signals-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _fingerprint_tab():
    return dbc.Container(
        [
            html.H4("Fund Manager Style Fingerprint", className="mb-2"),
            html.P(
                "Cap tilt, growth vs value, sector concentration (Herfindahl), turnover ratio. "
                "K-Means clustering finds managers who think alike.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Radar View"),
                            dbc.Select(
                                id="fingerprint-manager",
                                options=[{"label": f"{m['name']} ({m['amc']})", "value": k} for k, m in FUND_MANAGERS.items()],
                                value=list(FUND_MANAGERS.keys())[0],
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col([dbc.Label("Clusters"), dbc.Select(id="num-clusters", options=[{"label": str(n), "value": str(n)} for n in [2, 3, 4, 5]], value="3")], md=2),
                    dbc.Col(dbc.Button([html.I(className="bi bi-fingerprint me-1"), "Compute"], id="fingerprint-btn", color="primary", className="w-100 mt-4"), md=3),
                ],
                className="mb-4",
            ),
            loading_spinner("fingerprint-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _overlap_tab():
    return dbc.Container(
        [
            html.H4("Portfolio Overlap Heatmap", className="mb-2"),
            html.P(
                "Most investors hold multiple funds thinking they're diversified — "
                "often 70% of holdings overlap. Check yours.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Select Funds"), dbc.Checklist(id="overlap-fund-checklist", options=[], value=[], className="mb-3")], md=6),
                    dbc.Col(
                        [
                            dbc.Label("Method"),
                            dbc.RadioItems(id="overlap-method", options=[
                                {"label": "Jaccard (count)", "value": "jaccard"},
                                {"label": "Weighted (AUM %)", "value": "weighted"},
                            ], value="jaccard", className="mb-3"),
                        ],
                        md=4,
                    ),
                    dbc.Col(dbc.Button([html.I(className="bi bi-grid-3x3 me-1"), "Compute"], id="overlap-btn", color="primary", className="w-100 mt-4"), md=2),
                ],
                className="mb-3",
            ),
            loading_spinner("overlap-output"),
            html.Hr(),
            html.H5("Common Holdings Detail", className="mt-3"),
            loading_spinner("common-holdings-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _drift_tab():
    return dbc.Container(
        [
            html.H4("Style-Drift Detector", className="mb-2"),
            html.P(
                "SEBI mandates allocation ranges per category. Track which funds drift outside mandate.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Category"),
                            dbc.Select(id="drift-category", options=[
                                {"label": "Large Cap (80%+)", "value": "large_cap"},
                                {"label": "Mid Cap (65%+)", "value": "mid_cap"},
                                {"label": "Small Cap (65%+)", "value": "small_cap"},
                                {"label": "Flexi Cap (no mandate)", "value": "flexi_cap"},
                                {"label": "Large & Mid (35% each)", "value": "large_and_mid_cap"},
                                {"label": "Multi Cap (25% each)", "value": "multi_cap"},
                            ], value="large_cap"),
                        ],
                        md=5,
                    ),
                    dbc.Col(dbc.Button([html.I(className="bi bi-shield-exclamation me-1"), "Scan"], id="drift-btn", color="warning", className="w-100 mt-4"), md=3),
                ],
                className="mb-4",
            ),
            loading_spinner("drift-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _xirr_tab():
    return dbc.Container(
        [
            sip_input_form(),
            loading_spinner("xirr-output"),
            html.Hr(),
            loading_spinner("xirr-growth-chart"),
            html.Hr(),
            html.H5("Peer Comparison", className="mt-3"),
            loading_spinner("peer-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _risk_tab():
    all_tickers = []
    for mgr in FUND_MANAGERS.values():
        for fund in mgr["funds"]:
            if fund.get("ticker"):
                all_tickers.append({"label": fund["name"], "value": fund["ticker"]})

    return dbc.Container(
        [
            html.H4("Risk & Return Analytics", className="mb-2"),
            html.P(
                "Sharpe, Sortino, max drawdown, volatility, rolling returns, monthly return heatmap.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Fund"),
                            dbc.Select(id="risk-ticker", options=all_tickers, value=all_tickers[0]["value"] if all_tickers else None),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Period"),
                            dbc.Select(id="risk-period", options=[
                                {"label": "1Y", "value": "1y"}, {"label": "3Y", "value": "3y"},
                                {"label": "5Y", "value": "5y"}, {"label": "10Y", "value": "10y"},
                                {"label": "Max", "value": "max"},
                            ], value="5y"),
                        ],
                        md=2,
                    ),
                    dbc.Col(dbc.Button([html.I(className="bi bi-bar-chart-line me-1"), "Analyze"], id="risk-btn", color="danger", className="w-100 mt-4"), md=3),
                    dbc.Col(dbc.Button([html.I(className="bi bi-grid me-1"), "Compare All"], id="risk-compare-btn", color="info", className="w-100 mt-4"), md=3),
                ],
                className="mb-4",
            ),
            loading_spinner("risk-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _screener_tab():
    return dbc.Container(
        [
            html.H4("Fund Screener", className="mb-2"),
            html.P(
                "Filter and rank funds by risk-return metrics. Find funds matching your criteria.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Min Sharpe Ratio"),
                        dbc.Input(id="screen-min-sharpe", type="number", value=0.5, step=0.1, min=0),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Max Drawdown (%)"),
                        dbc.Input(id="screen-max-dd", type="number", value=-30, step=5, max=0),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Min CAGR (%)"),
                        dbc.Input(id="screen-min-cagr", type="number", value=8, step=1),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Max Volatility (%)"),
                        dbc.Input(id="screen-max-vol", type="number", value=30, step=5),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Period"),
                        dbc.Select(id="screen-period", options=[
                            {"label": "3Y", "value": "3y"}, {"label": "5Y", "value": "5y"},
                            {"label": "10Y", "value": "10y"},
                        ], value="5y"),
                    ], md=2),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-funnel me-1"), "Screen"], id="screen-btn", color="primary", className="w-100 mt-4"),
                        md=2,
                    ),
                ],
                className="mb-4",
            ),
            loading_spinner("screener-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _backtest_tab():
    return dbc.Container(
        [
            html.H4("Backtest: Follow the Smart Money", className="mb-2"),
            html.P(
                "If you bought the top stocks being accumulated by fund managers each month, "
                "how would your portfolio have performed vs Nifty 50?",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Initial Capital (INR)"),
                        dbc.Input(id="bt-capital", type="number", value=100000, step=50000),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Top N Stocks per Rebalance"),
                        dbc.Select(id="bt-top-n", options=[
                            {"label": str(n), "value": str(n)} for n in [5, 10, 15, 20]
                        ], value="10"),
                    ], md=3),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-play-circle me-1"), "Run Backtest"], id="bt-btn", color="danger", className="w-100 mt-4"),
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            loading_spinner("backtest-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _frontier_tab():
    return dbc.Container(
        [
            html.H4("Markowitz Efficient Frontier", className="mb-2"),
            html.P(
                "Mean-Variance Optimization across top fund holdings. "
                "Find the optimal allocation that maximizes Sharpe ratio.",
                className="text-muted small",
            ),
            dbc.Alert(
                "Fetch holdings in Fund Holdings tab first. The optimizer uses top stocks from the latest portfolio.",
                color="info",
                className="small",
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max stocks to optimize"),
                    dbc.Select(id="frontier-n", options=[
                        {"label": str(n), "value": str(n)} for n in [5, 10, 15, 20]
                    ], value="10"),
                ], md=3),
                dbc.Col(
                    dbc.Button([html.I(className="bi bi-bullseye me-1"), "Compute Frontier"], id="frontier-btn", color="success", className="w-100 mt-4"),
                    md=3,
                ),
            ], className="mb-4"),
            loading_spinner("frontier-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _sector_tab():
    return dbc.Container(
        [
            html.H4("Sector Breakdown", className="mb-2"),
            html.P(
                "Sector allocation pie chart, holdings treemap, sector evolution over time.",
                className="text-muted small",
            ),
            dbc.Alert(
                "Fetch holdings in the Fund Holdings tab first, then come here to analyze sector composition.",
                color="info",
                className="small",
            ),
            loading_spinner("sector-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _attribution_tab():
    return dbc.Container(
        [
            html.H4("Brinson-Hood-Beebower Attribution", className="mb-2"),
            html.P(
                "Decompose outperformance: Allocation (right sectors?), Selection (right stocks?), Interaction.",
                className="text-muted small",
            ),
            dbc.Button([html.I(className="bi bi-pie-chart me-1"), "Run Attribution"], id="attribution-btn", color="success", className="mb-4"),
            loading_spinner("attribution-output"),
        ],
        fluid=True,
        className="py-3",
    )
