"""Dash app layout — 6-tab MF Intelligence Dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import FUND_MANAGERS
from src.dashboard.components import (
    fund_manager_selector,
    loading_spinner,
    sip_input_form,
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                [
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    "Smart Money Tracker",
                ],
                className="fs-4 fw-bold",
            ),
            html.Span(
                "MF Intelligence Dashboard",
                className="text-light opacity-75 ms-3 d-none d-md-inline",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-4",
)


def create_layout():
    return html.Div(
        [
            navbar,
            dbc.Container(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                _smart_money_tab(),
                                label="Smart Money Flow",
                                tab_id="tab-smart-money",
                            ),
                            dbc.Tab(
                                _holdings_tab(),
                                label="Fund Holdings",
                                tab_id="tab-holdings",
                            ),
                            dbc.Tab(
                                _fingerprint_tab(),
                                label="Manager Fingerprint",
                                tab_id="tab-fingerprint",
                            ),
                            dbc.Tab(
                                _drift_tab(),
                                label="Style Drift",
                                tab_id="tab-drift",
                            ),
                            dbc.Tab(
                                _xirr_tab(),
                                label="SIP XIRR",
                                tab_id="tab-xirr",
                            ),
                            dbc.Tab(
                                _overlap_tab(),
                                label="Portfolio Overlap",
                                tab_id="tab-overlap",
                            ),
                            dbc.Tab(
                                _attribution_tab(),
                                label="BHB Attribution",
                                tab_id="tab-attribution",
                            ),
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
                        [
                            "Data: AMFI India, SEBI Disclosures, NSE/BSE, yfinance | ",
                            "Built with Plotly Dash",
                        ],
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
                                "What are India's top fund managers collectively buying and selling right now? "
                                "Aggregated signals across all tracked managers.",
                                className="text-muted",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Months to Compare"),
                            dbc.Select(
                                id="flow-months",
                                options=[{"label": f"{m} months", "value": str(m)} for m in [2, 3, 6]],
                                value="3",
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Scan All Managers",
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
                    dbc.Col(
                        [
                            html.H5("Select Fund Manager"),
                            fund_manager_selector(FUND_MANAGERS),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.H5("Select Fund"),
                            dbc.Select(id="fund-select", options=[], className="mb-3"),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.H5("Months to Fetch"),
                            dbc.Select(
                                id="months-select",
                                options=[
                                    {"label": f"{m} months", "value": str(m)}
                                    for m in [3, 6, 9, 12, 24]
                                ],
                                value="6",
                                className="mb-3",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Fetch Holdings", id="fetch-holdings-btn", color="primary", className="mb-4"
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
                "Large vs mid vs small cap tilt, growth vs value score, sector concentration (Herfindahl), "
                "turnover ratio. Managers clustered by K-Means — find who thinks alike.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Select Manager for Radar"),
                            dbc.Select(
                                id="fingerprint-manager",
                                options=[
                                    {"label": f"{m['name']} ({m['amc']})", "value": k}
                                    for k, m in FUND_MANAGERS.items()
                                ],
                                value=list(FUND_MANAGERS.keys())[0],
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Number of Clusters"),
                            dbc.Select(
                                id="num-clusters",
                                options=[{"label": str(n), "value": str(n)} for n in [2, 3, 4, 5]],
                                value="3",
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Compute Fingerprints",
                            id="fingerprint-btn",
                            color="primary",
                            className="w-100 mt-4",
                        ),
                        md=3,
                    ),
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
                "often 70% of holdings are the same stocks. Check your overlap.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Select Funds to Compare"),
                            dbc.Checklist(
                                id="overlap-fund-checklist",
                                options=[],
                                value=[],
                                className="mb-3",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Overlap Method"),
                            dbc.RadioItems(
                                id="overlap-method",
                                options=[
                                    {"label": "Jaccard (stock count)", "value": "jaccard"},
                                    {"label": "Weighted (by AUM %)", "value": "weighted"},
                                ],
                                value="jaccard",
                                className="mb-3",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Compute Overlap", id="overlap-btn", color="primary", className="w-100 mt-4"
                        ),
                        md=2,
                    ),
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
                "SEBI mandates market-cap allocation ranges for each fund category. "
                "Track which funds are drifting outside their mandate — a real red flag for investors.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Fund Category to Analyze"),
                            dbc.Select(
                                id="drift-category",
                                options=[
                                    {"label": "Large Cap (80%+ large)", "value": "large_cap"},
                                    {"label": "Mid Cap (65%+ mid)", "value": "mid_cap"},
                                    {"label": "Small Cap (65%+ small)", "value": "small_cap"},
                                    {"label": "Flexi Cap (no mandate)", "value": "flexi_cap"},
                                    {"label": "Large & Mid Cap (35% each)", "value": "large_and_mid_cap"},
                                    {"label": "Multi Cap (25% each)", "value": "multi_cap"},
                                ],
                                value="large_cap",
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Scan for Drift", id="drift-btn", color="warning", className="w-100 mt-4"
                        ),
                        md=3,
                    ),
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


def _attribution_tab():
    return dbc.Container(
        [
            html.H4("Brinson-Hood-Beebower Performance Attribution", className="mb-2"),
            html.P(
                "Decompose a fund's outperformance into: Allocation Effect (right sectors?), "
                "Selection Effect (right stocks?), and Interaction Effect.",
                className="text-muted",
            ),
            dbc.Button(
                "Run Attribution",
                id="attribution-btn",
                color="success",
                className="mb-4",
            ),
            loading_spinner("attribution-output"),
        ],
        fluid=True,
        className="py-3",
    )
