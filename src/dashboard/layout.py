"""Dash app layout."""

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
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Holdings", href="#", active=True)),
                    dbc.NavItem(dbc.NavLink("Overlap", href="#")),
                    dbc.NavItem(dbc.NavLink("Style Drift", href="#")),
                    dbc.NavItem(dbc.NavLink("SIP XIRR", href="#")),
                    dbc.NavItem(dbc.NavLink("Attribution", href="#")),
                ],
                navbar=True,
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
                                _holdings_tab(),
                                label="Fund Manager Holdings",
                                tab_id="tab-holdings",
                            ),
                            dbc.Tab(
                                _overlap_tab(),
                                label="Portfolio Overlap",
                                tab_id="tab-overlap",
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
                                _attribution_tab(),
                                label="BHB Attribution",
                                tab_id="tab-attribution",
                            ),
                        ],
                        id="main-tabs",
                        active_tab="tab-holdings",
                        className="mb-4",
                    ),
                ],
                fluid=True,
            ),
            dcc.Store(id="portfolio-store"),
            dcc.Store(id="classification-store"),
        ]
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
                                    for m in [3, 6, 9, 12]
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


def _overlap_tab():
    return dbc.Container(
        [
            html.H5("Compare Funds", className="mb-3"),
            html.P(
                "Select multiple fund tickers to see portfolio overlap.",
                className="text-muted",
            ),
            dbc.Textarea(
                id="overlap-tickers",
                placeholder="Enter fund names (one per line)",
                value="HDFC Flexi Cap Fund\nPPFAS Flexi Cap Fund",
                style={"height": "120px"},
                className="mb-3",
            ),
            dbc.Button("Compute Overlap", id="overlap-btn", color="primary", className="mb-4"),
            loading_spinner("overlap-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _drift_tab():
    return dbc.Container(
        [
            html.H5("Style-Drift Detector", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Fund Category"),
                            dbc.Select(
                                id="drift-category",
                                options=[
                                    {"label": "Large Cap", "value": "large_cap"},
                                    {"label": "Mid Cap", "value": "mid_cap"},
                                    {"label": "Small Cap", "value": "small_cap"},
                                    {"label": "Flexi Cap", "value": "flexi_cap"},
                                    {"label": "Large & Mid Cap", "value": "large_and_mid_cap"},
                                ],
                                value="flexi_cap",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Analyze Drift", id="drift-btn", color="warning", className="mb-4"
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
            html.H5("Peer Comparison", className="mt-4 mb-3"),
            dbc.Textarea(
                id="peer-tickers",
                placeholder="Ticker=Name (one per line)\n0P0000XVAA.BO=PPFAS Flexi Cap\n0P0000XVAL.BO=HDFC Flexi Cap",
                value="0P0000XVAA.BO=PPFAS Flexi Cap\n0P0000XVAL.BO=HDFC Flexi Cap",
                style={"height": "100px"},
                className="mb-3",
            ),
            dbc.Button(
                "Compare Peers", id="peer-compare-btn", color="info", className="mb-4"
            ),
            loading_spinner("peer-output"),
        ],
        fluid=True,
        className="py-3",
    )


def _attribution_tab():
    return dbc.Container(
        [
            html.H5("Brinson-Hood-Beebower Performance Attribution", className="mb-3"),
            html.P(
                "Decomposes fund alpha into allocation, selection, and interaction effects.",
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
