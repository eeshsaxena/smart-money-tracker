"""Reusable Dash UI components."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def metric_card(title: str, value: str, subtitle: str = "", color: str = "primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-subtitle mb-1 text-muted"),
                html.H3(value, className=f"card-title text-{color}"),
                html.P(subtitle, className="card-text small text-muted") if subtitle else None,
            ]
        ),
        className="shadow-sm h-100",
    )


def signal_badge(signal: str):
    color_map = {
        "ACCUMULATE": "success",
        "EXIT": "danger",
        "NEW ENTRY": "info",
        "COMPLETE EXIT": "dark",
        "STRONG": "danger",
        "MODERATE": "warning",
        "WEAK": "secondary",
    }
    return dbc.Badge(signal, color=color_map.get(signal, "primary"), className="ms-2")


def fund_manager_selector(managers: dict):
    options = []
    for key, info in managers.items():
        options.append({"label": f"{info['name']} ({info['amc']})", "value": key})
    return dbc.Select(
        id="fund-manager-select",
        options=options,
        value=list(managers.keys())[0] if managers else None,
        className="mb-3",
    )


def sip_input_form():
    from config import PEER_GROUPS

    peer_options = [{"label": name, "value": name} for name in PEER_GROUPS.keys()]

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("SIP XIRR Calculator", className="card-title"),
                html.P(
                    "Compute true annualized returns (XIRR) accounting for monthly cash flows — not fake point-to-point returns.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Monthly SIP Amount (INR)"),
                                dbc.Input(
                                    id="sip-amount",
                                    type="number",
                                    value=10000,
                                    min=500,
                                    step=500,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Fund Ticker (BSE/NSE)"),
                                dbc.Input(
                                    id="sip-ticker",
                                    type="text",
                                    value="0P0000XVAA.BO",
                                    placeholder="e.g. 0P0000XVAA.BO",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Period"),
                                dbc.Select(
                                    id="sip-period",
                                    options=[
                                        {"label": "1 Year", "value": "1y"},
                                        {"label": "3 Years", "value": "3y"},
                                        {"label": "5 Years", "value": "5y"},
                                        {"label": "10 Years", "value": "10y"},
                                        {"label": "Max", "value": "max"},
                                    ],
                                    value="5y",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Peer Group"),
                                dbc.Select(
                                    id="peer-group-select",
                                    options=peer_options,
                                    value=peer_options[0]["value"] if peer_options else None,
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Calculate XIRR", id="calc-xirr-btn", color="primary", className="w-100"),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Button("Compare Peers", id="peer-compare-btn", color="info", className="w-100"),
                        md=6,
                    ),
                ]),
            ]
        ),
        className="shadow-sm mb-4",
    )


def loading_spinner(component_id: str):
    return dcc.Loading(
        id=f"loading-{component_id}",
        type="circle",
        children=html.Div(id=component_id),
    )


def empty_state(message: str, icon: str = "bi-inbox"):
    return html.Div(
        [
            html.I(className=f"bi {icon} display-4 text-muted"),
            html.P(message, className="text-muted mt-2"),
        ],
        className="text-center py-5",
    )
