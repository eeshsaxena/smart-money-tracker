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
    color = "success" if signal == "ACCUMULATE" else "danger"
    return dbc.Badge(signal, color=color, className="ms-2")


def fund_manager_selector(managers: dict):
    options = []
    for key, info in managers.items():
        options.append({"label": info["name"], "value": key})
    return dbc.Select(
        id="fund-manager-select",
        options=options,
        value=list(managers.keys())[0] if managers else None,
        className="mb-3",
    )


def sip_input_form():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("SIP Calculator", className="card-title"),
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
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Fund Ticker (NSE)"),
                                dbc.Input(
                                    id="sip-ticker",
                                    type="text",
                                    value="0P0000XVAA.BO",
                                    placeholder="e.g. 0P0000XVAA.BO",
                                ),
                            ],
                            md=4,
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
                                    ],
                                    value="5y",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Button(
                    "Calculate XIRR",
                    id="calc-xirr-btn",
                    color="primary",
                    className="w-100",
                ),
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
