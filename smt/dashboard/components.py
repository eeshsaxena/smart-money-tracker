"""Small layout helpers shared by the dashboard's tabs."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, html
from dash.dash_table.Format import Format, Scheme

PCT = Format(precision=1, scheme=Scheme.percentage)
PCT0 = Format(precision=0, scheme=Scheme.percentage)
PCT2 = Format(precision=2, scheme=Scheme.percentage)
NUM2 = Format(precision=2, scheme=Scheme.fixed)
NUM1 = Format(precision=1, scheme=Scheme.fixed)
NUM0 = Format(precision=0, scheme=Scheme.fixed).group(True)


def kpi(label: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="text-muted small"),
        html.Div(value, className="fs-4 fw-semibold"),
        html.Div(sub, className="small text-muted"),
    ])), xs=6, md=3, className="mb-3")


def pct(v, digits: int = 1, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "–"
    return f"{v * 100:{'+' if signed else ''}.{digits}f}%"


def table(df: pd.DataFrame, id_: str | None = None, formats: dict | None = None, page_size: int = 15,
          **kw) -> dash_table.DataTable:
    formats = formats or {}
    cols = [{"name": c, "id": c, "type": "numeric" if c in formats else "text",
             **({"format": formats[c]} if c in formats else {})} for c in df.columns]
    return dash_table.DataTable(
        id=id_ or f"t{id(df)}", data=df.to_dict("records"), columns=cols, page_size=page_size,
        sort_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif", "fontSize": 13,
                    "padding": "6px 10px", "textAlign": "left",
                    "maxWidth": 320, "whiteSpace": "normal"},
        style_header={"fontWeight": 600, "backgroundColor": "#f1f5f9"}, **kw)


def note(text: str) -> html.P:
    return html.P(text, className="text-muted small mb-2")


def finding(text: str) -> html.P:
    """The one-sentence answer that heads an evidence section."""
    return html.P(text, className="mb-2")


def section(title: str, *children) -> html.Div:
    return html.Div([html.H5(title, className="mt-4 mb-2"), *children])
