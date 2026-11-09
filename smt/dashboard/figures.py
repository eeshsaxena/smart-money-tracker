"""Plotly figure builders. Pure: data in, figure out."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ACCENT = "#0f766e"
BENCH = "#94a3b8"
NEG = "#dc2626"
POS = "#16a34a"
SEQ = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#ea580c"]
TEMPLATE = "plotly_white"


def _style(fig: go.Figure, title: str = "", height: int = 380, pct_y: bool = False) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified", colorway=SEQ)
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#64748b"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=250)


def growth(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_scatter(x=df.index, y=df[col], name=col, mode="lines",
                        line=dict(width=2.2 if i == 0 else 1.6, color=SEQ[0] if i == 0 else BENCH))
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    return _style(fig, title)


def lines(df: pd.DataFrame, title: str, pct: bool = True, colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_scatter(x=df.index, y=df[col], name=col, mode="lines",
                        line=dict(width=1.8, color=(colors or SEQ)[i % len(colors or SEQ)]))
    return _style(fig, title, pct_y=pct)


def drawdown(dd: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(dd.columns):
        fig.add_scatter(x=dd.index, y=dd[col], name=col, mode="lines",
                        fill="tozeroy" if i == 0 else None,
                        line=dict(width=1.4, color=NEG if i == 0 else BENCH))
    return _style(fig, title, pct_y=True)


def calendar_bars(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_bar(x=df.index.astype(str), y=df[col], name=col, marker_color=SEQ[0] if i == 0 else BENCH)
    fig.update_layout(barmode="group")
    return _style(fig, title, pct_y=True)


def heatmap(mat: pd.DataFrame, title: str, fmt: str = ".2f", zmin=None, zmax=None,
            scale: str = "Teal") -> go.Figure:
    fig = px.imshow(mat, text_auto=fmt, color_continuous_scale=scale, zmin=zmin, zmax=zmax, aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, title, height=max(380, 34 * len(mat) + 120)).update_layout(hovermode="closest")


def frontier(opt: dict, labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    fr = opt["frontier"]
    fig.add_scatter(x=fr["volatility"], y=fr["return"], mode="lines", name="Efficient frontier",
                    line=dict(color=ACCENT, width=2.5))
    funds = opt["funds"]
    fig.add_scatter(x=funds["volatility"], y=funds["return"], mode="markers+text", name="Funds",
                    text=[labels.get(c, c) for c in funds.index], textposition="top center",
                    marker=dict(size=9, color=BENCH))
    for key, sym, color in (("max_sharpe", "star", "#d97706"), ("min_volatility", "diamond", "#2563eb")):
        p = opt[key]
        fig.add_scatter(x=[p["volatility"]], y=[p["return"]], mode="markers", name=key.replace("_", " ").title(),
                        marker=dict(size=16, symbol=sym, color=color))
    fig.update_xaxes(title="Volatility (annualised)", tickformat=".0%")
    fig.update_yaxes(title="Return (annualised)", tickformat=".0%")
    return _style(fig, "Risk vs return of fund combinations", height=460).update_layout(hovermode="closest")


def hbar(s: pd.Series, title: str, color: str = ACCENT, suffix: str = "%", height: int = 380) -> go.Figure:
    s = s.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color,
                           text=[f"{v:,.1f}{suffix}" for v in s.values], textposition="outside",
                           cliponaxis=False))
    hi = float(s.max()) if len(s) else 1.0
    fig.update_xaxes(range=[min(0.0, float(s.min()) * 1.25) if len(s) else 0, hi * 1.25])
    return _style(fig, title, height=height).update_layout(hovermode="closest", showlegend=False)


def diverging_bars(s: pd.Series, title: str) -> go.Figure:
    """Horizontal bars for a fraction that can be either side of zero."""
    s = s.sort_values()
    colors = [POS if v > 0 else NEG for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           text=[f"{v * 100:+.1f}%" for v in s.values], textposition="outside", cliponaxis=False))
    lo, hi = (float(s.min()), float(s.max())) if len(s) else (0.0, 0.0)
    span = max(hi - lo, 0.01)
    # Room either side for the labels drawn outside the bars.
    fig.update_xaxes(tickformat=".0%", zeroline=True, range=[min(lo, 0) - 0.15 * span, max(hi, 0) + 0.15 * span])
    return _style(fig, title, height=max(380, 26 * len(s) + 100)).update_layout(hovermode="closest", showlegend=False)


def flows(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.sort_values("net_flow_cr")
    fig = go.Figure(go.Bar(
        x=d["net_flow_cr"], y=d["instrument"], orientation="h",
        marker_color=[POS if v > 0 else NEG for v in d["net_flow_cr"]],
        text=[f"₹{v:,.0f} cr" for v in d["net_flow_cr"]], textposition="outside", cliponaxis=False,
        customdata=d[["funds_buying", "funds_selling"]],
        hovertemplate="%{y}<br>Net: ₹%{x:,.0f} cr<br>Funds buying: %{customdata[0]}"
                      "<br>Funds selling: %{customdata[1]}<extra></extra>"))
    return _style(fig, title, height=max(380, 26 * len(d) + 100)).update_layout(hovermode="closest")


def donut(s: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=s.index, values=s.values, hole=0.55, sort=False,
                           marker=dict(colors=SEQ), textinfo="label+percent"))
    return _style(fig, title, height=340).update_layout(showlegend=False)


def drift(df: pd.DataFrame, rule: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(["Large Cap", "Mid Cap", "Small Cap", "Foreign"]):
        if col in df and df[col].abs().sum() > 0:
            fig.add_bar(x=df["as_of"], y=df[col], name=col, marker_color=SEQ[i])
    fig.update_layout(barmode="stack")
    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    return _style(fig, f"Market-cap mix by month (SEBI rule: {rule or 'none for this category'})")
