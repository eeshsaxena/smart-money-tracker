"""Smart Money Tracker — MF Intelligence Dashboard."""

import dash
import dash_bootstrap_components as dbc
import plotly.io as pio

pio.templates["smart_money"] = pio.templates["plotly_dark"]
pio.templates["smart_money"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,33,48,0.8)",
    font=dict(family="Inter, sans-serif", color="#e8eaed"),
    colorway=[
        "#00bfa5", "#2979ff", "#00c853", "#ff9100",
        "#ff1744", "#aa00ff", "#00b0ff", "#76ff03",
        "#ffd600", "#ff6d00",
    ],
)
pio.templates.default = "smart_money"

from src.dashboard.layout import create_layout
import src.dashboard.callbacks  # noqa: F401

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    title="Smart Money Tracker",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = app.server
app.layout = create_layout()

if __name__ == "__main__":
    app.run(debug=True, port=8050)
