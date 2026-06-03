"""Smart Money Tracker — MF Intelligence Dashboard."""

import dash
import dash_bootstrap_components as dbc

from src.dashboard.layout import create_layout
import src.dashboard.callbacks  # noqa: F401 — registers callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,
    ],
    title="Smart Money Tracker",
    suppress_callback_exceptions=True,
)

server = app.server
app.layout = create_layout()

if __name__ == "__main__":
    app.run(debug=True, port=8050)
