"""Dashboard: build the app on an offline database and run every callback through Flask."""
from __future__ import annotations

import json

import pytest

from smt import pipeline, universe
from smt.dashboard.app import create_app


@pytest.fixture
def app(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=2, src=fake_sources)
    return create_app(db)


INPUTS = {
    "score-cat.value": None,
    "fund.value": "122639", "fund-period.value": "3Y",
    "sip-funds.value": ["122639", universe.BENCHMARK_CODE], "sip-amount.value": 10000, "sip-start.value": 2024,
    "port-funds.value": ["122639", "119076", universe.BENCHMARK_CODE],
    "hold-fund.value": "122639", "hold-month.value": "2026-09-30",
    "flow-month.value": "2026-09-30",
    "ov-a.value": "122639", "ov-b.value": "119076",
}


def call(app, key: str, spec: dict):
    outputs = [{"id": o.split(".")[0], "property": o.split(".")[1]}
               for o in key.strip(".").split("...")]
    payload = {
        "output": key,
        "outputs": outputs if len(outputs) > 1 else outputs[0],
        "inputs": [{"id": i["id"], "property": i["property"], "value": INPUTS[f"{i['id']}.{i['property']}"]}
                   for i in spec["inputs"]],
        "changedPropIds": [f"{i['id']}.{i['property']}" for i in spec["inputs"]],
        "state": [],
    }
    return app.server.test_client().post("/_dash-update-component", json=payload)


def test_layout_serves(app):
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout = json.dumps(client.get("/_dash-layout").get_json())
    for tab in ("Scorecard", "Holdings", "Smart money", "Data status"):
        assert tab in layout


def test_every_callback_runs_without_error(app):
    app.server.test_client().get("/")
    assert len(app.callback_map) == 8
    for key, spec in app.callback_map.items():
        resp = call(app, key, spec)
        assert resp.status_code == 200, f"{key}: {resp.data[:500]}"
        body = resp.get_json()
        assert "response" in body, key


def test_empty_database_shows_instructions(tmp_path):
    app = create_app(tmp_path / "empty.db")
    layout = json.dumps(app.server.test_client().get("/_dash-layout").get_json())
    assert "smt refresh" in layout
