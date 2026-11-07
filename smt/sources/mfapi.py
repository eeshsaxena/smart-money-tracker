"""Full NAV history per scheme from mfapi.in (a free mirror of AMFI's NAV archive)."""
from __future__ import annotations

import pandas as pd

from smt import http

URL = "https://api.mfapi.in/mf/{code}"


def parse_history(payload: dict) -> tuple[str, pd.DataFrame]:
    """mfapi JSON -> (scheme name, DataFrame[date, nav]) sorted by date, de-duplicated."""
    name = (payload.get("meta") or {}).get("scheme_name", "")
    rows = []
    for entry in payload.get("data") or []:
        try:
            rows.append((pd.to_datetime(entry["date"], format="%d-%m-%Y"), float(entry["nav"])))
        except (KeyError, ValueError, TypeError):
            continue
    df = pd.DataFrame(rows, columns=["date", "nav"])
    df = df[df["nav"] > 0].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return name, df


def fetch_history(scheme_code: str) -> tuple[str, pd.DataFrame]:
    return parse_history(http.get(URL.format(code=scheme_code)).json())
