"""SQLite storage. The refresh pipeline is the only writer; the dashboard only reads.

Keeping downloads out of the request path means pages load instantly and a
source being down never breaks the app, it just leaves the data a little older
(the Data status tab shows how old).
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smt import settings
from smt.sources.amfi import label_date

SCHEMA = """
CREATE TABLE IF NOT EXISTS funds (
    scheme_code     TEXT PRIMARY KEY,
    scheme_name     TEXT NOT NULL,      -- as published by AMFI
    expected_name   TEXT NOT NULL,      -- what smt/universe.py says it should be
    amc             TEXT NOT NULL,
    category        TEXT,               -- SEBI category from AMFI
    holdings_source TEXT,
    is_benchmark    INTEGER NOT NULL DEFAULT 0,
    verified        INTEGER NOT NULL,   -- 1 if AMFI's name matches expected_name
    verified_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nav (
    scheme_code TEXT NOT NULL, date TEXT NOT NULL, nav REAL NOT NULL,
    PRIMARY KEY (scheme_code, date)
);
DROP TABLE IF EXISTS market_cap;        -- replaced by one list per half-year
CREATE TABLE IF NOT EXISTS market_cap_lists (
    label TEXT NOT NULL, isin TEXT NOT NULL, company TEXT, nse_symbol TEXT, cap_class TEXT NOT NULL,
    PRIMARY KEY (label, isin)
);
CREATE TABLE IF NOT EXISTS holdings (
    scheme_code TEXT NOT NULL, as_of TEXT NOT NULL, isin TEXT NOT NULL, asset_class TEXT NOT NULL,
    instrument TEXT, industry TEXT, quantity REAL, market_value_lakh REAL, pct_nav REAL,
    PRIMARY KEY (scheme_code, as_of, isin, asset_class)
);
CREATE TABLE IF NOT EXISTS disclosure_files (
    url TEXT PRIMARY KEY, connector TEXT NOT NULL, as_of TEXT NOT NULL,
    fetched_at TEXT NOT NULL, bytes INTEGER, schemes_matched INTEGER, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, started_at TEXT NOT NULL,
    seconds REAL, status TEXT NOT NULL, rows INTEGER, message TEXT
);
CREATE INDEX IF NOT EXISTS holdings_isin ON holdings (isin);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or settings.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")      # readers are not blocked while a refresh writes
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------- #
# Writes (used by the pipeline)
# --------------------------------------------------------------------------- #
def upsert_funds(con: sqlite3.Connection, rows: list[dict]) -> None:
    con.executemany(
        """INSERT INTO funds VALUES (:scheme_code, :scheme_name, :expected_name, :amc, :category,
               :holdings_source, :is_benchmark, :verified, :verified_at)
           ON CONFLICT(scheme_code) DO UPDATE SET scheme_name=excluded.scheme_name,
               expected_name=excluded.expected_name, amc=excluded.amc, category=excluded.category,
               holdings_source=excluded.holdings_source, is_benchmark=excluded.is_benchmark,
               verified=excluded.verified, verified_at=excluded.verified_at""",
        rows,
    )


def replace_nav(con: sqlite3.Connection, scheme_code: str, df: pd.DataFrame) -> int:
    with con:
        con.execute("DELETE FROM nav WHERE scheme_code = ?", (scheme_code,))
        con.executemany(
            "INSERT INTO nav VALUES (?,?,?)",
            [(scheme_code, d.strftime("%Y-%m-%d"), float(v)) for d, v in zip(df["date"], df["nav"], strict=True)],
        )
    return len(df)


def replace_market_cap(con: sqlite3.Connection, df: pd.DataFrame, label: str) -> int:
    with con:
        con.execute("DELETE FROM market_cap_lists WHERE label = ?", (label,))
        con.executemany(
            "INSERT INTO market_cap_lists VALUES (?,?,?,?,?)",
            [(label, r.isin, r.company, r.nse_symbol, r.cap_class) for r in df.itertuples()],
        )
    return len(df)


def replace_holdings(con: sqlite3.Connection, scheme_code: str, as_of: str, df: pd.DataFrame) -> int:
    with con:
        con.execute("DELETE FROM holdings WHERE scheme_code = ? AND as_of = ?", (scheme_code, as_of))
        con.executemany(
            "INSERT INTO holdings VALUES (?,?,?,?,?,?,?,?,?)",
            [(scheme_code, as_of, r.isin, r.asset_class, r.instrument, r.industry,
              None if pd.isna(r.quantity) else float(r.quantity),
              None if pd.isna(r.market_value_lakh) else float(r.market_value_lakh),
              float(r.pct_nav)) for r in df.itertuples()],
        )
    return len(df)


def record_file(con: sqlite3.Connection, url: str, connector: str, as_of: str,
                size: int | None, matched: int, status: str) -> None:
    with con:
        con.execute(
            "INSERT OR REPLACE INTO disclosure_files VALUES (?,?,?,?,?,?,?)",
            (url, connector, as_of, now(), size, matched, status),
        )



@contextmanager
def job(con: sqlite3.Connection, name: str) -> Iterator[dict]:
    """Time a pipeline step and log its outcome. The body sets info['rows'] / info['message']."""
    info: dict = {"rows": 0, "message": ""}
    started, t0 = now(), time.perf_counter()
    try:
        yield info
        status = info.get("status", "ok")
    except Exception as e:  # noqa: BLE001 - logged, then re-raised
        status, info["message"] = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        with con:
            con.execute(
                "INSERT INTO refresh_log (job, started_at, seconds, status, rows, message) VALUES (?,?,?,?,?,?)",
                (name, started, round(time.perf_counter() - t0, 2), status, info["rows"], info["message"][:2000]),
            )


# --------------------------------------------------------------------------- #
# Reads (used by analytics and the dashboard)
# --------------------------------------------------------------------------- #
def funds(con: sqlite3.Connection, verified_only: bool = True) -> pd.DataFrame:
    sql = "SELECT * FROM funds" + (" WHERE verified = 1" if verified_only else "")
    return pd.read_sql_query(sql + " ORDER BY is_benchmark, amc, scheme_name", con)


def nav_wide(con: sqlite3.Connection, codes: list[str] | None = None) -> pd.DataFrame:
    """NAV table with a DatetimeIndex and one column per scheme code."""
    sql, params = "SELECT scheme_code, date, nav FROM nav", ()
    if codes:
        sql += f" WHERE scheme_code IN ({','.join('?' * len(codes))})"
        params = tuple(codes)
    df = pd.read_sql_query(sql, con, params=params, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="scheme_code", values="nav").sort_index()


def market_cap_labels(con: sqlite3.Connection) -> list[str]:
    """Half-year lists held, newest first."""
    labels = [r[0] for r in con.execute("SELECT DISTINCT label FROM market_cap_lists")]
    return sorted(labels, key=label_date, reverse=True)


def list_in_force(as_of: pd.Timestamp, labels: list[str]) -> str | None:
    """The list a portfolio dated `as_of` is judged against: the newest half-year ended before that date.

    Older portfolios fall back to the oldest list held. Using today's list for a 2023
    portfolio would class stocks by what they became, not by what they were then.
    """
    if not labels:
        return None
    day = as_of.date()
    return next((x for x in labels if label_date(x) < day), labels[-1])


def _classifier(lists: pd.DataFrame) -> tuple[dict, dict]:
    """Exact ISIN map, plus a 9-character prefix map for prefixes naming exactly one company."""
    exact = {r.isin: (r.cap_class, r.nse_symbol) for r in lists.itertuples()}
    prefix = lists["isin"].str[:9]
    unique = lists[~prefix.duplicated(keep=False)]
    return exact, {r.isin[:9]: (r.cap_class, r.nse_symbol) for r in unique.itertuples()}


def holdings(con: sqlite3.Connection, codes: list[str] | None = None) -> pd.DataFrame:
    """Holdings with AMFI's market-cap class as of each portfolio's date.

    A stock split or bonus issue gives a company a new ISIN, so a stock can be
    missing from an older AMFI list under its new ISIN. The first 9 characters
    (country, issuer, security type) stay the same, so those are the fallback key
    for equity when they identify exactly one company in the list.
    """
    sql, params = "SELECT * FROM holdings", ()
    if codes:
        sql += f" WHERE scheme_code IN ({','.join('?' * len(codes))})"
        params = tuple(codes)
    df = pd.read_sql_query(sql, con, params=params, parse_dates=["as_of"])
    df["cap_class"] = None
    df["nse_symbol"] = None
    labels = market_cap_labels(con)
    if df.empty or not labels:
        return df
    lists = pd.read_sql_query("SELECT label, isin, nse_symbol, cap_class FROM market_cap_lists", con)
    maps = {label: _classifier(g) for label, g in lists.groupby("label")}
    in_force = {d: list_in_force(d, labels) for d in df["as_of"].unique()}
    caps, symbols = [], []
    for r in df[["as_of", "isin", "asset_class"]].itertuples(index=False):
        exact, prefix = maps[in_force[r.as_of]]
        hit = exact.get(r.isin) or (prefix.get(r.isin[:9]) if r.asset_class.startswith("Equity") else None)
        caps.append(hit[0] if hit else None)
        symbols.append(hit[1] if hit else None)
    df["cap_class"], df["nse_symbol"] = caps, symbols
    return df


def refresh_log(con: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT job, started_at, seconds, status, rows, message FROM refresh_log ORDER BY id DESC LIMIT ?",
        con, params=(limit,),
    )


def disclosure_files(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM disclosure_files ORDER BY as_of DESC, connector", con)


def market_cap_label(con: sqlite3.Connection) -> str | None:
    labels = market_cap_labels(con)
    return labels[0] if labels else None
