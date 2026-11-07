"""Refresh pipeline: download from the sources, validate, and write to SQLite.

Steps (each logged in refresh_log, each isolated so one failure doesn't stop the rest):
  1. universe     - check every scheme code against AMFI's scheme master
  2. nav          - full NAV history for every verified fund
  3. market_cap   - AMFI's large / mid / small cap list
  4. holdings     - month-end portfolios from each AMC connector

Sources are injected (see `Sources`) so tests can run the whole pipeline offline.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from smt import settings, store, universe
from smt.holdings.connectors import Connector, DisclosureFile, default_connectors
from smt.sources import amfi, mfapi

log = logging.getLogger("smt.pipeline")


@dataclass
class Sources:
    scheme_master: Callable[[], pd.DataFrame] = amfi.fetch_scheme_master
    nav_history: Callable[[str], tuple[str, pd.DataFrame]] = mfapi.fetch_history
    market_cap: Callable[[], tuple[str, pd.DataFrame]] = amfi.fetch_market_cap
    connectors: dict[str, Connector] = field(default_factory=default_connectors)
    raw_dir: Path = settings.RAW_DIR      # downloaded disclosure files are kept here


def cached_download(connector: Connector, f: DisclosureFile, raw_dir: Path) -> bytes:
    """Download a disclosure file once and keep it, so re-parsing never re-downloads.

    Month-end files don't change after publication, and keeping the originals
    means every number in the database can be traced back to its source file.
    """
    name = f.url.split("?")[0].rsplit("/", 1)[-1]
    path = raw_dir / connector.name / f"{hashlib.sha1(f.url.encode()).hexdigest()[:10]}_{name}"
    if path.exists():
        return path.read_bytes()
    content = connector.download(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


@dataclass
class Report:
    verified: int = 0
    mismatched: list[str] = field(default_factory=list)
    nav_rows: int = 0
    holdings_rows: int = 0
    files: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def refresh_universe(con: sqlite3.Connection, src: Sources, report: Report) -> None:
    with store.job(con, "universe") as info:
        master = src.scheme_master().set_index("scheme_code")
        rows = []
        for f in universe.FUNDS:
            amfi_row = master.loc[f.scheme_code] if f.scheme_code in master.index else None
            if isinstance(amfi_row, pd.DataFrame):
                amfi_row = amfi_row.iloc[0]
            published = "" if amfi_row is None else str(amfi_row["scheme_name"])
            ok = universe.normalise(published) == universe.normalise(f.expected_name)
            if not ok:
                report.mismatched.append(f"{f.scheme_code}: expected '{f.expected_name}', AMFI says '{published or 'not found'}'")
            rows.append({
                "scheme_code": f.scheme_code, "scheme_name": published or f.expected_name,
                "expected_name": f.expected_name, "amc": f.amc,
                "category": None if amfi_row is None else amfi_row["category"],
                "holdings_source": f.holdings, "is_benchmark": int(f.is_benchmark),
                "verified": int(ok), "verified_at": store.now(),
            })
        with con:
            store.upsert_funds(con, rows)
        report.verified = sum(r["verified"] for r in rows)
        info["rows"] = report.verified
        if report.mismatched:
            info["status"] = "warning"
            info["message"] = "; ".join(report.mismatched)


def refresh_nav(con: sqlite3.Connection, src: Sources, report: Report) -> None:
    codes = store.funds(con)["scheme_code"].tolist()
    with store.job(con, "nav") as info:
        failed = []
        for code in codes:
            try:
                _, df = src.nav_history(code)
                if df.empty:
                    raise ValueError("no NAV rows")
                report.nav_rows += store.replace_nav(con, code, df)
            except Exception as e:  # noqa: BLE001 - one fund failing must not stop the rest
                failed.append(f"{code}: {e}")
                log.warning("NAV %s failed: %s", code, e)
        info["rows"] = report.nav_rows
        if failed:
            report.errors += failed
            info["status"], info["message"] = "warning", "; ".join(failed)


def refresh_market_cap(con: sqlite3.Connection, src: Sources, report: Report) -> None:
    with store.job(con, "market_cap") as info:
        try:
            label, df = src.market_cap()
            info["rows"] = store.replace_market_cap(con, df, label)
            info["message"] = f"AMFI list for six months ended {label}"
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"market_cap: {e}")
            info["status"], info["message"] = "warning", str(e)


def refresh_holdings(con: sqlite3.Connection, src: Sources, report: Report,
                     months: int = settings.HOLDINGS_MONTHS) -> None:
    funds = store.funds(con)
    for name, connector in src.connectors.items():
        targets = {
            universe.normalise(r.expected_name): r.scheme_code
            for r in funds.itertuples() if r.holdings_source == name
        }
        if not targets:
            continue
        with store.job(con, f"holdings:{name}") as info:
            try:
                files = connector.discover()
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"{name} discover: {e}")
                info["status"], info["message"] = "error", f"could not list files: {e}"
                continue
            wanted_months = sorted({f.as_of for f in files}, reverse=True)[:months]
            problems = []
            for f in (f for f in files if f.as_of in wanted_months):
                try:
                    content = cached_download(connector, f, src.raw_dir)
                    matched = 0
                    for p in connector.parse(content):
                        code = targets.get(universe.normalise(p.scheme_title))
                        if code is None:
                            continue
                        as_of = (p.as_of or f.as_of).isoformat()
                        report.holdings_rows += store.replace_holdings(con, code, as_of, p.holdings)
                        matched += 1
                    store.record_file(con, f.url, name, f.as_of.isoformat(), len(content), matched, "ok")
                    report.files += 1
                    info["rows"] += matched
                except Exception as e:  # noqa: BLE001
                    store.record_file(con, f.url, name, f.as_of.isoformat(), None, 0, f"error: {e}")
                    problems.append(f"{f.url.rsplit('/', 1)[-1]}: {e}")
            if problems:
                report.errors += problems
                info["status"], info["message"] = "warning", "; ".join(problems)


def refresh(db_path=None, months: int = settings.HOLDINGS_MONTHS, holdings: bool = True,
            src: Sources | None = None) -> Report:
    src = src or Sources()
    report = Report()
    con = store.connect(db_path)
    try:
        refresh_universe(con, src, report)
        refresh_nav(con, src, report)
        refresh_market_cap(con, src, report)
        if holdings:
            refresh_holdings(con, src, report, months)
    finally:
        con.close()
    return report
