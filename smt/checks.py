"""Data-quality checks, shown on the Data status tab and by `python -m smt check`."""
from __future__ import annotations

import sqlite3

import pandas as pd

from smt import store


def run(con: sqlite3.Connection) -> pd.DataFrame:
    rows: list[dict] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        rows.append({"check": name, "status": "pass" if ok else ("warn" if ok is None else "fail"),
                     "detail": detail})

    allf = store.funds(con, verified_only=False)
    bad = allf[allf["verified"] == 0]
    add("Scheme codes match AMFI names", bad.empty and not allf.empty,
        f"{len(allf) - len(bad)}/{len(allf)} verified" +
        ("" if bad.empty else "; mismatched: " + ", ".join(bad["scheme_code"])))

    last = pd.read_sql_query("SELECT scheme_code, MAX(date) AS last, COUNT(*) AS n FROM nav GROUP BY scheme_code",
                             con, parse_dates=["last"])
    if last.empty:
        add("NAV history loaded", False, "no NAV rows")
    else:
        newest = last["last"].max()
        stale = last[last["last"] < newest - pd.Timedelta(days=7)]
        add("NAV is current for every fund", stale.empty,
            f"latest NAV {newest.date()}" + ("" if stale.empty else "; stale: " + ", ".join(stale["scheme_code"])))
        short = last[last["n"] < 250]
        add("At least one year of NAV history", None if not short.empty else True,
            "all funds" if short.empty else "short history (new fund): " + ", ".join(short["scheme_code"]))

    h = store.holdings(con)
    hf = allf[allf["holdings_source"].notna() & (allf["verified"] == 1)]
    if h.empty:
        add("Holdings loaded", False if not hf.empty else None, "no holdings rows")
    else:
        latest = h["as_of"].max()
        missing = sorted(set(hf["scheme_code"]) - set(h.loc[h["as_of"] == latest, "scheme_code"]))
        add("Latest month present for every holdings fund", not missing,
            f"latest month {latest.date()}" + ("" if not missing else "; missing: " + ", ".join(missing)))
        totals = h.groupby(["scheme_code", "as_of"])["pct_nav"].sum()
        broken = totals[(totals < 60) | (totals > 101)]
        low = totals[(totals >= 60) & (totals < 85)]
        add("Securities with an ISIN add up to 60-101% of NAV", broken.empty,
            f"{len(totals)} fund-months checked" + ("" if broken.empty else
                f"; outside range: {', '.join(f'{c} {d.date()}' for c, d in broken.index)}"))
        add("Cash, TREPS and derivatives under 15% of NAV", True if low.empty else None,
            "all funds" if low.empty else "higher cash/derivatives in: " +
            ", ".join(sorted({c for c, _ in low.index})))
        eq = h[h["asset_class"] == "Equity"]
        if not eq.empty:
            covered = eq.loc[eq["cap_class"].notna(), "pct_nav"].sum() / eq["pct_nav"].sum() * 100
            add("Indian stocks with an AMFI market-cap class", covered >= 95 if covered < 100 else True,
                f"{covered:.1f}% of equity weight classified ({store.market_cap_label(con) or 'no list'})")
    return pd.DataFrame(rows)
