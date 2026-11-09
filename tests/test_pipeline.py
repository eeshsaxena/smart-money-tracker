"""End-to-end pipeline runs with fake sources: no network."""
from __future__ import annotations

from datetime import date

import pandas as pd

from smt import checks, pipeline, store, universe
from smt.sources import amfi
from tests.conftest import FakeConnector, disclosure_titles, fake_master


def test_refresh_loads_everything(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    rep = pipeline.refresh(db, months=2, src=fake_sources)
    assert rep.verified == len(universe.FUNDS) and not rep.mismatched and not rep.errors
    con = store.connect(db)
    h = store.holdings(con)
    # every holdings fund, including DSP Midcap (spelt 'Mid Cap' in the file); untracked schemes ignored
    assert set(h["scheme_code"]) == {f.scheme_code for f in universe.with_holdings()}
    assert sorted(h["as_of"].dt.month.unique()) == [8, 9]          # only the 2 newest months
    assert h.loc[h["isin"] == "INE522F01014", "cap_class"].iloc[0] == "Mid Cap"   # AMFI list joined
    log = store.refresh_log(con)
    assert {"universe", "nav", "market_cap", "holdings:ppfas", "holdings:dsp"} <= set(log["job"])
    assert (log["status"] == "ok").all()
    con.close()


def test_wrong_scheme_code_is_caught_and_excluded(tmp_path, fake_sources):
    fake_sources.scheme_master = lambda: fake_master(mismatch="119071")
    db = tmp_path / "smt.db"
    rep = pipeline.refresh(db, months=1, src=fake_sources)
    assert rep.verified == len(universe.FUNDS) - 1 and rep.mismatched and "119071" in rep.mismatched[0]
    con = store.connect(db)
    assert "119071" not in set(store.funds(con)["scheme_code"])           # excluded from analytics
    assert "119071" not in set(store.nav_wide(con).columns)
    result = checks.run(con).set_index("check")
    assert result.loc["Scheme codes match AMFI names", "status"] == "fail"
    con.close()


def test_downloads_are_cached_and_reruns_are_idempotent(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=2, src=fake_sources)
    first = {n: c.downloads for n, c in fake_sources.connectors.items()}
    con = store.connect(db)
    rows_before = len(store.holdings(con))
    pipeline.refresh(db, months=2, src=fake_sources)
    assert {n: c.downloads for n, c in fake_sources.connectors.items()} == first   # served from raw_dir
    assert len(store.holdings(con)) == rows_before                                  # no duplicates
    assert any((fake_sources.raw_dir / "dsp").iterdir())
    con.close()


def test_one_failing_fund_does_not_stop_the_rest(tmp_path, fake_sources):
    good = fake_sources.nav_history

    def flaky(code):
        if code == "118955":
            raise ConnectionError("timeout")
        return good(code)

    fake_sources.nav_history = flaky
    db = tmp_path / "smt.db"
    rep = pipeline.refresh(db, months=1, src=fake_sources, holdings=False)
    assert any("118955" in e for e in rep.errors)
    con = store.connect(db)
    assert len(store.nav_wide(con).columns) == len(universe.FUNDS) - 1
    assert store.refresh_log(con).set_index("job").loc["nav", "status"] == "warning"
    con.close()


def test_market_cap_falls_back_to_isin_prefix_after_a_split(tmp_path, fake_sources, market_cap_df):
    # HDFC Bank gets a new ISIN serial after a split; the AMFI list still has the old one.
    fake_sources.market_cap = lambda label: market_cap_df.replace({"INE040A01034": "INE040A01026"})
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=1, src=fake_sources)
    con = store.connect(db)
    h = store.holdings(con)
    assert (h.loc[h["isin"] == "INE040A01034", "cap_class"] == "Large Cap").all()
    con.close()


def test_source_outage_keeps_last_good_market_cap_list(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=1, src=fake_sources)
    con = store.connect(db)
    newest, older = store.market_cap_labels(con)[:2]
    with con:
        con.execute("DELETE FROM market_cap_lists WHERE label = ?", (newest,))    # a new list is now due

    def down(label):
        raise RuntimeError("AMFI website unreachable: ConnectionError")

    fake_sources.market_cap = down
    rep = pipeline.refresh(db, months=1, src=fake_sources)
    assert any("market_cap" in e for e in rep.errors)
    assert store.market_cap_label(con) == older                   # what we had is kept
    assert store.refresh_log(con).iloc[0:5].set_index("job").loc["market_cap", "status"] == "warning"
    con.close()


def test_market_cap_list_not_yet_published_is_not_an_error(tmp_path, fake_sources, market_cap_df):
    newest, older = amfi.classification_labels(count=2)

    def published(label):
        if label == newest:
            raise LookupError(label)
        return market_cap_df

    fake_sources.market_cap = published
    db = tmp_path / "smt.db"
    rep = pipeline.refresh(db, months=1, src=fake_sources)
    assert not rep.errors
    con = store.connect(db)
    assert store.market_cap_label(con) == older
    con.close()


def test_stocks_are_classed_by_the_list_in_force_at_the_portfolio_date(tmp_path, fake_sources, market_cap_df):
    """Coal India is mid cap in lists up to Dec 2025 and large cap from Jun 2026, so Sep 2026 sees large."""
    def by_period(label):
        cap = "Large Cap" if label == "30Jun2026" else "Mid Cap"
        return market_cap_df.assign(cap_class=[cap if i == "INE522F01014" else c for i, c in
                                               zip(market_cap_df["isin"], market_cap_df["cap_class"], strict=True)])

    fake_sources.market_cap = by_period
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=6, src=fake_sources)
    con = store.connect(db)
    assert store.list_in_force(pd.Timestamp("2026-09-30"), store.market_cap_labels(con)) == "30Jun2026"
    assert store.list_in_force(pd.Timestamp("2026-06-30"), store.market_cap_labels(con)) == "31Dec2025"
    h = store.holdings(con)
    coal = h[h["isin"] == "INE522F01014"]
    assert set(coal["as_of"].dt.month) == {9} and (coal["cap_class"] == "Large Cap").all()
    con.close()


def test_portfolios_filed_under_a_former_name_are_kept(tmp_path, fake_sources):
    """DSP renamed Top 100 Equity to Large Cap in 2025; older files still use the old name."""
    titles = [t.replace("DSP Large Cap Fund", "DSP Top 100 Equity Fund") for t in disclosure_titles("dsp")]
    fake_sources.connectors["dsp"] = FakeConnector("dsp", titles)
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=2, src=fake_sources)
    con = store.connect(db)
    assert "119250" in set(store.holdings(con)["scheme_code"])
    con.close()


def test_undatable_portfolio_file_is_reported_not_silently_dropped(tmp_path, fake_sources):
    connector = fake_sources.connectors["dsp"]
    connector.unreadable = ("monthend-portfolio-final.zip",)
    rep = pipeline.refresh(tmp_path / "smt.db", months=1, src=fake_sources)
    assert any("monthend-portfolio-final.zip" in e for e in rep.errors)


def _seeded(tmp_path, fake_sources, months=2):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=months, src=fake_sources)
    return store.connect(db)


def test_checks_pass_on_clean_data(tmp_path, fake_sources):
    con = _seeded(tmp_path, fake_sources)
    res = checks.run(con, today=date(2026, 11, 2))
    assert not (res["status"] == "fail").any(), res.to_string()
    con.close()


def test_stale_data_is_measured_against_today_not_against_other_funds(tmp_path, fake_sources):
    """Every fund stopping on the same day must still fail: they would all agree with each other."""
    con = _seeded(tmp_path, fake_sources)
    res = checks.run(con, today=date(2026, 12, 31)).set_index("check")
    assert res.loc["NAV is current for every fund", "status"] == "fail"
    assert res.loc["Holdings are recent", "status"] == "warn"
    con.close()


def test_market_cap_list_going_stale_is_flagged(tmp_path, fake_sources):
    con = _seeded(tmp_path, fake_sources)
    newest = store.market_cap_labels(con)[0]
    year = int(newest[-4:]) + 1
    res = checks.run(con, today=date(year, 12, 31)).set_index("check")
    assert res.loc["Market-cap list is the latest half-year", "status"] == "warn"
    con.close()


def test_a_month_missing_from_the_middle_of_a_history_fails(tmp_path, fake_sources):
    """The fake source has June, August and September, so July is a hole."""
    con = _seeded(tmp_path, fake_sources, months=3)
    res = checks.run(con, today=date(2026, 11, 2)).set_index("check")
    row = res.loc["No month missing inside each fund's holdings history"]
    assert row["status"] == "fail" and "2026-07" in row["detail"]
    con.close()
