"""End-to-end pipeline runs with fake sources: no network."""
from __future__ import annotations

from smt import checks, pipeline, store, universe
from tests.conftest import fake_master


def test_refresh_loads_everything(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    rep = pipeline.refresh(db, months=2, src=fake_sources)
    assert rep.verified == 21 and not rep.mismatched and not rep.errors
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
    assert rep.verified == 20 and rep.mismatched and "119071" in rep.mismatched[0]
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
    assert len(store.nav_wide(con).columns) == 20
    assert store.refresh_log(con).set_index("job").loc["nav", "status"] == "warning"
    con.close()


def test_market_cap_falls_back_to_isin_prefix_after_a_split(tmp_path, fake_sources, market_cap_df):
    # HDFC Bank gets a new ISIN serial after a split; the AMFI list still has the old one.
    fake_sources.market_cap = lambda: ("31Dec2025", market_cap_df.replace({"INE040A01034": "INE040A01026"}))
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=1, src=fake_sources)
    con = store.connect(db)
    h = store.holdings(con)
    assert (h.loc[h["isin"] == "INE040A01034", "cap_class"] == "Large Cap").all()
    con.close()


def test_source_outage_keeps_last_good_market_cap_list(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=1, src=fake_sources)

    def down():
        raise RuntimeError("AMFI website unreachable: ConnectionError")

    fake_sources.market_cap = down
    rep = pipeline.refresh(db, months=1, src=fake_sources)
    assert any("market_cap" in e for e in rep.errors)
    con = store.connect(db)
    assert store.market_cap_label(con) == "31Dec2025"            # previous list kept
    assert store.refresh_log(con).iloc[0:5].set_index("job").loc["market_cap", "status"] == "warning"
    con.close()


def test_checks_pass_on_clean_data(tmp_path, fake_sources):
    db = tmp_path / "smt.db"
    pipeline.refresh(db, months=2, src=fake_sources)
    con = store.connect(db)
    res = checks.run(con)
    assert not (res["status"] == "fail").any(), res.to_string()
    con.close()
