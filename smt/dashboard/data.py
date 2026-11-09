"""Everything the dashboard shows, loaded once from SQLite at startup."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smt import checks, store, universe
from smt.analytics import evidence as evidence_mod
from smt.analytics import performance as perf


def has_holdings(source) -> bool:
    """SQLite NULL comes back as None or NaN depending on the pandas version; NaN is truthy."""
    return isinstance(source, str) and bool(source)


def short_name(name: str) -> str:
    name = name.split("(")[0].strip()
    return name[:-5] if name.endswith(" Fund") else name


def short_category(category: object) -> str:
    """'Equity Scheme - Mid Cap Fund' -> 'Mid Cap'. AMFI files ELSS funds under two spellings."""
    text = str(category or "").split(" - ")[-1].replace(" Funds", "").replace(" Fund", "").strip()
    return "ELSS" if text.upper().startswith("ELSS") else text


@dataclass
class Data:
    funds: pd.DataFrame          # verified funds, indexed by scheme_code
    nav: pd.DataFrame            # date x scheme_code
    holdings: pd.DataFrame
    checks: pd.DataFrame
    log: pd.DataFrame
    files: pd.DataFrame
    market_cap_label: str | None

    @classmethod
    def load(cls, db_path: Path | str | None = None) -> Data:
        con = store.connect(db_path)
        try:
            f = store.funds(con).set_index("scheme_code")
            f["label"] = f["scheme_name"].map(short_name)
            for code in f.index[f["is_benchmark"] == 1]:
                f.loc[code, "label"] = universe.by_code(code).label
            f["category_short"] = f["category"].map(short_category)
            return cls(f, store.nav_wide(con), store.holdings(con), checks.run(con),
                       store.refresh_log(con), store.disclosure_files(con), store.market_cap_label(con))
        finally:
            con.close()

    # ------------------------------------------------------------------ helpers
    @property
    def empty(self) -> bool:
        return self.nav.empty

    @property
    def bench(self) -> pd.Series:
        return self.nav[universe.BENCHMARK_CODE]

    def label(self, code: str) -> str:
        return self.funds.at[code, "label"] if code in self.funds.index else code

    def index_for(self, code: str) -> str:
        """The index fund this fund is fairly compared with (Nifty 50 if that index isn't loaded)."""
        index = universe.category_index(self.funds.at[code, "category"])
        return index if index in self.nav else universe.BENCHMARK_CODE

    @property
    def active_codes(self) -> list[str]:
        return list(self.funds.index[self.funds["is_benchmark"] == 0])

    def __post_init__(self) -> None:
        self._evidence: dict | None = None
        self._evidence_lock = threading.Lock()

    @property
    def evidence(self) -> dict:
        """Computed once. The event study walks every fund and month, so it takes a while.

        The lock means a page that asks while `warm()` is still computing waits for that
        result instead of starting a second computation.
        """
        with self._evidence_lock:
            if self._evidence is None:
                self._evidence = evidence_mod.compute(self.funds, self.nav, self.holdings)
            return self._evidence

    def warm(self) -> None:
        """Start computing the evidence in the background, so the tab is ready when someone opens it."""
        threading.Thread(target=lambda: self.evidence, name="smt-evidence", daemon=True).start()

    def fund_options(self, include_benchmark: bool = False) -> list[dict]:
        f = self.funds if include_benchmark else self.funds[self.funds["is_benchmark"] == 0]
        return [{"label": r.label, "value": c} for c, r in f.iterrows()]

    def holdings_codes(self) -> list[str]:
        have = set(self.holdings["scheme_code"])
        return [c for c in self.funds.index if c in have]

    def scorecard(self) -> pd.DataFrame:
        rows = []
        for code, f in self.funds.iterrows():
            if code == universe.CASH_PROXY:
                continue
            nav = self.nav[code].dropna()
            tr = perf.trailing_returns(nav)
            index_code = None if f.is_benchmark else self.index_for(code)
            rm = perf.risk_metrics(nav, None if index_code is None else self.nav[index_code])
            hit = {} if index_code is None else perf.rolling_vs_index(nav, self.nav[index_code].dropna())
            rows.append({
                "code": code, "Fund": f.label, "AMC": f.amc, "Category": f.category_short,
                "1Y": tr["1Y"], "3Y": tr["3Y"], "5Y": tr["5Y"],
                "Compared with": "" if index_code is None else self.label(index_code).split(" (")[0],
                "3Y vs index": rm.get("active_return"), "Beat index (3Y windows)": hit.get("hit_rate"),
                "Alpha": rm.get("alpha"), "Beta": rm.get("beta"), "Info ratio": rm.get("information_ratio"),
                "Volatility": rm.get("volatility"), "Sharpe": rm.get("sharpe"), "Sortino": rm.get("sortino"),
                "Max drawdown": rm.get("max_drawdown"),
                "Holdings": "yes" if has_holdings(f.holdings_source) else "",
            })
        df = pd.DataFrame(rows)
        active = df["Compared with"] != ""
        df.loc[active, "3Y rank"] = perf.category_rank(df.loc[active, "3Y"], df.loc[active, "Category"])
        df["3Y rank"] = df["3Y rank"].fillna("")
        return df
