"""Everything the dashboard shows, loaded once from SQLite at startup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smt import checks, store, universe
from smt.analytics import performance as perf


def has_holdings(source) -> bool:
    """SQLite NULL comes back as None or NaN depending on the pandas version; NaN is truthy."""
    return isinstance(source, str) and bool(source)


def short_name(name: str) -> str:
    name = name.split("(")[0].strip()
    return name[:-5] if name.endswith(" Fund") else name


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
            f.loc[f["is_benchmark"] == 1, "label"] = universe.BENCHMARK_NAME
            f["category_short"] = f["category"].fillna("").str.split(" - ").str[-1].str.replace(" Fund", "")
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

    def fund_options(self, include_benchmark: bool = False) -> list[dict]:
        f = self.funds if include_benchmark else self.funds[self.funds["is_benchmark"] == 0]
        return [{"label": r.label, "value": c} for c, r in f.iterrows()]

    def holdings_codes(self) -> list[str]:
        have = set(self.holdings["scheme_code"])
        return [c for c in self.funds.index if c in have]

    def scorecard(self) -> pd.DataFrame:
        rows = []
        for code, f in self.funds.iterrows():
            nav = self.nav[code].dropna()
            tr = perf.trailing_returns(nav)
            rm = perf.risk_metrics(nav, None if f.is_benchmark else self.bench)
            rows.append({
                "code": code, "Fund": f.label, "AMC": f.amc, "Category": f.category_short,
                "1Y": tr["1Y"], "3Y": tr["3Y"], "5Y": tr["5Y"],
                "Volatility": rm.get("volatility"), "Sharpe": rm.get("sharpe"),
                "Max drawdown": rm.get("max_drawdown"), "Beta": rm.get("beta"), "Alpha": rm.get("alpha"),
                "Holdings": "yes" if has_holdings(f.holdings_source) else "",
                "NAV date": nav.index[-1].date().isoformat() if len(nav) else "",
            })
        df = pd.DataFrame(rows)
        active = df["code"] != universe.BENCHMARK_CODE
        df.loc[active, "3Y rank"] = perf.category_rank(df.loc[active, "3Y"], df.loc[active, "Category"])
        df["3Y rank"] = df["3Y rank"].fillna("")
        return df
