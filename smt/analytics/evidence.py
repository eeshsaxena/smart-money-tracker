"""The project's four "does it actually hold up?" questions, computed from the same data the app shows.

Shared by the dashboard's Evidence tab and by `python -m smt report`, so the written
findings can never disagree with the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smt import universe
from smt.analytics import flows, style
from smt.analytics import performance as perf
from smt.analytics import portfolio as pf

MIN_WALK_FORWARD_HISTORY = pd.Timestamp("2014-01-01")   # funds with a decade of Direct Plan history


def funds_vs_index(funds: pd.DataFrame, nav: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each active fund against the index matching its category, and against the Nifty 50."""
    rows = []
    nifty = nav[universe.BENCHMARK_CODE].dropna() if universe.BENCHMARK_CODE in nav else None
    for code, f in funds[funds["is_benchmark"] == 0].iterrows():
        index_code = universe.category_index(f["category"])
        if code not in nav or index_code not in nav:
            continue
        s, b = nav[code].dropna(), nav[index_code].dropna()
        m3, m5 = perf.risk_metrics(s, b, years=3), perf.risk_metrics(s, b, years=5)
        vs_nifty = perf.risk_metrics(s, nifty, years=3) if nifty is not None else {}
        rolling = perf.rolling_vs_index(s, b)
        rows.append({
            "scheme_code": code, "fund": f["label"], "index": universe.by_code(index_code).label,
            "return_3y": m3.get("return"), "index_3y": perf.period_return(b, 3),
            "active_3y": m3.get("active_return"), "active_5y": m5.get("active_return"),
            "alpha": m3.get("alpha"), "alpha_vs_nifty50": vs_nifty.get("alpha"), "beta": m3.get("beta"),
            "information_ratio": m3.get("information_ratio"), "tracking_error": m3.get("tracking_error"),
            "hit_rate": rolling["hit_rate"], "median_active": rolling["median_active"],
            "windows": rolling["windows"],
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t, {}
    a3, a5 = t["active_3y"].dropna(), t["active_5y"].dropna()
    return t.sort_values("active_3y", ascending=False, na_position="last").reset_index(drop=True), {
        "funds_3y": len(a3), "beat_3y": int((a3 > 0).sum()), "median_active_3y": float(a3.median()),
        "funds_5y": len(a5), "beat_5y": int((a5 > 0).sum()),
        "median_active_5y": float(a5.median()) if len(a5) else float("nan"),
        "median_alpha": float(t["alpha"].median()), "median_alpha_vs_nifty50": float(t["alpha_vs_nifty50"].median()),
        "funds_with_windows": int(t["hit_rate"].notna().sum()),
        "beat_most_windows": int((t["hit_rate"] >= 0.5).sum()),
    }


def style_factors(nav: pd.DataFrame) -> pd.DataFrame:
    have = {k: v for k, v in universe.STYLE_FACTORS.items() if v in nav}
    return nav[list(have.values())].rename(columns={v: k for k, v in have.items()})


def style_evidence(funds: pd.DataFrame, nav: pd.DataFrame, h: pd.DataFrame) -> dict:
    """Validation against disclosed portfolios, then estimates for the funds with no disclosures."""
    factors = style_factors(nav)
    held = sorted(set(h["scheme_code"])) if not h.empty else []
    table = style.validate(nav, factors, h, [c for c in held if c in nav]) if held and factors.shape[1] > 1 else (
        pd.DataFrame())
    if not table.empty:
        table.insert(1, "fund", table["scheme_code"].map(funds["label"]))
    right = 0
    if not table.empty:
        for r in table.to_dict("records"):
            top_est = max(style.CAP_CLASSES, key=lambda c: r[f"est {c}"])
            top_act = max(style.CAP_CLASSES, key=lambda c: r[f"actual {c}"])
            right += top_est == top_act
    nav_only = []
    for code, f in funds[funds["is_benchmark"] == 0].iterrows():
        if code in held or code not in nav or factors.shape[1] < 2:
            continue
        est = style.estimate(nav[code].dropna(), factors)
        if est is not None:
            nav_only.append({"fund": f["label"], **est.weights.round(1).to_dict(), "r_squared": est.r_squared})
    return {"validation": table, "errors": style.validation_errors(table), "dominant_right": right,
            "validated": len(table), "nav_only": pd.DataFrame(nav_only)}


def flow_evidence(funds: pd.DataFrame, h: pd.DataFrame) -> dict:
    if h.empty:
        return {"next_month": {"monthly": pd.DataFrame(), "summary": {}}, "after_a_month": {"summary": {}},
                "robustness": pd.DataFrame()}
    codes = [c for c in funds.index if c in set(h["scheme_code"])]
    returns, trades = flows.panels(h, codes)
    return {"next_month": flows.follow_the_money(h, codes, 0, returns, trades),
            "after_a_month": flows.follow_the_money(h, codes, 1, returns, trades),
            "robustness": flows.robustness(h, funds, returns, trades),
            "funds": len(codes), "trades": len(trades)}


def optimiser_evidence(funds: pd.DataFrame, nav: pd.DataFrame) -> dict | None:
    codes = [c for c in funds[funds["is_benchmark"] == 0].index
             if c in nav and nav[c].first_valid_index() is not None
             and nav[c].first_valid_index() < MIN_WALK_FORWARD_HISTORY]
    if len(codes) < 2:      # short test histories: use whichever funds are there
        codes = [c for c in funds[funds["is_benchmark"] == 0].index if c in nav]
    try:
        out = pf.walk_forward(nav[codes])
    except ValueError:
        return None
    d = out["weekly"]["Max Sharpe"] - out["weekly"]["Equal weight"]
    out["gap_vs_equal"] = float(d.mean() * 52)
    out["gap_t"] = float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else float("nan")
    return out


def compute(funds: pd.DataFrame, nav: pd.DataFrame, h: pd.DataFrame) -> dict:
    table, summary = funds_vs_index(funds, nav)
    return {"index_table": table, "index_summary": summary, "style": style_evidence(funds, nav, h),
            "flows": flow_evidence(funds, h), "optimiser": optimiser_evidence(funds, nav)}
