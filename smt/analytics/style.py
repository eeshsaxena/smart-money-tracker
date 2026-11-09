"""Returns-based style analysis (Sharpe, 1992): infer a fund's market-cap mix from its NAV alone.

A fund's weekly returns are regressed on large, mid and small cap index funds plus a
liquid fund, with weights that are non-negative and add up to 100%. The weights are the
mix of segments that best reproduces the fund's returns.

This matters because most AMCs here publish holdings in a form that can't be scripted,
so NAV is all there is. The estimate is worth using only if it can be checked, so
`validate` compares it with the real portfolios of the funds that do have holdings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from smt.analytics import holdings as H

MIN_WEEKS = 26
CAP_CLASSES = ["Large Cap", "Mid Cap", "Small Cap"]


@dataclass
class StyleEstimate:
    weights: pd.Series     # factor name -> weight in %
    r_squared: float
    weeks: int


def estimate(nav: pd.Series, factors: pd.DataFrame, weeks: int = 52) -> StyleEstimate | None:
    """Best-fitting mix of `factors` (a NAV table, one column per segment) over the last `weeks` weeks."""
    both = pd.concat([nav.rename("__fund__"), factors], axis=1, join="inner", sort=True).dropna()
    r = both.resample("W-FRI").last().pct_change().dropna().tail(weeks) * 100
    if len(r) < MIN_WEEKS:
        return None
    y, x = r["__fund__"].to_numpy(), r[list(factors.columns)].to_numpy()
    n = x.shape[1]
    res = minimize(lambda w: float(((y - x @ w) ** 2).sum()), np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        return None
    total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - x @ res.x) ** 2).sum() / total if total > 0 else float("nan")
    return StyleEstimate(pd.Series(res.x * 100, index=factors.columns), float(r2), len(r))


def actual_mix(h: pd.DataFrame, code: str, months: int = 12) -> pd.Series:
    """Average disclosed market-cap mix (% of NAV) over the last `months` portfolios."""
    ms = H.months(h, code)[:months]
    return pd.DataFrame([H.cap_mix(H.snapshot(h, code, m)) for m in ms]).fillna(0.0).mean()


def validate(nav: pd.DataFrame, factors: pd.DataFrame, h: pd.DataFrame, codes: list[str],
             weeks: int = 52) -> pd.DataFrame:
    """Estimated vs disclosed mix for every fund that has both, over the same window. One row per fund."""
    rows = []
    for code in codes:
        est = estimate(nav[code].dropna(), factors, weeks)
        if est is None:
            continue
        act = actual_mix(h, code, months=round(weeks / 52 * 12))
        row = {"scheme_code": code, "r_squared": est.r_squared, "weeks": est.weeks}
        for c in CAP_CLASSES:
            row[f"est {c}"], row[f"actual {c}"] = float(est.weights.get(c, 0.0)), float(act.get(c, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def validation_errors(table: pd.DataFrame) -> dict[str, float]:
    """Mean absolute error of the estimate in percentage points, by segment and overall."""
    if table.empty:
        return {}
    errs = {c: float((table[f"est {c}"] - table[f"actual {c}"]).abs().mean()) for c in CAP_CLASSES}
    errs["overall"] = float(np.mean(list(errs.values())))
    return errs
