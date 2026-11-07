"""Combining funds: correlation and mean-variance optimisation (long-only, fully invested).

Weekly returns are used rather than daily: funds with foreign holdings strike
their NAV on a different clock, which makes daily correlations look lower
than they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from smt import settings


def weekly_returns(navs: pd.DataFrame, years: float = 3.0) -> pd.DataFrame:
    navs = navs.dropna(how="all")
    start = navs.index[-1] - pd.DateOffset(days=round(years * 365.25))
    weekly = navs.loc[start:].resample("W-FRI").last()
    return weekly.pct_change().dropna(how="any")


def correlation(navs: pd.DataFrame, years: float = 3.0) -> pd.DataFrame:
    return weekly_returns(navs, years).corr()


def _stats(rets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return rets.mean().to_numpy() * 52, rets.cov().to_numpy() * 52


def _solve(objective, n: int, extra: list[dict] | None = None) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + (extra or [])
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=cons, options={"maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def optimise(navs: pd.DataFrame, years: float = 3.0, rf: float = settings.RISK_FREE_RATE,
             points: int = 25) -> dict:
    """Efficient frontier plus the max-Sharpe and min-volatility portfolios."""
    rets = weekly_returns(navs, years)
    if rets.shape[1] < 2 or len(rets) < 26:
        raise ValueError("need at least 2 funds with 6 months of common history")
    mu, cov = _stats(rets)
    n = len(mu)

    def vol(w):
        return float(np.sqrt(w @ cov @ w))

    w_min = _solve(vol, n)
    w_sharpe = _solve(lambda w: -((w @ mu - rf) / vol(w)), n)

    frontier = []
    for target in np.linspace(w_min @ mu, mu.max(), points):
        w = _solve(vol, n, [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
        frontier.append({"return": float(w @ mu), "volatility": vol(w)})

    def describe(w):
        return {"weights": pd.Series(w, index=rets.columns).round(4),
                "return": float(w @ mu), "volatility": vol(w),
                "sharpe": float((w @ mu - rf) / vol(w))}

    return {
        "frontier": pd.DataFrame(frontier).drop_duplicates().sort_values("volatility"),
        "funds": pd.DataFrame({"return": mu, "volatility": np.sqrt(np.diag(cov))}, index=rets.columns),
        "max_sharpe": describe(w_sharpe),
        "min_volatility": describe(w_min),
        "weeks": len(rets),
    }
