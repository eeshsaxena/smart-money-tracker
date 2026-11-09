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
    if not res.success:              # a failed solve returns an arbitrary point, not an answer
        raise ValueError(f"optimiser did not converge: {res.message}")
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


def _annualise(weekly: pd.Series, rf: float) -> dict[str, float]:
    growth = (1 + weekly).cumprod()
    years = len(weekly) / 52
    ret = float(growth.iloc[-1] ** (1 / years) - 1)
    vol = float(weekly.std() * np.sqrt(52))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol > 0 else float("nan"),
            "max_drawdown": float((growth / growth.cummax() - 1).min())}


def walk_forward(navs: pd.DataFrame, train_years: float = 3.0, rebalance_months: int = 6,
                 rf: float = settings.RISK_FREE_RATE) -> dict:
    """Would the optimiser's weights have worked on data it had not seen?

    At each rebalance date the weights are fitted only on the preceding `train_years`,
    then held for the next period. Equal weight is the benchmark: an optimiser that
    can't beat it out of sample is describing the past, not finding anything.
    """
    weekly = navs.dropna(how="any").resample("W-FRI").last().pct_change().dropna()
    first = weekly.index[0] + pd.DateOffset(days=round(train_years * 365.25))
    dates = pd.date_range(first, weekly.index[-1], freq=f"{rebalance_months}MS")
    if weekly.shape[1] < 2 or len(dates) < 2:
        raise ValueError("need at least 2 funds and more history than the training window")
    n = weekly.shape[1]
    out: dict[str, list[pd.Series]] = {"Max Sharpe": [], "Min volatility": [], "Equal weight": []}
    previous = np.full(n, 1 / n)
    turnover, failures = [], 0
    for start, end in zip(dates, [*dates[1:], weekly.index[-1] + pd.Timedelta(days=1)], strict=True):
        train = weekly.loc[start - pd.DateOffset(days=round(train_years * 365.25)):start - pd.Timedelta(days=1)]
        test = weekly.loc[start:end - pd.Timedelta(days=1)]
        if test.empty:
            continue
        mu, cov = _stats(train)

        def vol(w, cov=cov):
            return float(np.sqrt(w @ cov @ w))

        equal = np.full(n, 1 / n)
        try:
            w_min = _solve(vol, n)
            w_sharpe = _solve(lambda w, mu=mu: -((w @ mu - rf) / vol(w)), n)
        except ValueError:
            w_min = w_sharpe = equal
            failures += 1
        turnover.append(float(np.abs(w_sharpe - previous).sum() / 2))
        previous = w_sharpe
        for name, w in (("Max Sharpe", w_sharpe), ("Min volatility", w_min), ("Equal weight", equal)):
            out[name].append(test @ w)
    paths = pd.DataFrame({k: pd.concat(v) for k, v in out.items()})
    summary = pd.DataFrame({k: _annualise(paths[k], rf) for k in paths}).T
    return {"weekly": paths, "summary": summary, "periods": len(turnover),
            "turnover": float(np.mean(turnover[1:])) if len(turnover) > 1 else 0.0, "failed_solves": failures,
            "funds": list(weekly.columns), "start": paths.index[0], "end": paths.index[-1]}
