"""Does following the money work? An event study on the funds' own disclosures.

The test is fixed in advance so it can't be tuned to the answer:

  * Each month-end, a stock counts as bought if more tracked funds actively added to it
    than cut it, and sold if the reverse.
  * "Actively" removes new money. A fund receiving inflows buys a bit of everything, so
    each fund's median share-count growth is taken out first. Only a position that grew
    or shrank more than that fund's norm is a decision. Corporate actions are excluded.
  * Baskets are equal-weighted and measured over the following calendar month, using the
    price per share implied by the disclosures themselves.
  * `skip=1` waits one further month. Portfolios are published about 10 days after
    month-end, so the first month's return is not fully available to a follower.

Limits, reported alongside the result: prices exist only while some tracked fund holds
the stock, so names every fund sold out of drop from the sold basket; and the funds come
from two AMCs, so their decisions are not independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smt.analytics import holdings as H

ACTIVE_THRESHOLD = 0.01       # ignore rounding-sized changes relative to the fund's norm
MAX_MONTHLY_MOVE = 2.5        # a bigger factor after adjustment is a data artefact, not a return


def consecutive_months(h: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ms = sorted(pd.to_datetime(h["as_of"].unique()))
    return [(a, b) for a, b in zip(ms, ms[1:], strict=False) if (b - a).days <= 35]


def panels(h: pd.DataFrame, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly stock returns, active trades) from one pass over consecutive month-ends.

    Returns: index month-end, columns ISIN at that date, adjusted for bonuses and splits.
    Trades: one row per fund-stock-month that was a decision: as_of, scheme_code, isin, direction (+1 / -1).
    """
    by_month = {m: g for m, g in h[h["asset_class"].isin(H.STOCKS)].groupby("as_of")}
    returns, trades = {}, []
    for prev, cur in consecutive_months(h):
        pair = pd.concat([by_month[prev], by_month[cur]])
        acts = H.corporate_actions(pair, prev, cur)
        spun = H.spin_offs(pair, prev, cur, acts)

        p0 = H.implied_prices(H._stock_rows(pair, prev))
        p1 = H.implied_prices(H._stock_rows(pair, cur))
        p0 = p0.rename(index=dict(zip(acts["isin_prev"], acts["isin"], strict=True)))
        p0 = p0[~p0.index.duplicated()]
        factor = pd.Series(dict(zip(acts["isin"], acts["factor"], strict=True)), dtype=float)
        common = p1.index.intersection(p0.index)
        growth = p1[common] / p0[common] * factor.reindex(common).fillna(1.0)
        growth = growth[(growth < MAX_MONTHLY_MOVE) & (growth > 1 / MAX_MONTHLY_MOVE)]
        # A demerger takes value out of the parent's price; that fall is not a return.
        returns[cur] = growth.drop(index=[i for i in spun["parent_isin"] if i in growth.index]) - 1

        # Every fund holding portfolios in both months, compared in one pass.
        old_rows, cur_rows = by_month[prev], by_month[cur]
        present = set(old_rows["scheme_code"]) & set(cur_rows["scheme_code"]) & set(codes)
        if not present:
            continue
        c = H.compare(cur_rows[cur_rows["scheme_code"].isin(present)],
                      old_rows[old_rows["scheme_code"].isin(present)], acts, spun)
        c = c[~c["status"].isin(H.NOT_A_TRADE)]
        ratio = c["quantity"] / c["quantity_prev"].replace(0, np.nan)
        held = (c["quantity_prev"] > 0) & (c["quantity"] > 0)
        # Each fund's typical share-count growth that month: new money spread across the portfolio.
        norm = ratio.where(held).groupby(c["scheme_code"]).transform("median").fillna(1.0)
        relative = ratio / norm - 1
        direction = pd.Series(0, index=c.index)
        direction[relative > ACTIVE_THRESHOLD] = 1
        direction[relative < -ACTIVE_THRESHOLD] = -1
        direction[c["status"] == "New"] = 1
        direction[c["status"] == "Exited"] = -1
        chosen = c.loc[direction != 0, ["isin", "scheme_code", "status"]].assign(
            direction=direction[direction != 0], as_of=cur)
        trades.append(chosen[TRADE_COLUMNS])
    ret = pd.DataFrame(returns).T.sort_index() if returns else pd.DataFrame()
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=TRADE_COLUMNS)
    return ret, tr


TRADE_COLUMNS = ["isin", "direction", "as_of", "scheme_code", "status"]


def _t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def follow_the_money(h: pd.DataFrame, codes: list[str], skip: int = 0, returns: pd.DataFrame | None = None,
                     trades: pd.DataFrame | None = None, min_net: int = 1) -> dict:
    """Monthly returns of the bought and sold baskets, and the evidence for a difference.

    `min_net` is how many more funds must be on one side than the other for a stock to count.
    """
    if returns is None or trades is None:
        returns, trades = panels(h, codes)
    if returns.empty or trades.empty:
        return {"monthly": pd.DataFrame(), "summary": {}}
    net = trades.groupby(["as_of", "isin"])["direction"].sum()
    dates = list(returns.index)
    rows = []
    for t in sorted(net.index.get_level_values("as_of").unique()):
        if t not in dates or dates.index(t) + 1 + skip >= len(dates):
            continue
        future = returns.loc[dates[dates.index(t) + 1 + skip]]
        n = net.loc[t]
        bought, sold = n[n >= min_net].index, n[n <= -min_net].index
        held = H._stock_rows(h, t)["isin"].unique()
        rb, rs, ra = future.reindex(bought).dropna(), future.reindex(sold).dropna(), future.reindex(held).dropna()
        if len(rb) < 5 or len(rs) < 5:
            continue
        rows.append({"month": t, "n_bought": len(rb), "n_sold": len(rs), "bought": rb.mean(), "sold": rs.mean(),
                     "held": ra.mean(), "coverage_sold": len(rs) / max(len(sold), 1)})
    m = pd.DataFrame(rows)
    if m.empty:
        return {"monthly": m, "summary": {}}
    m["spread"] = m["bought"] - m["sold"]
    m["excess"] = m["bought"] - m["held"]
    summary = {
        "months": len(m), "skip": skip,
        "bought": float(m["bought"].mean()), "sold": float(m["sold"].mean()), "held": float(m["held"].mean()),
        "spread": float(m["spread"].mean()), "spread_t": _t_stat(m["spread"]),
        "spread_hit_rate": float((m["spread"] > 0).mean()),
        "excess": float(m["excess"].mean()), "excess_t": _t_stat(m["excess"]),
        "coverage_sold": float(m["coverage_sold"].mean()),
        "avg_bought": float(m["n_bought"].mean()), "avg_sold": float(m["n_sold"].mean()),
        "first": m["month"].min(), "last": m["month"].max(),
    }
    return {"monthly": m, "summary": summary}


def _row(name: str, spread: pd.Series, note: str = "") -> dict:
    spread = spread.dropna()
    se = spread.std() / np.sqrt(len(spread)) if len(spread) > 2 else float("nan")
    mean = float(spread.mean()) if len(spread) else float("nan")
    return {"variant": name, "months": len(spread), "spread": mean, "t": _t_stat(spread),
            "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se, "note": note}


def robustness(h: pd.DataFrame, funds: pd.DataFrame, returns: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """The main test repeated on other reasonable cuts of the same data.

    A real effect should keep its sign across them. One that appears in a single cut and
    not the others is more likely noise found by looking. `funds` needs an `amc` column.
    """
    codes = sorted(set(trades["scheme_code"]))
    main = follow_the_money(h, codes, 0, returns, trades)
    if not main["summary"]:
        return pd.DataFrame()
    m = main["monthly"].sort_values("month")
    rows = [_row("All active trades (main test)", m["spread"])]
    later = follow_the_money(h, codes, 1, returns, trades)
    if later["summary"]:
        rows.append(_row("Waiting one more month", later["monthly"]["spread"],
                         "what a follower could act on after publication"))
    agree = follow_the_money(h, codes, 0, returns, trades, min_net=2)
    if agree["summary"]:
        rows.append(_row("Two or more funds on the same side", agree["monthly"]["spread"]))
    strong = trades[trades["status"].isin(["New", "Exited"])]
    conviction = follow_the_money(h, codes, 0, returns, strong)
    if conviction["summary"]:
        rows.append(_row("New positions and full exits only", conviction["monthly"]["spread"]))
    for amc, group in funds[funds.index.isin(codes)].groupby("amc"):
        if len(group) < 2:
            continue
        sub = follow_the_money(h, list(group.index), 0, returns, trades[trades["scheme_code"].isin(group.index)])
        if sub["summary"]:
            rows.append(_row(f"{amc} funds only ({len(group)})", sub["monthly"]["spread"]))
    half = len(m) // 2
    rows.append(_row("First half of the period", m["spread"].iloc[:half],
                     f"{m['month'].iloc[0]:%b %Y} to {m['month'].iloc[half - 1]:%b %Y}"))
    rows.append(_row("Second half of the period", m["spread"].iloc[half:],
                     f"{m['month'].iloc[half]:%b %Y} to {m['month'].iloc[-1]:%b %Y}"))
    return pd.DataFrame(rows)
