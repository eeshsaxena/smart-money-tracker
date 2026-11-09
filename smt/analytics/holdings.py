"""Analytics on month-end portfolio holdings.

Buying and selling is measured with share *quantities*, not portfolio weights.
A weight can rise just because the stock went up; a quantity only rises when
the fund bought more, unless a corporate action created the shares. Bonus issues
and splits multiply the share count (a split also changes the ISIN), and a demerger
hands holders shares in a new company. None of those is a decision to buy, so they
are detected first (`corporate_actions`, `spin_offs`) and removed from the flows.
"""
from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pandas as pd

EQUITY = "Equity"
STOCKS = ("Equity", "Foreign Equity")     # directional stock holdings
# Share-count multiples that bonus issues and splits actually produce (1:2 bonus = 1.5x, 1:1 = 2x, ...).
CLEAN_RATIOS = (1.1, 1.2, 1.25, 4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0)
RATIO_TOLERANCE = 0.001                   # a fund that didn't trade lands this close to the multiple


def months(h: pd.DataFrame, code: str) -> list[pd.Timestamp]:
    return sorted(h.loc[h["scheme_code"] == code, "as_of"].unique(), reverse=True)


def snapshot(h: pd.DataFrame, code: str, as_of=None) -> pd.DataFrame:
    sub = h[h["scheme_code"] == code]
    if sub.empty:
        return sub
    as_of = as_of if as_of is not None else sub["as_of"].max()
    return sub[sub["as_of"] == as_of]


def asset_mix(snap: pd.DataFrame) -> pd.Series:
    mix = snap.groupby("asset_class")["pct_nav"].sum().sort_values(ascending=False)
    other = 100 - mix.sum()
    if other > 0.05:
        mix["Cash, TREPS & receivables"] = other
    return mix


def sector_mix(snap: pd.DataFrame) -> pd.Series:
    s = snap[snap["asset_class"].isin(STOCKS)]
    return s.groupby(s["industry"].replace("", "Unclassified"))["pct_nav"].sum().sort_values(ascending=False)


def cap_mix(snap: pd.DataFrame) -> pd.Series:
    """% of NAV in Large / Mid / Small cap Indian stocks (AMFI list), plus foreign stocks."""
    eq = snap[snap["asset_class"] == EQUITY]
    out = eq.groupby(eq["cap_class"].fillna("Unclassified"))["pct_nav"].sum()
    foreign = snap.loc[snap["asset_class"] == "Foreign Equity", "pct_nav"].sum()
    if foreign:
        out["Foreign"] = foreign
    order = ["Large Cap", "Mid Cap", "Small Cap", "Foreign", "Unclassified"]
    return out.reindex([o for o in order if o in out.index])


def concentration(snap: pd.DataFrame) -> dict[str, float]:
    s = snap[snap["asset_class"].isin(STOCKS)]["pct_nav"].sort_values(ascending=False)
    w = s / s.sum() if s.sum() else s
    return {
        "stocks": int(len(s)),
        "top10_pct": float(s.head(10).sum()),
        # Herfindahl index of stock weights; 1/HHI = the "effective" number of equal-weight stocks
        "effective_stocks": float(1 / (w ** 2).sum()) if len(w) else 0.0,
    }


# --------------------------------------------------------------------------- #
# Corporate actions: share-count changes nobody decided to make
# --------------------------------------------------------------------------- #
def _stock_rows(h: pd.DataFrame, as_of) -> pd.DataFrame:
    return h[(h["as_of"] == as_of) & h["asset_class"].isin(STOCKS) & (h["quantity"] > 0)]


def implied_prices(rows: pd.DataFrame) -> pd.Series:
    """Price per share (₹) by ISIN from disclosed value / quantity. Funds agree to within rounding."""
    good = rows[rows["market_value_lakh"] > 0]
    return (good["market_value_lakh"] * 1e5 / good["quantity"]).groupby(good["isin"]).median()


def _isin_changes(old: pd.DataFrame, cur: pd.DataFrame) -> dict[str, str]:
    """Old ISIN -> new ISIN where a stock's ISIN changed: same issuer prefix, one vanished and one appeared."""
    old_set, cur_set = set(old["isin"]), set(cur["isin"])
    fresh: dict[str, list[str]] = {}
    for isin in cur_set - old_set:
        if isin.startswith("IN"):
            fresh.setdefault(isin[:9], []).append(isin)
    return {i: fresh[i[:9]][0] for i in old_set - cur_set
            if i.startswith("IN") and len(fresh.get(i[:9], [])) == 1}


def _nearest_clean_ratio(ratios: np.ndarray) -> np.ndarray:
    """The clean multiple each ratio sits on, or NaN if it isn't on one."""
    k = np.array(CLEAN_RATIOS)
    errors = np.abs(ratios[:, None] / k - 1)
    out = k[errors.argmin(axis=1)]
    return np.where(errors.min(axis=1) <= RATIO_TOLERANCE, out, np.nan)


def corporate_actions(h: pd.DataFrame, prev, as_of) -> pd.DataFrame:
    """Bonus issues and splits between two month-ends: isin_prev, isin, factor (share multiple).

    Two independent signals must agree. A fund that didn't trade shows a share count
    multiplied by exactly a clean ratio, and the disclosed price per share falls by about
    the same ratio. That keeps an ordinary purchase in a falling stock from being mistaken
    for a bonus, and catches ratios a fixed "shares doubled" rule misses (a 1:2 bonus is 1.5x).

    A small multiple barely moves the price, so it needs two funds landing on it exactly.
    A changed ISIN with the same issuer prefix is itself evidence of a split, so there the
    multiple can come from the price alone when every holder also traded that month.
    """
    empty = pd.DataFrame(columns=["isin_prev", "isin", "factor"])
    old, cur = _stock_rows(h, prev), _stock_rows(h, as_of)
    if old.empty or cur.empty:
        return empty
    remap = _isin_changes(old, cur)
    new_to_old = {v: k for k, v in remap.items()}
    old = old.assign(isin=old["isin"].replace(remap))
    p0, p1 = implied_prices(old), implied_prices(cur)
    both = old[["scheme_code", "isin", "quantity"]].merge(
        cur[["scheme_code", "isin", "quantity"]], on=["scheme_code", "isin"], suffixes=("_prev", ""))
    both["clean"] = _nearest_clean_ratio((both["quantity"] / both["quantity_prev"]).to_numpy())
    rows = {}
    for (isin, k), g in both.dropna(subset=["clean"]).groupby(["isin", "clean"]):
        if isin not in p0.index or isin not in p1.index:
            continue
        move, n = p1[isin] / p0[isin], len(g)
        plausible = 0.7 <= move * k <= 1.4                     # what's left is an ordinary month's move
        confirmed = (k >= 1.5 and move <= 0.85) or (k < 1.5 and n >= 2 and move < 1.0)
        if plausible and confirmed and n > rows.get(isin, (0, 0))[1]:
            rows[isin] = (k, n)
    out = [{"isin_prev": new_to_old.get(i, i), "isin": i, "factor": v[0]} for i, v in rows.items()]
    for new, old_isin in new_to_old.items():
        if new in rows:
            continue
        factor = 1.0
        if new in p0.index and new in p1.index:
            move = p1[new] / p0[new]
            candidates = (1.0, *CLEAN_RATIOS)
            best = min(candidates, key=lambda c: abs(np.log(move * c)))
            factor = best if abs(np.log(move * best)) <= np.log(1.25) else 1.0
        out.append({"isin_prev": old_isin, "isin": new, "factor": factor})
    return pd.DataFrame(out, columns=empty.columns)


def _first_word(name: object) -> str:
    words = re.findall(r"[a-z0-9]+", str(name).lower())
    return words[0] if words else ""


def _is_demerger_ratio(x: float, tol: float = 0.002) -> bool:
    """Entitlements are small simple fractions: 1 for 1, 1 for 10, 3 for 4. Not 23 for 3, or 4 for 3."""
    f = Fraction(x).limit_denominator(10)
    return 0 < f.numerator <= 3 and x <= 2 and abs(x / float(f) - 1) <= tol


def spin_offs(h: pd.DataFrame, prev, as_of, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Demerged shares: isin (the new company), parent_isin, ratio.

    A stock no tracked fund held last month appears, and in every fund holding it the share
    count is one simple fraction of what that fund held of a single other stock. With 50-odd
    possible parents, one fund can hit a simple fraction by chance. So a parent must share
    the new company's first name (ITC Hotels from ITC, Vedanta Aluminium from Vedanta), or
    at least three funds must land on the identical ratio, which chance doesn't produce.
    """
    empty = pd.DataFrame(columns=["isin", "parent_isin", "ratio"])
    old, cur = _stock_rows(h, prev), _stock_rows(h, as_of)
    if old.empty or cur.empty:
        return empty
    actions = corporate_actions(h, prev, as_of) if actions is None else actions
    fresh = set(cur["isin"]) - set(old["isin"]) - set(actions["isin"])
    if not fresh:
        return empty
    # Last month's share counts as one table (stock x fund), so every candidate parent is tested at once.
    held = old.pivot_table(index="isin", columns="scheme_code", values="quantity", aggfunc="sum")
    words = old.drop_duplicates("isin").set_index("isin")["instrument"].map(_first_word).reindex(held.index)
    rows = []
    for isin, g in cur[cur["isin"].isin(fresh)].groupby("isin"):
        child = g.groupby("scheme_code")["quantity"].sum()
        if child.min() < 100 or not child.index.isin(held.columns).all():
            continue
        candidates = held[child.index]
        if len(child) < 3:                                  # few holders: the name must match too
            candidates = candidates[(words == _first_word(g["instrument"].iloc[0])).to_numpy()]
        candidates = candidates.dropna()                    # every holder must have held the parent
        if candidates.empty:
            continue
        ratios = child.to_numpy() / candidates.to_numpy()
        consistent = np.abs(ratios / ratios[:, :1] - 1).max(axis=1) <= 0.002
        for parent, r in zip(candidates.index[consistent], ratios[consistent, 0], strict=True):
            if _is_demerger_ratio(float(r)):
                rows.append({"isin": isin, "parent_isin": parent, "ratio": float(r)})
                break
    return pd.DataFrame(rows, columns=empty.columns)


def changes(h: pd.DataFrame, code: str, as_of=None, prev=None, actions: pd.DataFrame | None = None,
            spun: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stock-level changes between two month-ends, classified by split-adjusted quantity."""
    ms = months(h, code)
    if len(ms) < 2:
        return pd.DataFrame()
    as_of = as_of if as_of is not None else ms[0]
    prev = prev if prev is not None else next((m for m in ms if m < as_of), None)
    if prev is None:
        return pd.DataFrame()
    actions = corporate_actions(h, prev, as_of) if actions is None else actions
    spun = spin_offs(h, prev, as_of, actions) if spun is None else spun
    cur, old = snapshot(h, code, as_of), snapshot(h, code, prev)
    m = compare(cur[cur["asset_class"].isin(STOCKS)], old[old["asset_class"].isin(STOCKS)], actions, spun)
    m["as_of"], m["prev"] = as_of, prev
    # quantity_prev is restated for any bonus or split, so it compares like with like.
    keep = ["isin", "instrument", "industry", "cap_class", "status", "quantity_prev", "quantity",
            "pct_nav_prev", "pct_nav", "weight_change", "flow_cr", "as_of", "prev"]
    return m[keep].sort_values("flow_cr", ascending=False).reset_index(drop=True)


def compare(cur: pd.DataFrame, old: pd.DataFrame, actions: pd.DataFrame, spun: pd.DataFrame) -> pd.DataFrame:
    """Classify every (fund, stock) between two month-ends. Works on any number of funds at once."""
    cols = ["scheme_code", "isin", "instrument", "industry", "cap_class", "quantity", "pct_nav", "market_value_lakh"]
    cur, old = cur[cols], old[cols]
    # Restate last month in today's terms: today's ISIN, and share counts after any bonus or split.
    factors = dict(zip(actions["isin_prev"], actions["factor"], strict=True))
    old = old.assign(quantity=old["quantity"] * old["isin"].map(factors).fillna(1.0),
                     isin=old["isin"].replace(dict(zip(actions["isin_prev"], actions["isin"], strict=True))))
    adjusted = {i for i, f in zip(actions["isin"], actions["factor"], strict=True) if f != 1.0}
    demerged = set(spun["isin"])
    m = cur.merge(old, on=["scheme_code", "isin"], how="outer", suffixes=("", "_prev"))
    for c in ("instrument", "industry", "cap_class"):
        m[c] = m[c].fillna(m[f"{c}_prev"])
    m[["quantity", "pct_nav", "market_value_lakh", "quantity_prev", "pct_nav_prev"]] = \
        m[["quantity", "pct_nav", "market_value_lakh", "quantity_prev", "pct_nav_prev"]].fillna(0.0)

    # Classified with vectorised conditions, first match wins (np.select). The event study
    # calls this for every fund and month, so a row-by-row apply was the slowest step.
    q, q0 = m["quantity"].to_numpy(), m["quantity_prev"].to_numpy()
    ratio = np.divide(q, q0, out=np.full(len(m), np.nan), where=q0 > 0)
    flat_weight = (m["pct_nav"] - m["pct_nav_prev"]).abs() < 0.25 * m["pct_nav_prev"].clip(lower=0.01)
    m["status"] = np.select(
        [(q0 == 0) & (q > 0) & m["isin"].isin(demerged),
         (q0 == 0) & (q > 0),
         (q == 0) & (q0 > 0),
         q0 <= 0,
         m["isin"].isin(adjusted) & (np.abs(ratio - 1) <= 0.01),
         # Fallback for rows with no usable price: shares doubled while the weight stayed flat.
         (ratio >= 1.9) & flat_weight,
         ratio > 1.01,
         ratio < 0.99],
        ["Spin-off", "New", "Exited", "Unchanged", "Split/bonus?", "Split/bonus?", "Added", "Trimmed"],
        default="Unchanged")
    # Price per share now (or last month for exits) -> rupee value of the shares bought or sold.
    price_now = (m["market_value_lakh"] / m["quantity"]).where(m["quantity"] > 0)
    price_prev = (m["market_value_lakh_prev"] / m["quantity_prev"]).where(m["quantity_prev"] > 0)
    price = price_now.fillna(price_prev)
    m["flow_cr"] = ((m["quantity"] - m["quantity_prev"]) * price / 100).where(
        m["status"].isin(["New", "Exited", "Added", "Trimmed"]), 0.0)
    m["weight_change"] = m["pct_nav"] - m["pct_nav_prev"]
    return m


NOT_A_TRADE = ("Split/bonus?", "Spin-off")


def smart_money_flow(h: pd.DataFrame, codes: list[str], as_of=None) -> pd.DataFrame:
    """Across funds: which stocks were bought or sold most in the latest month (₹ crore)."""
    frames = []
    cache: dict[tuple, tuple] = {}
    for code in codes:
        ms = months(h, code)
        if len(ms) < 2:
            continue
        target = as_of if as_of is not None and as_of in ms else ms[0]
        prev = next((m for m in ms if m < target), None)
        if prev is None:
            continue
        if (prev, target) not in cache:
            acts = corporate_actions(h, prev, target)
            cache[(prev, target)] = (acts, spin_offs(h, prev, target, acts))
        c = changes(h, code, target, prev, *cache[(prev, target)])
        if not c.empty:
            frames.append(c.assign(scheme_code=code))
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames)
    allc = allc[~allc["status"].isin(NOT_A_TRADE)]
    g = allc.groupby("isin").agg(
        instrument=("instrument", "first"), industry=("industry", "first"),
        cap_class=("cap_class", "first"),
        net_flow_cr=("flow_cr", "sum"),
        funds_buying=("status", lambda s: int(s.isin(["New", "Added"]).sum())),
        funds_selling=("status", lambda s: int(s.isin(["Exited", "Trimmed"]).sum())),
        new_positions=("status", lambda s: int((s == "New").sum())),
        full_exits=("status", lambda s: int((s == "Exited").sum())),
        funds_holding=("pct_nav", lambda s: int((s > 0).sum())),
    )
    return g.sort_values("net_flow_cr", ascending=False).reset_index()


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Portfolio overlap of two snapshots: sum over common stocks of the smaller weight."""
    wa = a[a["asset_class"].isin(STOCKS)].groupby("isin")["pct_nav"].sum()
    wb = b[b["asset_class"].isin(STOCKS)].groupby("isin")["pct_nav"].sum()
    common = wa.index.intersection(wb.index)
    names = a.drop_duplicates("isin").set_index("isin")["instrument"]
    detail = pd.DataFrame({"instrument": names.reindex(common), "weight_a": wa.reindex(common),
                           "weight_b": wb.reindex(common)})
    detail["overlap"] = detail[["weight_a", "weight_b"]].min(axis=1)
    union = wa.index.union(wb.index)
    return {
        "weighted_pct": float(detail["overlap"].sum()),
        "common_stocks": int(len(common)),
        "jaccard": float(len(common) / len(union)) if len(union) else 0.0,
        "detail": detail.sort_values("overlap", ascending=False).reset_index(),
    }


def overlap_matrix(h: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    snaps = {c: snapshot(h, c) for c in codes}
    snaps = {c: s for c, s in snaps.items() if not s.empty}
    keys = list(snaps)
    mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for i, x in enumerate(keys):
        for y in keys[i:]:
            v = 100.0 if x == y else overlap(snaps[x], snaps[y])["weighted_pct"]
            mat.loc[x, y] = mat.loc[y, x] = v
    return mat


# --------------------------------------------------------------------------- #
# Style drift against SEBI's category rules (Oct 2017 categorisation circular)
# --------------------------------------------------------------------------- #
def category_rule(category: str | None) -> dict | None:
    c = (category or "").lower()
    if "large & mid" in c or "large and mid" in c:
        return {"label": "≥35% large and ≥35% mid cap", "min": {"Large Cap": 35, "Mid Cap": 35}}
    if "large cap" in c:
        return {"label": "≥80% large cap", "min": {"Large Cap": 80}}
    if "mid cap" in c:
        return {"label": "≥65% mid cap", "min": {"Mid Cap": 65}}
    if "small cap" in c:
        return {"label": "≥65% small cap", "min": {"Small Cap": 65}}
    if "focused" in c:
        return {"label": "at most 30 stocks", "max_stocks": 30}
    if "elss" in c:
        return {"label": "≥80% in equity", "min_equity": 80}
    if "flexi" in c or "value" in c or "contra" in c:
        return {"label": "≥65% in equity", "min_equity": 65}
    return None


def style_drift(h: pd.DataFrame, code: str, category: str | None, last: int | None = None) -> pd.DataFrame:
    """Per month (optionally only the `last` few): cap mix, equity share, stock count, SEBI rule met or not."""
    rule = category_rule(category)
    rows = []
    for m in sorted(months(h, code))[-(last or 0):]:
        snap = snapshot(h, code, m)
        caps = cap_mix(snap)
        equity = snap.loc[snap["asset_class"].str.startswith("Equity") |
                          (snap["asset_class"] == "Foreign Equity"), "pct_nav"].sum()
        stocks = concentration(snap)["stocks"]
        ok = None
        if rule:
            checks = [caps.get(k, 0) >= v for k, v in rule.get("min", {}).items()]
            if "min_equity" in rule:
                checks.append(equity >= rule["min_equity"])
            if "max_stocks" in rule:
                checks.append(stocks <= rule["max_stocks"])
            ok = all(checks)
        rows.append({"as_of": m, **{k: float(caps.get(k, 0)) for k in
                     ["Large Cap", "Mid Cap", "Small Cap", "Foreign", "Unclassified"]},
                     "equity_pct": float(equity), "stocks": stocks,
                     "rule": rule["label"] if rule else "", "meets_rule": ok})
    return pd.DataFrame(rows)
