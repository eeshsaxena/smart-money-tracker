"""Analytics on month-end portfolio holdings.

Buying and selling is measured with share *quantities*, not portfolio weights.
A weight can rise just because the stock went up; a quantity only rises when
the fund bought more. Stock splits and bonus issues also raise quantities, so
a jump in quantity with an unchanged weight is flagged rather than called a buy.
"""
from __future__ import annotations

import pandas as pd

EQUITY = "Equity"
STOCKS = ("Equity", "Foreign Equity")     # directional stock holdings


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


def changes(h: pd.DataFrame, code: str, as_of=None, prev=None) -> pd.DataFrame:
    """Stock-level changes between two month-ends, classified by quantity."""
    ms = months(h, code)
    if len(ms) < 2:
        return pd.DataFrame()
    as_of = as_of if as_of is not None else ms[0]
    prev = prev if prev is not None else next((m for m in ms if m < as_of), None)
    if prev is None:
        return pd.DataFrame()
    cols = ["isin", "instrument", "industry", "cap_class", "quantity", "pct_nav", "market_value_lakh"]
    cur = snapshot(h, code, as_of)
    old = snapshot(h, code, prev)
    cur = cur[cur["asset_class"].isin(STOCKS)][cols]
    old = old[old["asset_class"].isin(STOCKS)][cols]
    m = cur.merge(old, on="isin", how="outer", suffixes=("", "_prev"))
    for c in ("instrument", "industry", "cap_class"):
        m[c] = m[c].fillna(m[f"{c}_prev"])
    m[["quantity", "pct_nav", "market_value_lakh", "quantity_prev", "pct_nav_prev"]] = \
        m[["quantity", "pct_nav", "market_value_lakh", "quantity_prev", "pct_nav_prev"]].fillna(0.0)

    def status(r) -> str:
        if r.quantity_prev == 0 and r.quantity > 0:
            return "New"
        if r.quantity == 0 and r.quantity_prev > 0:
            return "Exited"
        if r.quantity_prev <= 0:
            return "Unchanged"
        ratio = r.quantity / r.quantity_prev
        if ratio >= 1.9 and abs(r.pct_nav - r.pct_nav_prev) < 0.25 * max(r.pct_nav_prev, 0.01):
            return "Split/bonus?"
        if ratio > 1.01:
            return "Added"
        if ratio < 0.99:
            return "Trimmed"
        return "Unchanged"

    m["status"] = m.apply(status, axis=1)
    # Price per share now (or last month for exits) -> rupee value of the shares bought or sold.
    price_now = (m["market_value_lakh"] / m["quantity"]).where(m["quantity"] > 0)
    price_prev = (m["market_value_lakh_prev"] / m["quantity_prev"]).where(m["quantity_prev"] > 0)
    price = price_now.fillna(price_prev)
    m["flow_cr"] = ((m["quantity"] - m["quantity_prev"]) * price / 100).where(
        m["status"].isin(["New", "Exited", "Added", "Trimmed"]), 0.0)
    m["weight_change"] = m["pct_nav"] - m["pct_nav_prev"]
    m["as_of"], m["prev"] = as_of, prev
    keep = ["isin", "instrument", "industry", "cap_class", "status", "quantity_prev", "quantity",
            "pct_nav_prev", "pct_nav", "weight_change", "flow_cr", "as_of", "prev"]
    return m[keep].sort_values("flow_cr", ascending=False).reset_index(drop=True)


def smart_money_flow(h: pd.DataFrame, codes: list[str], as_of=None) -> pd.DataFrame:
    """Across funds: which stocks were bought or sold most in the latest month (₹ crore)."""
    frames = []
    for code in codes:
        ms = months(h, code)
        if len(ms) < 2:
            continue
        target = as_of if as_of is not None and as_of in ms else ms[0]
        c = changes(h, code, target)
        if not c.empty:
            frames.append(c.assign(scheme_code=code))
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames)
    allc = allc[allc["status"] != "Split/bonus?"]
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
    wa = a[a["asset_class"].isin(STOCKS)].set_index("isin")["pct_nav"]
    wb = b[b["asset_class"].isin(STOCKS)].set_index("isin")["pct_nav"]
    common = wa.index.intersection(wb.index)
    names = a.set_index("isin")["instrument"]
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


def style_drift(h: pd.DataFrame, code: str, category: str | None) -> pd.DataFrame:
    """Per month: cap mix, equity share, stock count, and whether the SEBI rule was met."""
    rule = category_rule(category)
    rows = []
    for m in sorted(months(h, code)):
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
