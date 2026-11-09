"""The funds this app tracks.

Every fund is identified by its AMFI scheme code (Direct Plan, Growth option).
`expected_name` is checked against AMFI's own scheme master on every refresh,
so a wrong code fails loudly instead of silently analysing the wrong fund.

`holdings` names the connector that downloads the fund's official monthly
portfolio disclosure (see smt/holdings/connectors). Funds without one get
NAV-based analytics only.

Funds with `is_benchmark=True` are reference series, not funds being judged:
index funds are investable versions of each market segment (their NAV includes
real costs, which keeps comparisons fair to active managers), and a liquid fund
stands in for cash. They are verified against AMFI like every other fund.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Fund:
    scheme_code: str
    expected_name: str          # AMFI scheme name, without plan/option
    amc: str
    holdings: str | None = None  # connector name, e.g. "ppfas", "dsp"
    is_benchmark: bool = False
    former_names: tuple[str, ...] = ()  # old names still printed in earlier disclosure files
    label: str | None = None     # display name, for reference series


FUNDS: tuple[Fund, ...] = (
    # PPFAS: holdings from amc.ppfas.com monthly disclosures
    Fund("122639", "Parag Parikh Flexi Cap Fund", "PPFAS", holdings="ppfas"),
    Fund("147481", "Parag Parikh ELSS Tax Saver Fund", "PPFAS", holdings="ppfas",
         former_names=("Parag Parikh Tax Saver Fund",)),
    Fund("154155", "Parag Parikh Large Cap Fund", "PPFAS", holdings="ppfas"),
    # DSP: holdings from dspim.com month-end portfolio files
    Fund("119076", "DSP Flexi Cap Fund", "DSP", holdings="dsp"),
    Fund("119250", "DSP Large Cap Fund", "DSP", holdings="dsp", former_names=("DSP Top 100 Equity Fund",)),
    Fund("119218", "DSP Large & Mid Cap Fund", "DSP", holdings="dsp", former_names=("DSP Equity Opportunities Fund",)),
    Fund("119071", "DSP Midcap Fund", "DSP", holdings="dsp"),
    Fund("119212", "DSP Small Cap Fund", "DSP", holdings="dsp"),
    Fund("119242", "DSP ELSS Tax Saver Fund", "DSP", holdings="dsp", former_names=("DSP Tax Saver Fund",)),
    Fund("119096", "DSP Focused Fund", "DSP", holdings="dsp", former_names=("DSP Focus Fund",)),
    Fund("148595", "DSP Value Fund", "DSP", holdings="dsp"),
    # Other large AMCs: NAV analytics only (their disclosure sites need a browser)
    Fund("118955", "HDFC Flexi Cap Fund", "HDFC"),
    Fund("118989", "HDFC Mid Cap Fund", "HDFC"),
    Fund("119598", "SBI Large Cap Fund", "SBI"),
    Fund("125497", "SBI SMALL CAP FUND", "SBI"),
    Fund("120465", "Axis Large Cap Fund", "Axis"),
    Fund("120586", "ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund)", "ICICI Prudential"),
    Fund("118825", "Mirae Asset Large Cap Fund", "Mirae Asset"),
    Fund("118778", "Nippon India Small Cap Fund", "Nippon India"),
    Fund("120828", "Quant Small Cap Fund", "Quant"),
    # Reference series: an index fund's NAV is an investable index, costs included.
    Fund("120716", "UTI Nifty 50 Index Fund", "UTI", is_benchmark=True, label="Nifty 50 (UTI index fund)"),
    Fund("147666", "Axis Nifty 100 Index Fund", "Axis", is_benchmark=True, label="Nifty 100 (Axis index fund)"),
    Fund("149343", "Edelweiss NIFTY Large Midcap 250 Index Fund", "Edelweiss", is_benchmark=True,
         label="Nifty LargeMidcap 250 (Edelweiss index fund)"),
    Fund("147622", "Motilal Oswal Nifty Midcap 150 Index Fund", "Motilal Oswal", is_benchmark=True,
         label="Nifty Midcap 150 (Motilal Oswal index fund)"),
    Fund("147623", "Motilal Oswal Nifty Smallcap 250 Index Fund", "Motilal Oswal", is_benchmark=True,
         label="Nifty Smallcap 250 (Motilal Oswal index fund)"),
    Fund("147625", "Motilal Oswal Nifty 500 Index Fund", "Motilal Oswal", is_benchmark=True,
         label="Nifty 500 (Motilal Oswal index fund)"),
    Fund("143269", "Parag Parikh Liquid Fund", "PPFAS", is_benchmark=True, label="Cash (Parag Parikh Liquid Fund)"),
)

BENCHMARK_CODE = "120716"      # the broad-market reference used where no single category applies
BENCHMARK_NAME = "Nifty 50 (UTI index fund)"
NIFTY_100, LARGEMID_250, MIDCAP_150, SMALLCAP_250, NIFTY_500 = "147666", "149343", "147622", "147623", "147625"
CASH_PROXY = "143269"
# The segments used to estimate a fund's market-cap mix from its NAV alone (analytics/style.py).
STYLE_FACTORS = {"Large Cap": NIFTY_100, "Mid Cap": MIDCAP_150, "Small Cap": SMALLCAP_250, "Cash": CASH_PROXY}


def category_index(category: str | None) -> str:
    """The index fund matching a SEBI category's mandate.

    Judging a small-cap fund against the Nifty 50 credits the manager with the
    small-cap premium; the fair question is whether it beat small caps.
    """
    c = (category or "").lower()
    if "large & mid" in c or "large and mid" in c:
        return LARGEMID_250
    if "large cap" in c:
        return NIFTY_100
    if "mid cap" in c:
        return MIDCAP_150
    if "small cap" in c:
        return SMALLCAP_250
    return NIFTY_500               # flexi cap, ELSS, focused, value: the whole market


def by_code(code: str) -> Fund:
    for f in FUNDS:
        if f.scheme_code == code:
            return f
    raise KeyError(code)


def names_in_disclosures(fund: Fund) -> tuple[str, ...]:
    """Every name a fund's portfolio may be filed under, current name first."""
    return (fund.expected_name, *fund.former_names)


def tracked(include_benchmark: bool = False) -> list[Fund]:
    return [f for f in FUNDS if include_benchmark or not f.is_benchmark]


def with_holdings(connector: str | None = None) -> list[Fund]:
    return [f for f in FUNDS if f.holdings and (connector is None or f.holdings == connector)]


def normalise(name: str) -> str:
    """Name used to match a fund with the scheme title inside its disclosure file.

    AMFI and AMCs spell names differently ("DSP Midcap Fund" vs "DSP Mid Cap Fund",
    "&" vs "and", a descriptive suffix in brackets), so those differences are removed.
    """
    name = name.lower().replace("&", "and")
    name = name.split("(")[0]
    name = " ".join("".join(c if c.isalnum() else " " for c in name).split())
    return re.sub(r"\b(mid|small|large|flexi|multi)\s+cap\b", r"\1cap", name)
