"""The funds this app tracks.

Every fund is identified by its AMFI scheme code (Direct Plan, Growth option).
`expected_name` is checked against AMFI's own scheme master on every refresh,
so a wrong code fails loudly instead of silently analysing the wrong fund.

`holdings` names the connector that downloads the fund's official monthly
portfolio disclosure (see smt/holdings/connectors). Funds without one get
NAV-based analytics only.
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


FUNDS: tuple[Fund, ...] = (
    # PPFAS: holdings from amc.ppfas.com monthly disclosures
    Fund("122639", "Parag Parikh Flexi Cap Fund", "PPFAS", holdings="ppfas"),
    Fund("147481", "Parag Parikh ELSS Tax Saver Fund", "PPFAS", holdings="ppfas"),
    Fund("154155", "Parag Parikh Large Cap Fund", "PPFAS", holdings="ppfas"),
    # DSP: holdings from dspim.com month-end portfolio files
    Fund("119076", "DSP Flexi Cap Fund", "DSP", holdings="dsp"),
    Fund("119250", "DSP Large Cap Fund", "DSP", holdings="dsp"),
    Fund("119218", "DSP Large & Mid Cap Fund", "DSP", holdings="dsp"),
    Fund("119071", "DSP Midcap Fund", "DSP", holdings="dsp"),
    Fund("119212", "DSP Small Cap Fund", "DSP", holdings="dsp"),
    Fund("119242", "DSP ELSS Tax Saver Fund", "DSP", holdings="dsp"),
    Fund("119096", "DSP Focused Fund", "DSP", holdings="dsp"),
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
    # Benchmark: an index fund NAV is an investable Nifty 50 (includes its small cost).
    Fund("120716", "UTI Nifty 50 Index Fund", "UTI", is_benchmark=True),
)

BENCHMARK_CODE = next(f.scheme_code for f in FUNDS if f.is_benchmark)
BENCHMARK_NAME = "Nifty 50 (UTI index fund)"


def by_code(code: str) -> Fund:
    for f in FUNDS:
        if f.scheme_code == code:
            return f
    raise KeyError(code)


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
