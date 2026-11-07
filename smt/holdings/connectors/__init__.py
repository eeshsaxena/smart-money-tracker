"""AMC connectors. Each one only knows how to find and download its AMC's
monthly disclosure files; parsing is shared (smt.holdings.parser).

To add an AMC: subclass Connector, implement discover(), register it below,
and set holdings="<name>" on its funds in smt/universe.py.
"""
from __future__ import annotations

from smt.holdings.connectors.base import Connector, DisclosureFile
from smt.holdings.connectors.dsp import DSPConnector
from smt.holdings.connectors.ppfas import PPFASConnector

REGISTRY: dict[str, type[Connector]] = {
    PPFASConnector.name: PPFASConnector,
    DSPConnector.name: DSPConnector,
}


def default_connectors() -> dict[str, Connector]:
    return {name: cls() for name, cls in REGISTRY.items()}


__all__ = ["Connector", "DisclosureFile", "REGISTRY", "default_connectors"]
