"""PPFAS Mutual Fund: one file per scheme per month on amc.ppfas.com.

File names look like
  /downloads/portfolio-disclosure/2026/PPFCF_PPFAS_Monthly_Portfolio_Report_September_30_2026.xlsx?07102026
where the prefix is the scheme (PPFCF = Flexi Cap, PPTSF = ELSS Tax Saver, ...).
"""
from __future__ import annotations

import re

from smt.holdings.connectors.base import Connector, DisclosureFile, month_end, month_year, newest_first, page_links

FILE_RE = re.compile(
    r"/(?P<prefix>[A-Z]+)_PPFAS_Monthly_Portfolio_Report_(?P<rest>[A-Za-z]+_\d{1,2}_\d{4})\.xlsx?(\?|$)"
)
# Debt, liquid and arbitrage schemes are not analysed, so their files are skipped.
SKIP_PREFIXES = {"PPLF", "PPAF", "PPCHF", "PPDAAF"}


class PPFASConnector(Connector):
    name = "ppfas"
    page_url = "https://amc.ppfas.com/downloads/portfolio-disclosure/"

    def discover(self, html: str | None = None) -> list[DisclosureFile]:
        html = self.fetch_page() if html is None else html
        files = []
        for url in page_links(html, self.page_url):
            m = FILE_RE.search(url)
            if not m or m.group("prefix") in SKIP_PREFIXES:
                continue
            my = month_year(m.group("rest").split("_"))
            if my:
                files.append(DisclosureFile(self.name, url, month_end(*my), m.group("prefix")))
        return newest_first(files)
