"""DSP Mutual Fund: one zip per month on dspim.com holding every scheme.

File names are not consistent from month to month, for example
  monthend-portfolios_31_july_2026.zip
  monthend-portfolio-february-28-2026.zip
  dsp-monthend-portfolio-as-on-30-september-2026.zip
so the month and year are read from the name's tokens. Inside the zip, the
equity workbook has one sheet per scheme; the debt workbook is skipped.
"""
from __future__ import annotations

import re

from smt.holdings.connectors.base import Connector, DisclosureFile, month_end, month_year, newest_first, page_links


class DSPConnector(Connector):
    name = "dsp"
    page_url = "https://www.dspim.com/mandatory-disclosures/portfolio-disclosures"

    def discover(self, html: str | None = None) -> list[DisclosureFile]:
        html = self.fetch_page() if html is None else html
        files = []
        for url in page_links(html, self.page_url):
            fname = url.rsplit("/", 1)[-1].split("?")[0].lower()
            if not fname.endswith(".zip") or not re.search(r"month-?end-portfolio", fname):
                continue
            my = month_year(re.split(r"[-_.]", fname))
            if my:
                files.append(DisclosureFile(self.name, url, month_end(*my), "all"))
        return newest_first(files)

    def keep_member(self, filename: str) -> bool:
        name = filename.lower()
        return name.endswith((".xlsx", ".xls")) and "debt" not in name
