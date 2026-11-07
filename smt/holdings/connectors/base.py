"""Connector interface and shared helpers."""
from __future__ import annotations

import calendar
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin

from smt import http
from smt.holdings.parser import ParsedPortfolio, parse_workbook

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9


@dataclass(frozen=True)
class DisclosureFile:
    connector: str
    url: str
    as_of: date        # month-end date the portfolio describes
    label: str = ""    # connector-specific (e.g. the scheme prefix in the file name)


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def month_year(tokens: list[str]) -> tuple[int, int] | None:
    """Find a month name and a 4-digit year among filename tokens."""
    month = next((MONTHS[t.lower()] for t in tokens if t.lower() in MONTHS), None)
    year = next((int(t) for t in tokens if re.fullmatch(r"20\d\d", t)), None)
    return (year, month) if month and year else None


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def page_links(html: str, base_url: str) -> list[str]:
    parser = _Links()
    parser.feed(html)
    return [urljoin(base_url, h) for h in parser.hrefs]


class Connector:
    """Find and download one AMC's month-end portfolio files."""

    name = "base"
    page_url = ""

    def discover(self, html: str | None = None) -> list[DisclosureFile]:
        """List available disclosure files, newest first. `html` lets tests skip the network."""
        raise NotImplementedError

    def fetch_page(self) -> str:
        return http.get(self.page_url).text

    def download(self, f: DisclosureFile) -> bytes:
        return http.get(f.url).content

    def keep_member(self, filename: str) -> bool:
        """For zip downloads: which files inside the archive to parse."""
        return filename.lower().endswith((".xlsx", ".xls"))

    def parse(self, content: bytes) -> list[ParsedPortfolio]:
        if content[:2] == b"PK" and _is_zip_of_workbooks(content):
            out = []
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for member in z.namelist():
                    if self.keep_member(member):
                        out.extend(parse_workbook(z.read(member)))
            return out
        return parse_workbook(content)


def _is_zip_of_workbooks(content: bytes) -> bool:
    """An .xlsx is itself a zip; a zip *of* workbooks has no [Content_Types].xml at its root."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        return "[Content_Types].xml" not in z.namelist()


def newest_first(files: list[DisclosureFile]) -> list[DisclosureFile]:
    """Sort newest first and keep one file per (month, label), the last one listed."""
    seen: dict[tuple[date, str], DisclosureFile] = {}
    for f in files:
        seen[(f.as_of, f.label)] = f
    return sorted(seen.values(), key=lambda f: (f.as_of, f.label), reverse=True)
