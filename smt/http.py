"""One shared HTTP session with retries, timeouts and a browser-like User-Agent."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from smt import settings

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=settings.HTTP_RETRIES,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = settings.USER_AGENT
        _session = s
    return _session


def get(url: str, **kwargs) -> requests.Response:
    """GET a URL; raises requests.HTTPError on a non-2xx response."""
    resp = session().get(url, timeout=settings.HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp
