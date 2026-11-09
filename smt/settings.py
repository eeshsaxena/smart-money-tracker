"""Runtime settings. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("SMT_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("SMT_DB_PATH", DATA_DIR / "smt.db"))
RAW_DIR = DATA_DIR / "raw"          # downloaded disclosure files, kept for provenance

# Annual risk-free rate used for Sharpe / Sortino (91-day T-bill is ~6.5%).
RISK_FREE_RATE = float(os.getenv("SMT_RISK_FREE_RATE", "0.065"))
TRADING_DAYS = 252

HTTP_TIMEOUT = float(os.getenv("SMT_HTTP_TIMEOUT", "40"))
HTTP_RETRIES = int(os.getenv("SMT_HTTP_RETRIES", "3"))
USER_AGENT = os.getenv(
    "SMT_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
)

# How many month-end portfolio disclosures to load per AMC. Testing whether buying and
# selling predicts returns needs years of history, not months.
HOLDINGS_MONTHS = int(os.getenv("SMT_HOLDINGS_MONTHS", "36"))
