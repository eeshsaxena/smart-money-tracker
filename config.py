from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FUND_MANAGERS = {
    "prashant_jain": {
        "name": "Prashant Jain",
        "funds": [
            {"scheme_code": "100119", "name": "HDFC Flexi Cap Fund"},
            {"scheme_code": "100027", "name": "HDFC Balanced Advantage Fund"},
        ],
    },
    "rajeev_thakkar": {
        "name": "Rajeev Thakkar",
        "funds": [
            {"scheme_code": "122639", "name": "PPFAS Flexi Cap Fund"},
        ],
    },
}

MARKET_CAP_THRESHOLDS = {
    "large_cap": {"rank_start": 1, "rank_end": 100},
    "mid_cap": {"rank_start": 101, "rank_end": 250},
    "small_cap": {"rank_start": 251, "rank_end": 500},
}

CATEGORY_MANDATES = {
    "large_cap": {"large_cap_min": 0.80},
    "mid_cap": {"mid_cap_min": 0.65},
    "small_cap": {"small_cap_min": 0.65},
    "flexi_cap": {},
    "large_and_mid_cap": {"large_cap_min": 0.35, "mid_cap_min": 0.35},
}

BENCHMARK_TICKER = "^NSEI"
RISK_FREE_RATE = 0.065
