from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FUND_MANAGERS = {
    "prashant_jain": {
        "name": "Prashant Jain",
        "amc": "HDFC Mutual Fund",
        "funds": [
            {"scheme_code": "100119", "name": "HDFC Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XVAL.BO"},
            {"scheme_code": "100027", "name": "HDFC Balanced Advantage Fund", "category": "balanced_advantage", "ticker": "0P0000XVB3.BO"},
            {"scheme_code": "100025", "name": "HDFC Top 100 Fund", "category": "large_cap", "ticker": "0P0000XVAT.BO"},
        ],
    },
    "rajeev_thakkar": {
        "name": "Rajeev Thakkar",
        "amc": "PPFAS Mutual Fund",
        "funds": [
            {"scheme_code": "122639", "name": "PPFAS Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XVAA.BO"},
            {"scheme_code": "145455", "name": "PPFAS Conservative Hybrid Fund", "category": "conservative_hybrid", "ticker": ""},
        ],
    },
    "saurabh_mukherjea": {
        "name": "Saurabh Mukherjea",
        "amc": "Marcellus Investment Managers",
        "funds": [
            {"scheme_code": "149889", "name": "Marcellus Consistent Compounders Fund", "category": "flexi_cap", "ticker": ""},
        ],
    },
    "nilesh_shah": {
        "name": "Nilesh Shah",
        "amc": "Kotak Mutual Fund",
        "funds": [
            {"scheme_code": "100331", "name": "Kotak Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XVHZ.BO"},
            {"scheme_code": "100337", "name": "Kotak Emerging Equity Fund", "category": "mid_cap", "ticker": "0P0000XVI8.BO"},
            {"scheme_code": "120503", "name": "Kotak Small Cap Fund", "category": "small_cap", "ticker": "0P0000XVHQ.BO"},
            {"scheme_code": "100333", "name": "Kotak Bluechip Fund", "category": "large_cap", "ticker": "0P0000XVHK.BO"},
        ],
    },
    "sankaran_naren": {
        "name": "S Naren",
        "amc": "ICICI Prudential Mutual Fund",
        "funds": [
            {"scheme_code": "100123", "name": "ICICI Pru Value Discovery Fund", "category": "value", "ticker": "0P0000XVGS.BO"},
            {"scheme_code": "100356", "name": "ICICI Pru Bluechip Fund", "category": "large_cap", "ticker": "0P0000XVGF.BO"},
            {"scheme_code": "120587", "name": "ICICI Pru Midcap Fund", "category": "mid_cap", "ticker": "0P0000XVGO.BO"},
            {"scheme_code": "100124", "name": "ICICI Pru Multicap Fund", "category": "multi_cap", "ticker": "0P0000XVH1.BO"},
        ],
    },
    "neelesh_surana": {
        "name": "Neelesh Surana",
        "amc": "Mirae Asset Mutual Fund",
        "funds": [
            {"scheme_code": "118989", "name": "Mirae Asset Large Cap Fund", "category": "large_cap", "ticker": "0P0000XV3Y.BO"},
            {"scheme_code": "120847", "name": "Mirae Asset Midcap Fund", "category": "mid_cap", "ticker": ""},
            {"scheme_code": "147781", "name": "Mirae Asset Flexi Cap Fund", "category": "flexi_cap", "ticker": ""},
        ],
    },
    "shreyash_devalkar": {
        "name": "Shreyash Devalkar",
        "amc": "Axis Mutual Fund",
        "funds": [
            {"scheme_code": "120465", "name": "Axis Bluechip Fund", "category": "large_cap", "ticker": "0P0000XV1X.BO"},
            {"scheme_code": "120503", "name": "Axis Midcap Fund", "category": "mid_cap", "ticker": "0P0000XV24.BO"},
            {"scheme_code": "135781", "name": "Axis Small Cap Fund", "category": "small_cap", "ticker": "0P0000XV2A.BO"},
            {"scheme_code": "143579", "name": "Axis Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XV1W.BO"},
        ],
    },
    "r_srinivasan": {
        "name": "R Srinivasan",
        "amc": "SBI Mutual Fund",
        "funds": [
            {"scheme_code": "119598", "name": "SBI Bluechip Fund", "category": "large_cap", "ticker": "0P0000XVKU.BO"},
            {"scheme_code": "100057", "name": "SBI Magnum Midcap Fund", "category": "mid_cap", "ticker": "0P0000XVLF.BO"},
            {"scheme_code": "125497", "name": "SBI Small Cap Fund", "category": "small_cap", "ticker": "0P0000XVLN.BO"},
            {"scheme_code": "119607", "name": "SBI Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XVL5.BO"},
        ],
    },
    "anish_tawakley": {
        "name": "Anish Tawakley",
        "amc": "ICICI Prudential Mutual Fund",
        "funds": [
            {"scheme_code": "120587", "name": "ICICI Pru Flexi Cap Fund", "category": "flexi_cap", "ticker": "0P0000XVGE.BO"},
        ],
    },
    "jinesh_gopani": {
        "name": "Jinesh Gopani",
        "amc": "Axis Mutual Fund",
        "funds": [
            {"scheme_code": "120465", "name": "Axis Growth Opportunities Fund", "category": "large_and_mid_cap", "ticker": ""},
        ],
    },
}

ALL_SCHEME_CODES = []
for manager in FUND_MANAGERS.values():
    for fund in manager["funds"]:
        ALL_SCHEME_CODES.append(fund["scheme_code"])

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
    "multi_cap": {"large_cap_min": 0.25, "mid_cap_min": 0.25, "small_cap_min": 0.25},
    "value": {},
    "balanced_advantage": {},
    "conservative_hybrid": {},
}

SECTOR_MAP = {
    "Financial Services": "Financials",
    "Banks": "Financials",
    "Finance": "Financials",
    "Financial": "Financials",
    "Information Technology": "IT",
    "IT - Software": "IT",
    "IT": "IT",
    "Technology": "IT",
    "Automobile": "Auto",
    "Auto Ancillaries": "Auto",
    "Automobiles": "Auto",
    "Pharma": "Healthcare",
    "Healthcare": "Healthcare",
    "Pharmaceutical": "Healthcare",
    "Consumer Goods": "FMCG",
    "FMCG": "FMCG",
    "Consumer Staples": "FMCG",
    "Oil & Gas": "Energy",
    "Energy": "Energy",
    "Power": "Energy",
    "Metals": "Materials",
    "Cement": "Materials",
    "Chemicals": "Materials",
    "Construction": "Industrials",
    "Capital Goods": "Industrials",
    "Industrial Manufacturing": "Industrials",
    "Infrastructure": "Industrials",
    "Telecom": "Telecom",
    "Communication": "Telecom",
    "Realty": "Real Estate",
    "Real Estate": "Real Estate",
    "Consumer Durables": "Consumer Disc.",
    "Retail": "Consumer Disc.",
    "Textiles": "Consumer Disc.",
    "Media": "Consumer Disc.",
}

BENCHMARK_TICKER = "^NSEI"
NIFTY_MIDCAP_TICKER = "^NSEMDCP50"
NIFTY_SMALLCAP_TICKER = "^NSESMLCP"

RISK_FREE_RATE = 0.065

PEER_GROUPS = {
    "Flexi Cap": {
        "0P0000XVAA.BO": "PPFAS Flexi Cap",
        "0P0000XVAL.BO": "HDFC Flexi Cap",
        "0P0000XVHZ.BO": "Kotak Flexi Cap",
        "0P0000XVL5.BO": "SBI Flexi Cap",
        "0P0000XVGE.BO": "ICICI Pru Flexi Cap",
    },
    "Large Cap": {
        "0P0000XVAT.BO": "HDFC Top 100",
        "0P0000XVGF.BO": "ICICI Pru Bluechip",
        "0P0000XV1X.BO": "Axis Bluechip",
        "0P0000XVKU.BO": "SBI Bluechip",
        "0P0000XV3Y.BO": "Mirae Asset Large Cap",
        "0P0000XVHK.BO": "Kotak Bluechip",
    },
    "Mid Cap": {
        "0P0000XVI8.BO": "Kotak Emerging Equity",
        "0P0000XVGO.BO": "ICICI Pru Midcap",
        "0P0000XV24.BO": "Axis Midcap",
        "0P0000XVLF.BO": "SBI Magnum Midcap",
    },
    "Small Cap": {
        "0P0000XVHQ.BO": "Kotak Small Cap",
        "0P0000XV2A.BO": "Axis Small Cap",
        "0P0000XVLN.BO": "SBI Small Cap",
    },
}
