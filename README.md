# Smart Money Tracker — MF Intelligence Dashboard

SEBI mandates every mutual fund disclose their full portfolio holdings monthly — publicly, for free. This project scrapes those disclosures, cross-references with live stock prices, and builds a dashboard that answers: **what are India's top fund managers actually doing with money right now?**

## Features

### 1. Smart Money Flow Detector
Across all large/mid/small cap equity funds, which stocks are being collectively accumulated or exited by top managers this month vs. last? Ranked by net change in aggregate holding %. Conviction scores show the highest-weight stocks across all tracked managers.

### 2. Fund Manager Style Fingerprint
For each manager (Prashant Jain, Rajeev Thakkar, Saurabh Mukherjea, Nilesh Shah, S Naren, etc.):
- Large vs mid vs small cap tilt
- Growth vs value score (P/E, P/B of holdings vs index)
- Sector concentration (Herfindahl index)
- Turnover ratio (portfolio churn)
- **K-Means clustering** to find "managers who think alike"

### 3. Style Drift Detector
Tracks each fund's actual market-cap breakdown over time. Flags funds drifting outside their SEBI mandate — a genuine red flag for investors. Shows mandate compliance gauges and severity ratings (critical/warning/minor).

### 4. SIP XIRR Calculator with Peer Benchmarking
Computes true XIRR (annualized return accounting for monthly cash flows — not fake point-to-point returns). Benchmarks against Nifty 50 and the best fund in the same category. SIP growth visualization shows how INR 1,000/month would have grown.

### 5. Portfolio Overlap Heatmap
Pick any 2–5 funds, see what % of portfolios overlap. Supports Jaccard (stock count) and weighted (by AUM %) methods. Shows common holdings detail and unique holdings per fund. Sunburst visualization of fund composition.

### 6. Alpha Attribution (Brinson-Hood-Beebower)
Decomposes fund outperformance into:
- **Allocation effect** — did they overweight the right sectors?
- **Selection effect** — did they pick the right stocks?
- **Interaction effect** — combined

## Fund Managers Tracked (10+)

| Manager | AMC | Funds |
|---------|-----|-------|
| Prashant Jain | HDFC MF | Flexi Cap, Balanced Advantage, Top 100 |
| Rajeev Thakkar | PPFAS MF | Flexi Cap, Conservative Hybrid |
| Saurabh Mukherjea | Marcellus | Consistent Compounders |
| Nilesh Shah | Kotak MF | Flexi Cap, Emerging Equity, Small Cap, Bluechip |
| S Naren | ICICI Pru MF | Value Discovery, Bluechip, Midcap, Multicap |
| Neelesh Surana | Mirae Asset MF | Large Cap, Midcap, Flexi Cap |
| Shreyash Devalkar | Axis MF | Bluechip, Midcap, Small Cap, Flexi Cap |
| R Srinivasan | SBI MF | Bluechip, Midcap, Small Cap, Flexi Cap |
| Anish Tawakley | ICICI Pru MF | Flexi Cap |
| Jinesh Gopani | Axis MF | Growth Opportunities |

## Tech Stack

Python | Pandas | Plotly Dash | scikit-learn | pyxirr | yfinance | BeautifulSoup

## Data Sources (all free)

| Source | What you get |
|--------|-------------|
| amfiindia.com | Daily NAV for every mutual fund scheme |
| SEBI portfolio disclosures | Monthly stock-level holdings of every fund |
| yfinance | Historical stock prices for alpha calculation |
| Value Research scrape | AUM, category, fund manager name mapping |
| NSE/BSE | Market cap classification (large/mid/small) per stock |

## Setup

```bash
git clone https://github.com/eeshsaxena/smart-money-tracker.git
cd smart-money-tracker
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8050](http://localhost:8050).

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
smart-money-tracker/
├── app.py                              # Dash entry point
├── config.py                           # Fund managers, thresholds, peer groups
├── requirements.txt
├── src/
│   ├── scraper/
│   │   ├── sebi_scraper.py             # AMFI/SEBI portfolio disclosure scraper
│   │   ├── nse_classifier.py           # NSE/BSE market-cap classification
│   │   └── value_research.py           # Value Research metadata scraper
│   ├── analysis/
│   │   ├── smart_money_flow.py         # Cross-fund accumulation/exit signals
│   │   ├── style_fingerprint.py        # Manager fingerprint + K-Means clustering
│   │   ├── style_drift.py              # SEBI mandate compliance detector
│   │   ├── xirr_calc.py               # SIP XIRR + growth visualization
│   │   ├── overlap.py                  # Portfolio overlap (Jaccard + weighted)
│   │   └── bhb_attribution.py          # Brinson-Hood-Beebower attribution
│   └── dashboard/
│       ├── layout.py                   # 7-tab Dash layout
│       ├── callbacks.py                # All interactive callbacks
│       └── components.py               # Reusable UI components
└── tests/
    ├── test_analysis.py                # 30+ analysis tests
    └── test_scraper.py                 # Scraper + config tests
```

## What Makes It Different

Most tools (Value Research, Morningstar India) show trailing returns. This shows **why** — portfolio construction, manager behavior, and forward-looking signals. Strong for fintech DS roles at Zerodha, Groww, ET Money, INDmoney.
