# Smart Money Tracker — MF Intelligence Dashboard

Track stock-level holdings of top Indian fund managers, detect accumulation/exit signals, and benchmark your SIP returns.

## Features

- **SEBI Portfolio Scraper** — Fetches monthly portfolio disclosures from AMFI to track holdings of fund managers like Prashant Jain and Rajeev Thakkar
- **Accumulation/Exit Signals** — Detects collective buying/selling patterns across large/mid/small-cap funds
- **Portfolio Overlap Heatmap** — Visualizes Jaccard and weighted overlap between funds
- **Style-Drift Detector** — Flags funds deviating from their SEBI-mandated market-cap category using NSE classification data
- **SIP XIRR Calculator** — Computes annualized returns via XIRR and benchmarks against Nifty 50 and category peers
- **BHB Attribution** — Brinson-Hood-Beebower decomposition of fund alpha into allocation, selection, and interaction effects

## Tech Stack

Python | Pandas | Plotly Dash | pyxirr | yfinance | BeautifulSoup

## Setup

```bash
# Clone
git clone https://github.com/eeshsaxena/smart-money-tracker.git
cd smart-money-tracker

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python app.py
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
smart-money-tracker/
├── app.py                         # Dash application entry point
├── config.py                      # Fund managers, thresholds, constants
├── requirements.txt
├── src/
│   ├── scraper/
│   │   ├── sebi_scraper.py        # AMFI/SEBI portfolio disclosure scraper
│   │   └── nse_classifier.py      # NSE market-cap classification
│   ├── analysis/
│   │   ├── overlap.py             # Portfolio overlap (Jaccard + weighted)
│   │   ├── style_drift.py         # Style-drift detection vs mandates
│   │   ├── xirr_calc.py           # SIP XIRR + benchmark comparison
│   │   └── bhb_attribution.py     # Brinson-Hood-Beebower attribution
│   └── dashboard/
│       ├── layout.py              # Dash UI layout
│       ├── callbacks.py           # Dash callbacks
│       └── components.py          # Reusable UI components
└── tests/
    ├── test_analysis.py
    └── test_scraper.py
```
