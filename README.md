# option_put_guide

A small research tool that scans S&P 500 names, filters for weekly options and
earnings windows, and highlights tickers that sit below a rising regression
channel. It produces a CSV and an HTML report with channel plots.

This is for educational research only and is not financial advice.

## What it does
- Pulls the S&P 500 constituents list from Wikipedia.
- Fetches recent price history from Yahoo Finance via `yfinance`.
- Builds a linear regression channel and filters for:
  - price below the mean by a configurable standard deviation
  - positive (or optional negative) slope
  - weekly options availability
  - earnings exclusion window
- Generates:
  - `filtered_sp500.csv`
  - `report.html`
  - plots in `plots/`

Note: generated outputs are gitignored by default.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/filter_sp500.py
```

Open the report:
```bash
python3 -m http.server 8000
# then visit http://localhost:8000/report.html
```

## Run helper
There is a convenience runner that creates the venv, installs requirements,
and runs the main script based on `run.ini`:
```bash
./run
```

## Configuration
Edit the user settings near the top of `scripts/filter_sp500.py`:
- `MAX_PRICE`
- `REQUIRE_WEEKLY_OPTIONS`
- `EXCLUDE_EARNINGS_WITHIN_WEEKS`
- `MIN_ABS_SLOPE_PER_DAY`
- `INCLUDE_NEGATIVE_SLOPES`
- `MIN_STD_DEV_BELOW_MEAN`
- `LOOKBACK_WEEKS`
- `CHANNEL_STD_DEV`
- `WRITE_CSV`, `WRITE_HTML`, `EMBED_PLOTS`

## Offline / cached mode
If you already have a `filtered_sp500.csv` and want to rebuild the HTML
report without network calls:
```bash
SP500_OFFLINE=1 python scripts/filter_sp500.py
```

If network fetches fail, the script can fall back to the cached CSV when
`USE_CACHED_RESULTS_ON_NETWORK_FAILURE = True`.

## Notes
- This project makes network calls to Wikipedia and Yahoo Finance.
- `plots/`, `report.html`, and `filtered_sp500.csv` are ignored by git.
