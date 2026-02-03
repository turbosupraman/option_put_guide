#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, datetime, timedelta
import base64
from html import escape
from io import StringIO
import os
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- User-editable filters and settings ---
MAX_PRICE = 250.0
REQUIRE_WEEKLY_OPTIONS = True
EXCLUDE_EARNINGS_WITHIN_WEEKS = 4
LIMIT_TICKERS = None  # Set an int for quick testing.

MIN_ABS_SLOPE_PER_DAY = 0.0  # Skip near-flat channels.
INCLUDE_NEGATIVE_SLOPES = False
MIN_STD_DEV_BELOW_MEAN = 1.0  # Require current price <= mean - (N * std).

LOOKBACK_WEEKS = 9
LOOKBACK_DAYS = LOOKBACK_WEEKS * 7
FUTURE_EXTENSION_MONTHS = 2
CHANNEL_STD_DEV = 2.0

PLOT_DIR = "plots"
WRITE_CSV = True
CSV_OUTPUT = "filtered_sp500.csv"
WRITE_HTML = True
HTML_OUTPUT = "report.html"
EMBED_PLOTS = True
USE_CACHED_RESULTS_ON_NETWORK_FAILURE = True
OFFLINE_ENV_VAR = "SP500_OFFLINE"

YF_CACHE_DIR = ".cache/yfinance"
YF_THREADS = False
PRICE_BATCH_SIZE = 50

HOURLY_INTERVAL = "1h"
HOURLY_PERIOD_DAYS = LOOKBACK_DAYS
MIN_HOURLY_COVERAGE_DAYS = LOOKBACK_DAYS - 3

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def configure_yfinance() -> None:
    cache_dir = Path(YF_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except Exception as exc:
        print(f"Warning: could not set yfinance cache location: {exc}")


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def normalize_prices(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return prices

    prices = prices.sort_index()
    if getattr(prices.index, "tz", None) is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def extract_close_series(history: pd.DataFrame | pd.Series, yf_symbol: str) -> pd.Series:
    if history is None:
        return pd.Series(dtype=float)
    if isinstance(history, pd.Series):
        return history
    if history.empty:
        return pd.Series(dtype=float)

    if isinstance(history.columns, pd.MultiIndex):
        level0 = history.columns.get_level_values(0)
        level1 = history.columns.get_level_values(1)

        if yf_symbol in level0:
            try:
                close = history[yf_symbol]["Close"]
                if isinstance(close, pd.Series):
                    return close
            except Exception:
                pass

        if "Close" in level0:
            close = history["Close"]
            if isinstance(close, pd.DataFrame):
                if yf_symbol in close.columns:
                    return close[yf_symbol]
                if close.shape[1] == 1:
                    return close.iloc[:, 0]
            elif isinstance(close, pd.Series):
                return close

        if "Close" in level1:
            close = history.xs("Close", level=1, axis=1)
            if isinstance(close, pd.DataFrame):
                if yf_symbol in close.columns:
                    return close[yf_symbol]
                if close.shape[1] == 1:
                    return close.iloc[:, 0]
            elif isinstance(close, pd.Series):
                return close

        if history.shape[1] == 1:
            return history.iloc[:, 0]

        return pd.Series(dtype=float)

    if "Close" in history.columns:
        close = history["Close"]
        if isinstance(close, pd.Series):
            return close
        if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
            return close.iloc[:, 0]
    if history.shape[1] == 1:
        return history.iloc[:, 0]
    return pd.Series(dtype=float)


def fetch_sp500_table() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-screener/1.0)"}
    response = requests.get(WIKI_URL, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text

    try:
        tables = pd.read_html(StringIO(html), attrs={"id": "constituents"})
        df = tables[0]
    except ValueError:
        tables = pd.read_html(StringIO(html))
        df = tables[0]

    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Security": "name",
        }
    )
    return df[["symbol", "name"]].copy()


def to_yf_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def download_latest_prices(yf_symbols: Iterable[str]) -> dict[str, float]:
    symbols = [symbol for symbol in yf_symbols if isinstance(symbol, str) and symbol]
    if not symbols:
        return {}

    prices: dict[str, float] = {}
    for batch in chunked(symbols, PRICE_BATCH_SIZE):
        try:
            data = yf.download(
                tickers=batch,
                period="1d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=YF_THREADS,
                progress=False,
            )
        except Exception as exc:
            print(f"Warning: price download failed for batch {batch[:3]}: {exc}")
            continue

        if data is None or data.empty:
            continue

        for symbol in batch:
            close_series = extract_close_series(data, symbol).dropna()
            if not close_series.empty:
                prices[symbol] = float(close_series.iloc[-1])
    return prices


def third_friday(year: int, month: int) -> date:
    first_day = date(year, month, 1)
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    return first_friday + timedelta(days=14)


def has_weekly_options(yf_symbol: str) -> bool:
    try:
        expirations = yf.Ticker(yf_symbol).options
    except Exception:
        return False

    if not expirations:
        return False

    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            continue
        if exp_date != third_friday(exp_date.year, exp_date.month):
            return True
    return False


def coerce_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0:
            return None
        parsed = parsed[0]
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime().date()
    if isinstance(parsed, datetime):
        return parsed.date()
    return None


def get_next_earnings_date(yf_symbol: str) -> date | None:
    today = date.today()
    ticker = yf.Ticker(yf_symbol)
    candidates: list[date] = []

    try:
        earnings_df = ticker.get_earnings_dates(limit=8)
    except Exception:
        earnings_df = None

    if isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty:
        for ts in earnings_df.index:
            earnings_date = coerce_date(ts)
            if earnings_date and earnings_date >= today:
                candidates.append(earnings_date)

    if not candidates:
        try:
            calendar = ticker.calendar
        except Exception:
            calendar = {}

        raw_dates = calendar.get("Earnings Date")
        if raw_dates is not None:
            if isinstance(raw_dates, (list, tuple, pd.Series, pd.Index, np.ndarray)):
                values = raw_dates
            else:
                values = [raw_dates]
            for value in values:
                earnings_date = coerce_date(value)
                if earnings_date and earnings_date >= today:
                    candidates.append(earnings_date)

    if not candidates:
        return None
    return min(candidates)


def fetch_price_history(yf_symbol: str) -> tuple[pd.Series, str]:
    try:
        hourly = yf.download(
            tickers=yf_symbol,
            period=f"{HOURLY_PERIOD_DAYS}d",
            interval=HOURLY_INTERVAL,
            auto_adjust=False,
            threads=YF_THREADS,
            progress=False,
        )
    except Exception:
        hourly = pd.DataFrame()

    hourly_close = normalize_prices(extract_close_series(hourly, yf_symbol))
    if not hourly_close.empty:
        span_days = (hourly_close.index.max() - hourly_close.index.min()).days
        if span_days >= MIN_HOURLY_COVERAGE_DAYS:
            return hourly_close, HOURLY_INTERVAL

    try:
        daily = yf.download(
            tickers=yf_symbol,
            period=f"{LOOKBACK_DAYS}d",
            interval="1d",
            auto_adjust=False,
            threads=YF_THREADS,
            progress=False,
        )
    except Exception:
        daily = pd.DataFrame()

    daily_close = normalize_prices(extract_close_series(daily, yf_symbol))
    return daily_close, "1d"


def regression_channel(prices: pd.Series) -> dict[str, np.ndarray] | None:
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            return None

    prices = prices.dropna()
    if len(prices) < 5:
        return None

    index = prices.index
    if getattr(index, "tz", None) is not None:
        prices = prices.copy()
        prices.index = index.tz_localize(None)
        index = prices.index

    x = np.asarray((index - index[0]).total_seconds() / 86400, dtype=float)
    y = prices.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    residuals = y - predicted
    std_dev = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0

    step_days = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    future_days = FUTURE_EXTENSION_MONTHS * 30
    x_extended = np.arange(x[0], x[-1] + future_days + step_days, step_days)
    dates_extended = index[0] + pd.to_timedelta(x_extended, unit="D")

    predicted_extended = slope * x_extended + intercept
    upper = predicted_extended + CHANNEL_STD_DEV * std_dev
    lower = predicted_extended - CHANNEL_STD_DEV * std_dev

    predicted_current = float(predicted[-1])
    return {
        "dates": dates_extended,
        "predicted": predicted_extended,
        "upper": upper,
        "lower": lower,
        "slope_per_day": float(slope),
        "std_dev": float(std_dev),
        "predicted_current": predicted_current,
    }


def safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_").replace("/", "_")


def offline_mode_enabled() -> bool:
    value = os.getenv(OFFLINE_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_cached_results(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: could not read cached CSV {csv_path}: {exc}")
        return None

    required = {
        "symbol",
        "name",
        "price",
        "weekly_options",
        "slope_per_day",
        "std_from_mean",
        "interval",
        "plot_path",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: cached CSV missing columns: {', '.join(sorted(missing))}")
        return None

    for column in ("price", "slope_per_day", "std_from_mean"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_html_report(results_df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    criteria = [
        f"Max price: ${MAX_PRICE:.2f}",
        f"Weekly options required: {REQUIRE_WEEKLY_OPTIONS}",
        f"Exclude earnings within: {EXCLUDE_EARNINGS_WITHIN_WEEKS} weeks",
        f"Min abs slope/day: {MIN_ABS_SLOPE_PER_DAY}",
        f"Min std below mean at current date: {MIN_STD_DEV_BELOW_MEAN}",
        f"Lookback: {LOOKBACK_WEEKS} weeks",
        f"Forward extension: {FUTURE_EXTENSION_MONTHS} months",
    ]

    table_rows = []
    for _, row in results_df.iterrows():
        symbol = str(row["symbol"])
        table_rows.append(
            "<tr>"
            f"<td><a href='#{escape(symbol)}'>{escape(symbol)}</a></td>"
            f"<td>${row['price']:.2f}</td>"
            f"<td>{row['slope_per_day']:.4f}</td>"
            f"<td>{row['std_from_mean']:.2f}</td>"
            f"<td>{escape(str(row['interval']))}</td>"
            f"<td>{escape(str(row['weekly_options']))}</td>"
            f"<td>{escape(str(row['name']))}</td>"
            "</tr>"
        )

    chart_blocks = []
    for _, row in results_df.iterrows():
        symbol = str(row["symbol"])
        name = str(row["name"])
        plot_path = Path(row["plot_path"])
        if EMBED_PLOTS and plot_path.exists():
            with plot_path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            img_src = f"data:image/png;base64,{encoded}"
        else:
            img_src = escape(str(plot_path))

        chart_blocks.append(
            "<section class='card' id='{id}'>"
            "<h2>{symbol} <span class='sub'>{name}</span></h2>"
            "<p class='meta'>Price: ${price:.2f} | "
            "Slope/day: {slope:.4f} | Std from mean: {std_from_mean:.2f} | "
            "Interval: {interval}</p>"
            "<img src='{img_src}' alt='Regression channel for {symbol}' />"
            "</section>".format(
                id=escape(symbol),
                symbol=escape(symbol),
                name=escape(name),
                price=row["price"],
                slope=row["slope_per_day"],
                std_from_mean=row["std_from_mean"],
                interval=escape(str(row["interval"])),
                img_src=img_src,
            )
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Regression Channels</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 0; }}
    .subhead {{ color: #555; margin-top: 4px; }}
    .criteria {{ margin: 16px 0; padding: 0; list-style: none; }}
    .criteria li {{ margin: 4px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    .card {{ margin: 24px 0; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }}
    .card img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    .sub {{ color: #666; font-size: 0.9em; }}
    .meta {{ color: #444; }}
  </style>
</head>
<body>
  <h1>Stock Regression Channels</h1>
  <p class="subhead">Generated {escape(timestamp)}. Slope based on {LOOKBACK_WEEKS} week regression.</p>
  <ul class="criteria">
    {''.join(f'<li>{escape(item)}</li>' for item in criteria)}
  </ul>
  {"<p>No matches found.</p>" if results_df.empty else ""}
  {"<table><thead><tr><th>Symbol</th><th>Price</th><th>Slope/Day</th><th>Std From Mean</th><th>Interval</th><th>Weekly</th><th>Name</th></tr></thead><tbody>"
   + ''.join(table_rows) + "</tbody></table>" if not results_df.empty else ""}
  {''.join(chart_blocks)}
</body>
</html>
"""
    return html


def plot_channel(
    symbol: str,
    name: str,
    prices: pd.Series,
    channel: dict[str, np.ndarray],
    interval_label: str,
) -> Path:
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    slope_per_day = float(channel["slope_per_day"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prices.index, prices.to_numpy(), color="black", linewidth=1, label="Close")
    ax.plot(
        channel["dates"],
        channel["predicted"],
        color="tab:blue",
        linewidth=1.5,
        label="Regression",
    )
    ax.fill_between(
        channel["dates"],
        channel["lower"],
        channel["upper"],
        color="tab:blue",
        alpha=0.2,
        label=f"Channel (+/-{CHANNEL_STD_DEV} sd)",
    )
    ax.axvline(prices.index[-1], color="gray", linestyle="--", linewidth=1)
    ax.set_title(
        f"{symbol} - {name}\nSlope: {slope_per_day:.3f} $/day | Interval: {interval_label}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    fig.tight_layout()

    plot_path = Path(PLOT_DIR) / f"{safe_filename(symbol)}_channel.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main() -> int:
    configure_yfinance()
    cached_path = Path(CSV_OUTPUT)
    if offline_mode_enabled():
        cached = load_cached_results(cached_path)
        if cached is None:
            print(f"Offline mode enabled but cached CSV not found: {cached_path}")
            return 1
        if WRITE_HTML:
            html_output = build_html_report(cached)
            Path(HTML_OUTPUT).write_text(html_output, encoding="utf-8")
            print(f"Wrote {HTML_OUTPUT} (from cached CSV)")
        return 0

    try:
        sp500 = fetch_sp500_table()
    except requests.RequestException as exc:
        cached = None
        if USE_CACHED_RESULTS_ON_NETWORK_FAILURE:
            cached = load_cached_results(cached_path)
        if cached is not None:
            print(f"Warning: network fetch failed ({exc}). Using cached {cached_path}.")
            if WRITE_HTML:
                html_output = build_html_report(cached)
                Path(HTML_OUTPUT).write_text(html_output, encoding="utf-8")
                print(f"Wrote {HTML_OUTPUT} (from cached CSV)")
            return 0
        raise
    sp500["yf_symbol"] = sp500["symbol"].map(to_yf_symbol)

    if LIMIT_TICKERS:
        sp500 = sp500.head(LIMIT_TICKERS).copy()

    prices = download_latest_prices(sp500["yf_symbol"])
    sp500["price"] = sp500["yf_symbol"].map(prices)

    filtered = sp500[sp500["price"].notna()]
    filtered = filtered[filtered["price"] <= MAX_PRICE].copy()

    if REQUIRE_WEEKLY_OPTIONS:
        weekly_flags = []
        for yf_symbol in filtered["yf_symbol"]:
            weekly_flags.append(has_weekly_options(yf_symbol))
        filtered["weekly_options"] = weekly_flags
        filtered = filtered[filtered["weekly_options"]].copy()
    else:
        filtered["weekly_options"] = False

    if EXCLUDE_EARNINGS_WITHIN_WEEKS and EXCLUDE_EARNINGS_WITHIN_WEEKS > 0:
        cutoff = date.today() + timedelta(weeks=EXCLUDE_EARNINGS_WITHIN_WEEKS)
        keep_mask = []
        for yf_symbol in filtered["yf_symbol"]:
            next_earnings = get_next_earnings_date(yf_symbol)
            keep_mask.append(next_earnings is None or next_earnings > cutoff)
        filtered = filtered.loc[keep_mask].copy()

    filtered = filtered.sort_values(by="price")

    results = []
    for _, row in filtered.iterrows():
        price_series, interval_label = fetch_price_history(row["yf_symbol"])
        if price_series.empty:
            continue
        channel = regression_channel(price_series)
        if not channel:
            continue

        slope_per_day = float(channel["slope_per_day"])
        if abs(slope_per_day) < MIN_ABS_SLOPE_PER_DAY:
            continue
        if not INCLUDE_NEGATIVE_SLOPES and slope_per_day <= 0:
            continue

        current_price = float(price_series.iloc[-1])
        if current_price > MAX_PRICE:
            continue
        std_dev = float(channel["std_dev"])
        if std_dev <= 0:
            continue
        predicted_current = float(channel["predicted_current"])
        std_from_mean = (current_price - predicted_current) / std_dev
        if std_from_mean > -MIN_STD_DEV_BELOW_MEAN:
            continue

        plot_path = plot_channel(
            row["symbol"],
            row["name"],
            price_series,
            channel,
            interval_label,
        )
        results.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "price": current_price,
                "weekly_options": row["weekly_options"],
                "slope_per_day": slope_per_day,
                "std_from_mean": std_from_mean,
                "interval": interval_label,
                "plot_path": str(plot_path),
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="price")

    if results_df.empty:
        print("No matches found.")
    else:
        print(
            results_df[
                [
                    "symbol",
                    "price",
                    "slope_per_day",
                    "std_from_mean",
                    "interval",
                    "weekly_options",
                    "plot_path",
                    "name",
                ]
            ].to_string(index=False)
        )

    if WRITE_CSV and not results_df.empty:
        results_df.to_csv(CSV_OUTPUT, index=False)
        print(f"Wrote {CSV_OUTPUT}")
    if WRITE_HTML:
        html_output = build_html_report(results_df)
        Path(HTML_OUTPUT).write_text(html_output, encoding="utf-8")
        print(f"Wrote {HTML_OUTPUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
