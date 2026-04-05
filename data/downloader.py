"""
Market data downloader for US30 (Dow Jones) and BTCUSD.
Sources: yfinance, Binance (ccxt), Alpaca Markets, FRED.
Stores data as Parquet files for fast loading.
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, US30_YFINANCE_SYMBOL, US30_TIMEFRAMES_YFINANCE,
    US30_ALPACA_SYMBOL, US30_TIMEFRAMES_ALPACA,
    BTCUSD_CCXT_SYMBOL, BTCUSD_EXCHANGE, BTCUSD_TIMEFRAMES,
    FRED_SERIES, FRED_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def download_us30_yfinance():
    """Download DJIA data from yfinance (daily/weekly/monthly, back to ~1985)."""
    import yfinance as yf

    out_dir = DATA_DIR / "us30"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in US30_TIMEFRAMES_YFINANCE:
        log.info(f"Downloading US30 {tf} from yfinance...")
        ticker = yf.Ticker(US30_YFINANCE_SYMBOL)
        df = ticker.history(period="max", interval=tf)

        if df.empty:
            log.warning(f"No data for US30 {tf}")
            continue

        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        path = out_dir / f"yfinance_{tf}.parquet"
        df.to_parquet(path)
        log.info(f"  Saved {len(df)} rows -> {path}")


def download_btcusd_ccxt():
    """Download BTCUSD from Binance via ccxt (all timeframes, since 2017)."""
    import ccxt

    out_dir = DATA_DIR / "btcusd"
    out_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})

    for tf in BTCUSD_TIMEFRAMES:
        log.info(f"Downloading BTCUSD {tf} from Binance...")
        all_candles = []
        since = exchange.parse8601("2017-01-01T00:00:00Z")
        limit = 1000

        while True:
            try:
                candles = exchange.fetch_ohlcv(BTCUSD_CCXT_SYMBOL, tf, since=since, limit=limit)
            except Exception as e:
                log.warning(f"  Error fetching {tf}: {e}, retrying in 5s...")
                time.sleep(5)
                continue

            if not candles:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1  # next ms after last candle

            if len(candles) < limit:
                break

            time.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            log.warning(f"No data for BTCUSD {tf}")
            continue

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        path = out_dir / f"binance_{tf}.parquet"
        df.to_parquet(path)
        log.info(f"  Saved {len(df)} rows -> {path}")


def download_us30_alpaca():
    """Download DIA (DJIA ETF) intraday data from Alpaca."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log.warning("Alpaca API keys not set, skipping Alpaca download")
        return

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    out_dir = DATA_DIR / "us30"
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }

    for tf_name in US30_TIMEFRAMES_ALPACA:
        tf = tf_map.get(tf_name)
        if not tf:
            continue

        log.info(f"Downloading US30 ({US30_ALPACA_SYMBOL}) {tf_name} from Alpaca...")
        try:
            request = StockBarsRequest(
                symbol_or_symbols=US30_ALPACA_SYMBOL,
                timeframe=tf,
                start=datetime(2019, 1, 1),
            )
            bars = client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                log.warning(f"No Alpaca data for {tf_name}")
                continue

            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            df = df.rename(columns=str.lower)
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            path = out_dir / f"alpaca_{tf_name}.parquet"
            df.to_parquet(path)
            log.info(f"  Saved {len(df)} rows -> {path}")
        except Exception as e:
            log.error(f"  Alpaca error for {tf_name}: {e}")


def download_fred():
    """Download economic indicators from FRED."""
    if not FRED_API_KEY:
        log.warning("FRED API key not set, skipping FRED download")
        return

    from fredapi import Fred

    out_dir = DATA_DIR / "economic"
    out_dir.mkdir(parents=True, exist_ok=True)

    fred = Fred(api_key=FRED_API_KEY)

    for series_id in FRED_SERIES:
        log.info(f"Downloading FRED series {series_id}...")
        try:
            data = fred.get_series(series_id)
            df = data.to_frame(name=series_id.lower())
            df.index.name = "date"

            path = out_dir / f"{series_id.lower()}.parquet"
            df.to_parquet(path)
            log.info(f"  Saved {len(df)} rows -> {path}")
        except Exception as e:
            log.error(f"  FRED error for {series_id}: {e}")


def download_all():
    """Download all market data."""
    log.info("=" * 60)
    log.info("Starting full data download")
    log.info("=" * 60)

    log.info("\n--- US30 from yfinance ---")
    download_us30_yfinance()

    log.info("\n--- BTCUSD from Binance ---")
    download_btcusd_ccxt()

    log.info("\n--- US30 from Alpaca ---")
    download_us30_alpaca()

    log.info("\n--- Economic data from FRED ---")
    download_fred()

    log.info("\n" + "=" * 60)
    log.info("Data download complete!")

    # Summary
    for subdir in ["us30", "btcusd", "economic"]:
        d = DATA_DIR / subdir
        if d.exists():
            files = list(d.glob("*.parquet"))
            total_rows = 0
            for f in files:
                try:
                    total_rows += len(pd.read_parquet(f))
                except Exception:
                    pass
            log.info(f"  {subdir}: {len(files)} files, {total_rows:,} total rows")


if __name__ == "__main__":
    download_all()
