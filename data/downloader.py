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


def download_stooq():
    """Download deep historical data from multiple free sources.
    Gets DJIA back to 1985 via FRED and BTC via yfinance.
    """
    import yfinance as yf

    out_us30 = DATA_DIR / "us30"
    out_btc = DATA_DIR / "btcusd"
    out_us30.mkdir(parents=True, exist_ok=True)
    out_btc.mkdir(parents=True, exist_ok=True)

    # 1. US30 deep history via FRED (DJIA series goes back to 1985)
    log.info("Downloading US30 (DJIA) deep history from FRED...")
    try:
        if FRED_API_KEY:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            djia = fred.get_series("DJIA")
            if djia is not None and len(djia) > 100:
                df = djia.to_frame(name="close")
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                df["open"] = df["close"]
                df["high"] = df["close"]
                df["low"] = df["close"]
                df["volume"] = 0
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                path = out_us30 / "stooq_1d.parquet"
                df.to_parquet(path)
                log.info(f"  FRED DJIA: Saved {len(df)} rows -> {path} ({df.index.min()} to {df.index.max()})")
    except Exception as e:
        log.warning(f"  FRED DJIA error: {e}")

    # 2. BTC deep history via yfinance (goes back to 2014)
    log.info("Downloading BTCUSD deep history from yfinance...")
    try:
        df = yf.Ticker("BTC-USD").history(period="max", interval="1d")
        if not df.empty:
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            path = out_btc / "stooq_1d.parquet"
            df.to_parquet(path)
            log.info(f"  yfinance BTC: Saved {len(df)} rows -> {path} ({df.index.min()} to {df.index.max()})")
    except Exception as e:
        log.warning(f"  yfinance BTC error: {e}")


def download_btcusd_ccxt():
    """Download BTCUSD via ccxt (tries Binance, falls back to Binance.US, then KuCoin)."""
    import ccxt

    out_dir = DATA_DIR / "btcusd"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple exchanges in case one is geo-blocked (e.g. Binance on US Colab servers)
    exchange = None
    for ex_name, ex_symbol in [("binance", BTCUSD_CCXT_SYMBOL), ("binanceus", "BTC/USD"), ("kucoin", BTCUSD_CCXT_SYMBOL)]:
        try:
            ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
            ex.fetch_ohlcv(ex_symbol, "1d", limit=1)
            exchange = ex
            symbol = ex_symbol
            log.info(f"Using exchange: {ex_name}")
            break
        except Exception as e:
            log.warning(f"  {ex_name} not available: {e}")
            continue

    if exchange is None:
        log.error("No crypto exchange available, skipping BTCUSD download")
        return

    for tf in BTCUSD_TIMEFRAMES:
        log.info(f"Downloading BTCUSD {tf}...")
        all_candles = []
        since = exchange.parse8601("2017-01-01T00:00:00Z")
        limit = 1000

        while True:
            try:
                candles = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
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


def download_forex_factory_calendar():
    """
    Download economic calendar events (Forex Factory style).
    Uses Finnhub free API for economic calendar data with impact levels.
    Falls back to building calendar from FRED release dates.
    """
    import requests
    from config.settings import FINNHUB_API_KEY

    out_dir = DATA_DIR / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []

    # Method 1: Finnhub economic calendar (free, 60 calls/min)
    if FINNHUB_API_KEY:
        log.info("Downloading economic calendar from Finnhub...")
        # Fetch year by year to get maximum history
        for year in range(2015, 2027):
            for quarter_start, quarter_end in [
                (f"{year}-01-01", f"{year}-03-31"),
                (f"{year}-04-01", f"{year}-06-30"),
                (f"{year}-07-01", f"{year}-09-30"),
                (f"{year}-10-01", f"{year}-12-31"),
            ]:
                try:
                    url = "https://finnhub.io/api/v1/calendar/economic"
                    params = {
                        "from": quarter_start,
                        "to": quarter_end,
                        "token": FINNHUB_API_KEY,
                    }
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        events = data.get("economicCalendar", [])
                        all_events.extend(events)
                        log.info(f"  {quarter_start} to {quarter_end}: {len(events)} events")
                    else:
                        log.warning(f"  Finnhub {resp.status_code} for {quarter_start}")
                    time.sleep(1)  # Rate limit
                except Exception as e:
                    log.warning(f"  Finnhub error: {e}")
                    time.sleep(2)

        if all_events:
            df = pd.DataFrame(all_events)
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
            elif "date" in df.columns:
                df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            path = out_dir / "economic_calendar.parquet"
            df.to_parquet(path)
            log.info(f"  Saved {len(df)} economic events -> {path}")
            return

    # Method 2: Build calendar from FRED data release dates + major events
    log.info("Building economic event calendar from FRED data patterns...")

    # Create a calendar of major recurring events based on FRED data changes
    event_data = []
    econ_dir = DATA_DIR / "economic"
    if econ_dir.exists():
        for pfile in econ_dir.glob("*.parquet"):
            series_name = pfile.stem.upper()
            df = pd.read_parquet(pfile)
            df = df.dropna()

            # Each data point change = an economic release event
            for i in range(1, len(df)):
                prev_val = df.iloc[i - 1, 0]
                curr_val = df.iloc[i, 0]
                change = curr_val - prev_val if pd.notna(prev_val) and pd.notna(curr_val) else 0

                impact = "low"
                if series_name in ("DFF", "CPIAUCSL", "UNRATE"):
                    impact = "high"
                elif series_name in ("T10Y2Y", "VIXCLS"):
                    impact = "medium"

                event_data.append({
                    "event": f"{series_name}_RELEASE",
                    "country": "US",
                    "impact": impact,
                    "actual": curr_val,
                    "previous": prev_val,
                    "change": change,
                    "datetime": df.index[i],
                })

    if event_data:
        df = pd.DataFrame(event_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        path = out_dir / "economic_calendar.parquet"
        df.to_parquet(path)
        log.info(f"  Built {len(df)} events from FRED release dates -> {path}")
    else:
        log.warning("  No economic data available to build calendar")


def download_all():
    """Download all market data."""
    log.info("=" * 60)
    log.info("Starting full data download")
    log.info("=" * 60)

    log.info("\n--- US30 from yfinance ---")
    download_us30_yfinance()

    log.info("\n--- Historical data from Stooq ---")
    download_stooq()

    log.info("\n--- BTCUSD from Binance ---")
    download_btcusd_ccxt()

    log.info("\n--- US30 from Alpaca ---")
    download_us30_alpaca()

    log.info("\n--- Economic data from FRED ---")
    download_fred()

    log.info("\n--- Economic calendar (Forex Factory style) ---")
    download_forex_factory_calendar()

    log.info("\n" + "=" * 60)
    log.info("Data download complete!")

    # Summary
    for subdir in ["us30", "btcusd", "economic", "news"]:
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
