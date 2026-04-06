import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = BASE_DIR / "jobs"
STRATEGY_DIR = BASE_DIR / "strategy"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# US30/DJIA config
US30_YFINANCE_SYMBOL = "^DJI"
US30_TIMEFRAMES_YFINANCE = ["1d", "1wk", "1mo"]
US30_ALPACA_SYMBOL = "DIA"  # DJIA ETF (Alpaca doesn't have index directly)
US30_TIMEFRAMES_ALPACA = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

# BTCUSD config
BTCUSD_CCXT_SYMBOL = "BTC/USDT"
BTCUSD_EXCHANGE = "binance"
BTCUSD_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# FRED economic indicators
FRED_SERIES = [
    "DFF",       # Federal Funds Rate
    "T10Y2Y",    # 10Y-2Y Treasury Spread
    "VIXCLS",    # VIX
    "DTWEXBGS",  # Trade-weighted USD index
    "UNRATE",    # Unemployment rate
    "CPIAUCSL",  # CPI
]

# Experiment settings
EXPERIMENT_TIMEOUT_SECONDS = 300  # 5 minutes per backtest run
MAX_INNER_ITERATIONS = 100
OUTER_LOOP_EVERY_N = 10  # Run meta-agent every N inner iterations

# LLM settings
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 16000  # 4096 was too small — LLM truncated code causing SyntaxErrors
