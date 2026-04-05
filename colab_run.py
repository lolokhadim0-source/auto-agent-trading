"""One-shot Colab runner: downloads data, builds news, runs agent. No indentation issues."""
import os, sys

os.chdir('/content/auto-agent-trading')
sys.path.insert(0, '.')

# Set API keys
os.environ['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY', '')
os.environ['TELEGRAM_BOT_TOKEN'] = os.environ.get('TELEGRAM_BOT_TOKEN', '')
os.environ['TELEGRAM_CHAT_ID'] = os.environ.get('TELEGRAM_CHAT_ID', '')
os.environ['ALPACA_API_KEY'] = os.environ.get('ALPACA_API_KEY', '')
os.environ['ALPACA_SECRET_KEY'] = os.environ.get('ALPACA_SECRET_KEY', '')
os.environ['FRED_API_KEY'] = os.environ.get('FRED_API_KEY', '')

# Copy Binance BTC data from repo (pre-downloaded, bypasses geo-block)
import shutil
btc_repo = Path('data/btcusd_repo')
btc_out = Path('data/btcusd')
btc_out.mkdir(parents=True, exist_ok=True)
if btc_repo.exists():
    for f in btc_repo.glob('*.parquet'):
        if 'part' not in f.name:
            shutil.copy2(f, btc_out / f.name)
            print(f'Copied Binance {f.name}')
    # Merge split 1m files
    part1 = btc_repo / 'binance_1m_part1.parquet'
    part2 = btc_repo / 'binance_1m_part2.parquet'
    if part1.exists() and part2.exists():
        df1 = pd.read_parquet(part1)
        df2 = pd.read_parquet(part2)
        merged = pd.concat([df1, df2]).sort_index()
        merged.to_parquet(btc_out / 'binance_1m.parquet')
        print(f'Merged Binance 1m: {len(merged)} rows (full 2017-present)')
        del df1, df2, merged

# Download US30 + Stooq historical
from data.downloader import download_us30_yfinance, download_us30_alpaca, download_fred, download_stooq
download_us30_yfinance()
download_stooq()
download_us30_alpaca()
download_fred()

# Download BTC from yfinance
import yfinance as yf
import pandas as pd
from pathlib import Path

out = Path('data/btcusd')
out.mkdir(parents=True, exist_ok=True)
for tf, p in [('1d', 'max'), ('1wk', 'max'), ('1mo', 'max'), ('1h', '2y'), ('5m', '60d')]:
    df = yf.Ticker('BTC-USD').history(period=p, interval=tf)
    if not df.empty:
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
        df.to_parquet(out / f'yfinance_{tf}.parquet')
        print(f'BTC {tf}: {len(df)} rows')

# Build news calendar
news_dir = Path('data/news')
news_dir.mkdir(exist_ok=True)
econ_dir = Path('data/economic')
events = []
for f in econ_dir.glob('*.parquet'):
    d = pd.read_parquet(f).dropna()
    name = f.stem.upper()
    impact = 'high' if name in ('DFF', 'CPIAUCSL', 'UNRATE') else 'medium'
    for i in range(1, len(d)):
        events.append({
            'event': name + '_RELEASE',
            'country': 'US',
            'impact': impact,
            'actual': float(d.iloc[i, 0]),
            'previous': float(d.iloc[i - 1, 0]),
            'change': float(d.iloc[i, 0] - d.iloc[i - 1, 0]),
            'datetime': str(d.index[i]),
        })
cal = pd.DataFrame(events)
cal['datetime'] = pd.to_datetime(cal['datetime'])
cal = cal.set_index('datetime').sort_index()
cal.to_parquet('data/news/economic_calendar.parquet')
print(f'News calendar: {len(cal)} events')

# Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"}')
print('ALL DATA READY - Starting agent!')

# Run agent
os.makedirs('jobs', exist_ok=True)
from agent.agent import run
run(max_iterations=50, notify=True)
