# Research Directives

## Objective
Optimize strategy/train.py to PERFECTION. The strategy must work across ALL timeframes on both US30 and BTCUSD, connecting patterns across years of data, news events, and economic cycles. No errors, no skipped data, NOTHING missing.

## Scoring
Composite score = weighted average of: Sharpe (30%), Total Return (25%), Max Drawdown (20%), Profit Factor (15%), Win Rate (10%).

## ALL 9 COMBINED STRATEGIES — EVERY ONE MUST BE PRESENT:
The strategy MUST combine ALL 9 of these. Do NOT remove any. Improve them, tune weights, but keep ALL:

1. **Trend Following** (15%) - SMA/EMA crossover + ADX filter
2. **Mean Reversion** (10%) - Bollinger Bands + RSI extremes
3. **Breakout** (10%) - Donchian Channel breaks
4. **Momentum** (15%) - MACD + Stochastic
5. **Volume** (5%) - OBV divergence + volume spikes
6. **ICT Smart Money** (20%) - FVG, BOS, liquidity sweeps, order blocks, displacement
7. **GPU ML** (15%) - LSTM trained on A100 GPU using train_model/predict_signals
8. **Economic Regime** (5%) - VIX, Fed rate, CPI, unemployment, Treasury spread from context data
9. **Cross-Year Pattern** (5%) - Compare current month/quarter price action to same period in prior years. Detect recurring seasonal patterns, year-over-year momentum, and anniversary reactions to major events.

## Available ICT Indicators (strategy/indicators.py):
```python
from strategy.indicators import fair_value_gap, order_blocks, break_of_structure, liquidity_sweep, displacement
```
- `fair_value_gap(df)` -> 1=bullish FVG, -1=bearish FVG
- `order_blocks(df, lookback)` -> 1=at bullish OB, -1=at bearish OB
- `break_of_structure(df, period)` -> 1=bullish BOS, -1=bearish BOS
- `liquidity_sweep(df, period)` -> 1=bullish sweep (buy), -1=bearish sweep (sell)
- `displacement(df, threshold)` -> 1=bullish displacement, -1=bearish displacement

## Available ML Model (strategy/ml_model.py):
```python
from strategy.ml_model import train_model, predict_signals
model = train_model(df, lookback=60, epochs=10)  # GPU accelerated on A100
ml_signals = predict_signals(model, df, lookback=60)  # 1=long, -1=short, 0=flat
```

## Available Economic Context (second argument):
```python
def strategy(df, context=None):
    context["fred_dff"]      # Fed Funds Rate (daily, decades)
    context["fred_vixcls"]   # VIX volatility index (daily)
    context["fred_t10y2y"]   # 10Y-2Y Treasury Spread (daily)
    context["fred_cpiaucsl"] # CPI inflation (monthly)
    context["fred_unrate"]   # Unemployment Rate (monthly)
    context["fred_dtwexbgs"] # Trade-weighted USD Index (daily)
    context["economic_calendar"]  # 54K+ news events with impact levels, columns: event, country, impact, actual, previous, change
    context["market"]        # "us30" or "btcusd"
    context["timeframe"]     # current timeframe string
```
Use `pd.merge_asof()` or `.reindex()` to align economic data with price bars.

## Available Data Sources:
- **Alpaca**: US30 intraday (1Min, 5Min, 15Min, 1Hour, 1Day) from 2019
- **yfinance**: US30 daily/weekly/monthly back to 1992, BTC from 2014
- **Stooq**: US30 (DJIA) daily back to 1985, BTC daily back to 2010
- **Binance**: BTCUSD (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w) from 2017
- **FRED**: VIX, Fed Funds, CPI, Unemployment, Treasury spread, USD index
- **Economic Calendar**: 54K+ events with impact levels (high/medium/low)

## CRITICAL RULES FOR THE AGENT:
1. Strategy MUST return `pd.Series` (1D) — use `.squeeze()` and `.fillna(0)` at the end
2. Strategy MUST work on ALL timeframes (1m to 1mo) — don't hardcode periods
3. Strategy MUST accept `context` dict as optional second arg
4. Strategy MUST use `train_model()` for GPU ML — we're paying for A100
5. Strategy MUST connect patterns across the full history — use long lookback periods
6. **ALL 9 STRATEGIES MUST REMAIN** — do NOT simplify or remove any strategy. You may tune weights, improve logic within each, but all 9 must generate signals.
7. Strategy MUST use economic context data (FRED + calendar) — connect news events to price action
8. Strategy MUST use ICT concepts (at least FVG + BOS + liquidity sweeps)
9. Make ONE focused change per iteration — improve one strategy at a time, don't rewrite everything
10. The composite scoring rewards consistency across ALL timeframes — a strategy that works on 1d but fails on 1m is bad

## WHAT TO OPTIMIZE:
- Tune the weights between 9 strategies (find optimal allocation)
- Improve ICT detection (order blocks, FVG, liquidity sweeps)
- Add Wyckoff accumulation/distribution detection
- Add market session awareness (Asian/London/NY for US30, weekend gaps for BTC)
- Add multi-timeframe confirmation logic
- Improve ML model features (add indicators as input features)
- Add dynamic regime detection (trending vs ranging vs choppy)
- Correlate economic events with price reactions (e.g., reduce position before FOMC)
- Reduce false signals — higher conviction entries only
- Cross-year seasonal analysis (Q4 rally patterns, January effect, summer doldrums)
- Connect BTC halving cycles to price momentum
- Use VIX as volatility regime filter (low VIX = trend, high VIX = mean revert)
