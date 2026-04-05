# Research Directives

## Objective
Optimize strategy/train.py to PERFECTION. The strategy must work across ALL timeframes on both US30 and BTCUSD, connecting patterns across years of data, news events, and economic cycles. No errors, no skipped data.

## Scoring
Composite score = weighted average of: Sharpe (30%), Total Return (25%), Max Drawdown (20%), Profit Factor (15%), Win Rate (10%).

## CURRENT BASELINE: 8 Combined Strategies
The baseline strategy already combines these - improve each one:
1. **Trend Following** (15%) - SMA/EMA crossover + ADX filter
2. **Mean Reversion** (10%) - Bollinger Bands + RSI extremes
3. **Breakout** (10%) - Donchian Channel breaks
4. **Momentum** (15%) - MACD + Stochastic
5. **Volume** (5%) - OBV divergence + volume spikes
6. **ICT Smart Money** (20%) - FVG, BOS, liquidity sweeps, order blocks, displacement
7. **GPU ML** (15%) - LSTM trained on A100 GPU
8. **Economic Regime** - VIX + Fed rate from context data

## Available ICT Indicators (strategy/indicators.py):
```python
from strategy.indicators import fair_value_gap, order_blocks, break_of_structure, liquidity_sweep, displacement
```
- `fair_value_gap(df)` → 1=bullish FVG, -1=bearish FVG
- `order_blocks(df, lookback)` → 1=at bullish OB, -1=at bearish OB
- `break_of_structure(df, period)` → 1=bullish BOS, -1=bearish BOS
- `liquidity_sweep(df, period)` → 1=bullish sweep (buy), -1=bearish sweep (sell)
- `displacement(df, threshold)` → 1=bullish displacement, -1=bearish displacement

## Available ML Model (strategy/ml_model.py):
```python
from strategy.ml_model import train_model, predict_signals
model = train_model(df, lookback=60, epochs=10)  # GPU accelerated
ml_signals = predict_signals(model, df, lookback=60)  # 1=long, -1=short, 0=flat
```

## Available Economic Context (second argument):
```python
def strategy(df, context=None):
    context["fred_dff"]      # Fed Funds Rate
    context["fred_vixcls"]   # VIX
    context["fred_t10y2y"]   # Treasury Spread
    context["fred_cpiaucsl"] # CPI
    context["fred_unrate"]   # Unemployment
    context["economic_calendar"]  # 54K+ news events with impact levels
    context["market"]        # "us30" or "btcusd"
    context["timeframe"]     # current timeframe string
```

## CRITICAL RULES FOR THE AGENT:
1. Strategy MUST return `pd.Series` (1D) — use `.squeeze()` and `.fillna(0)` at the end
2. Strategy MUST work on ALL timeframes (1m to 1mo) — don't hardcode periods
3. Strategy MUST accept `context` dict as optional second arg
4. Strategy MUST use `train_model()` for GPU ML — we're paying for A100
5. Strategy MUST connect patterns across the full history — use long lookback periods
6. Make ONE focused change per iteration — don't rewrite everything
7. The 10% reserved weight slot is for YOUR improvements — add new signals there

## WHAT TO OPTIMIZE:
- Tune the weights between strategies (currently hardcoded)
- Improve ICT detection (order blocks, FVG, liquidity sweeps)
- Add Wyckoff accumulation/distribution detection
- Add market session awareness (Asian/London/NY)
- Add multi-timeframe confirmation logic
- Improve ML model features (add indicators as input features)
- Add dynamic regime detection (trending vs ranging vs choppy)
- Correlate economic events with price reactions
- Reduce false signals — higher conviction entries only
