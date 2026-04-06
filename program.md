# Research Directives — TARGET: SCORE 8-11 (IMPOSSIBLE MODE)

## Objective
Optimize strategy/train.py to achieve composite scores of 8+ across ALL timeframes. The MEDIAN score across all datasets must be maximized. Every dataset matters equally — one bad timeframe tanks everything.

## Scoring Formula (from backtest.py)
```
composite = (
    0.30 * clip(sharpe, -5, 5)              # Max contribution: 1.50
    + 0.25 * clip(total_return * 10, -10, 30) # Max contribution: 7.50 (300%+ return)
    + 0.20 * clip((1 + max_dd) * 5, 0, 5)    # Max contribution: 1.00 (0% drawdown)
    + 0.15 * clip(profit_factor, 0, 5)        # Max contribution: 0.75
    + 0.10 * win_rate * 5                     # Max contribution: 0.50
)
# THEORETICAL MAX = 11.25
```

**Return dominates (7.50 of 11.25)**. Maximize total return first, then protect with low drawdown.

## CRITICAL INSIGHT: TIMEFRAME ADAPTATION
The #1 score killer is using fixed periods across all timeframes. SMA(200) on 5m data = 16 hours. On daily = 10 months. YOU MUST ADAPT:

```python
# Parse timeframe and scale periods
tf = context.get("timeframe", "1d") if context else "1d"
# Determine bar-minutes for scaling
tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
              "1d": 1440, "1Day": 1440, "1wk": 10080, "1mo": 43200,
              "1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}.get(tf, 1440)
# Scale factor: how many bars = 1 day
scale = max(1, 1440 // tf_minutes)
# Now use: sma_fast = scale * 5, sma_slow = scale * 50, etc.
```

This ONE change can add +1 to +2 points because it fixes broken signals on every non-daily timeframe.

## ALL 9 COMBINED STRATEGIES — EVERY ONE MUST BE PRESENT:

1. **Trend Following** (15%) - SMA/EMA crossover + ADX filter. ADAPT periods to timeframe.
2. **Mean Reversion** (10%) - Bollinger Bands + RSI extremes. Only when ADX < 25.
3. **Breakout** (10%) - Donchian Channel breaks. ADAPT lookback to timeframe.
4. **Momentum** (15%) - MACD + Stochastic. ADAPT periods to timeframe.
5. **Volume** (8%) - OBV divergence + volume spikes > 1.5x avg. Use as CONFIRMATION for other signals.
6. **ICT Smart Money** (18%) - FVG + BOS + liquidity sweeps + order blocks + displacement. Weight sweeps highest.
7. **GPU ML** (12%) - LSTM with RICH features (not just 3). Feed it RSI, MACD, ATR, volume ratio, etc.
8. **Economic Regime** (7%) - Adaptive regime weights. High VIX = shift to mean-reversion. Low VIX = trend.
9. **Cross-Year Pattern** (5%) - Normalize seasonal scores to [-1, 1] range. No raw scaling by 100.

## HIGH-IMPACT IMPROVEMENTS (in priority order):

### Priority 1: TIMEFRAME ADAPTATION (do this FIRST)
- Parse `context["timeframe"]` to get bar duration
- Scale ALL indicator periods proportionally
- This fixes signals on 5m, 15m, 1h, 4h data that are currently garbage

### Priority 2: MULTI-STRATEGY CONFIRMATION
- Don't just sum scores. Require 2+ strategies aligned for entry:
```python
# Count how many strategies agree on direction
agree_long = (trend_score > 0).astype(int) + (mom_score > 0).astype(int) + (ict_score > 0).astype(int) + ...
confirmation = (agree_long >= 3).astype(float)  # Need 3+ strategies agreeing
signal = combined_score * confirmation  # Only trade with confirmation
```
- This dramatically improves Sharpe, win rate, and profit factor

### Priority 3: RICH ML FEATURES
- Feed the LSTM 10+ features, not just 3:
```python
features = np.column_stack([
    returns, high_low_ratio, volume_change,
    rsi_values / 100, macd_hist, atr_pct,
    stoch_k / 100, adx_values / 100,
    bb_position,  # (close - lower) / (upper - lower)
    obv_slope,
])
```

### Priority 4: ADAPTIVE REGIME WEIGHTS
- When VIX > 30: Trend 8%, MR 18%, Volume 10%, reduce ICT
- When VIX < 15: Trend 20%, Mom 18%, Breakout 14%, reduce MR
- During Fed rate changes: flatten exposure 50%
- Before high-impact calendar events: reduce position by 40%

### Priority 5: GRADUATED POSITION SIZING
- Low conviction (score 0.5-1.0): 60% position
- Medium conviction (score 1.0-1.5): 100% position
- High conviction (score 1.5+): 130% position (leverage high-confidence)
- This maximizes return on best signals

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
    context["economic_calendar"]  # 54K+ news events with impact levels
    context["market"]        # "us30" or "btcusd"
    context["timeframe"]     # current timeframe string
```

## CRITICAL RULES:
1. Strategy MUST return `pd.Series(result.values.flatten(), index=df.index, dtype=float)` — ALWAYS use df.index
2. ALL boolean masks MUST use df.index. NEVER create Series with integer/range index.
3. Strategy MUST accept `(df, context=None)` signature
4. ALL 9 strategies MUST remain — tune weights but NEVER remove any
5. Use ONLY vectorized pandas/numpy — NO for-loops over rows
6. Handle edge cases: short DataFrames (<100 bars), missing context, NaN values
7. Make ONE focused change per iteration — don't rewrite everything
8. ADAPT indicator periods to timeframe — this is the #1 priority
9. Use `train_model()` + `predict_signals()` for GPU ML
10. Use economic context data (FRED + calendar)
11. Use ICT indicators (FVG + BOS + liquidity sweeps minimum)
12. Reindex economic data to df.index BEFORE using in boolean operations

## EVOLVE YOUR OWN STRATEGY:
The 9 base strategies are your building blocks. As you iterate:
- Track which strategies consistently produce positive signals across ALL timeframes
- COMBINE winning signals into hybrid rules (e.g., "ICT sweep + Volume spike + Trend alignment = 3x weight")
- Create NEW composite signals that blend the best parts of multiple strategies
- If Trend + ICT + Momentum all agree, that's a SUPER signal — weight it 2-3x normal
- Build your own "meta-strategy" on top of the 9 — the 9 are ingredients, the final recipe is YOURS
- The experiment history tells you what worked. Double down on winners, reduce losers.
- IMPORTANT: Keep all 9 base strategies generating scores, but CREATE new combination logic on top

## WHAT NOT TO DO:
- Don't use fixed thresholds that only work on daily data
- Don't scale seasonal scores by raw returns * 100 (causes noise)
- Don't treat all trades equally — use conviction-based sizing
- Don't ignore volume — it confirms everything
- Don't create Series with range(len(df)) index — always use df.index
