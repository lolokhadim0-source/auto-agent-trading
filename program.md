# Research Directives

## Objective
Optimize the trading strategy in `strategy/train.py` to maximize the composite score across both US30 (Dow Jones) and BTCUSD markets, across all available timeframes.

## Scoring
The composite score is a weighted combination:
- Sharpe Ratio: 30% weight (risk-adjusted returns)
- Total Return: 25% weight (absolute profitability)
- Max Drawdown: 20% weight (capital preservation)
- Profit Factor: 15% weight (gross profit / gross loss)
- Win Rate: 10% weight (percentage of winning trades)

## Current Status Analysis
After 10 experiments with only 1 improvement (10% success rate), the system has struggled to improve beyond the initial baseline score of 3.0991. All subsequent modifications have decreased performance, suggesting the need for a more systematic approach.

## Phase 1: Consolidation and Risk Management (iterations 11-25)
**Priority**: Build upon the working baseline with conservative improvements
- Add basic risk management: fixed stop losses (2-5% of position)
- Implement position sizing limits (max 10% of capital per trade)
- Test simple take profit levels (1.5:1, 2:1, 3:1 risk-reward ratios)
- Add trade frequency controls to avoid overtrading
- Focus on reducing drawdown rather than increasing returns

## Phase 2: Signal Quality Enhancement (iterations 26-40)
**Priority**: Improve entry/exit timing without changing core logic
- Add RSI overbought/oversold filters (RSI > 70 for sells, RSI < 30 for buys)
- Implement volume confirmation (require above-average volume for signals)
- Test ADX trend strength filter (ADX > 25 for trend following)
- Add simple momentum confirmation (price above/below 5-day average)
- Use MACD histogram for timing refinement

## Phase 3: Multi-Asset Optimization (iterations 41-55)
**Priority**: Address the dual-market requirement more systematically
- Create separate parameter sets for US30 vs BTCUSD if needed
- Test volatility-adjusted position sizing using ATR
- Implement market regime detection (trending vs ranging)
- Add correlation-based filters between the two assets
- Consider time-of-day filters for each market's active hours

## Phase 4: Cross-Year Pattern Recognition & Cycle Detection (iterations 56-75)
**Priority**: The strategy MUST connect patterns across multiple years of data
- Detect seasonal/cyclical patterns (monthly, quarterly, yearly repetitions)
- Identify recurring manipulation patterns (stop hunts, liquidity grabs, fakeouts)
- Analyze historical support/resistance levels that repeat across years
- Detect accumulation/distribution phases using volume and price action
- Identify "smart money" patterns: Wyckoff accumulation/distribution, order blocks
- Look for time-of-day and day-of-week recurring patterns
- Detect range-bound manipulation (repeated false breakouts at key levels)
- Use long lookback periods (200+ bars) to capture macro cycles
- Cross-reference US30 and BTCUSD for correlated manipulation events

## Phase 5: News & Events Correlation (iterations 76-90)
**Priority**: Connect price action to macro news events across years
**Available data**: `data/news/economic_calendar.parquet` contains Forex Factory style events (2015-2026) with impact levels (high/medium/low), actual vs forecast vs previous values, country, and event name. Load with `pd.read_parquet()`.
**Also available**: `data/economic/` contains FRED series (Fed Funds Rate, Treasury Spread, CPI, Unemployment, USD Index) going back decades.
- Correlate FRED economic data (Fed rate decisions, CPI releases, unemployment) with price reactions
- Detect recurring patterns around scheduled economic events (FOMC, NFP, CPI release dates)
- Identify how US30 and BTCUSD react differently to the same macro events
- Look for post-news momentum vs mean-reversion patterns
- Detect "buy the rumor, sell the news" patterns around major events
- Use economic regime indicators (rate hiking vs cutting, expansion vs recession) to adjust strategy
- Cross-reference VIX-like volatility spikes with entry/exit timing
- Identify multi-year narratives (QE, tightening cycles) and how they affect both markets

## Phase 6: GPU Machine Learning (iterations 91-110)
**Priority**: USE THE GPU for ML-based price prediction
**Available**: `from strategy.ml_model import train_model, predict_signals`
- Train LSTM neural network on GPU using `train_model(df, lookback=60, epochs=20)`
- Get ML signals using `predict_signals(model, df, lookback=60)` -> returns 1=long, -1=short, 0=flat
- Combine ML predictions with price action for higher conviction entries
- Use ML to detect patterns humans can't see in the data
- Train separate models for different market regimes
- Use ML confidence scores as position sizing input
- The GPU (T4) accelerates training 10-50x vs CPU

## Phase 7: Advanced Techniques (iterations 111+)
**Priority**: Explore fundamentally different approaches if needed
- Test mean reversion strategies (Bollinger Band reversals)
- Implement breakout strategies (Donchian channels)
- Add multi-timeframe confirmation (higher timeframe trend)
- Experiment with ensemble approaches (combine multiple signals)
- Test adaptive parameters that adjust based on recent performance

## Key Constraints and Guidelines
- **Conservative approach**: Make smaller, incremental changes rather than major overhauls
- **Preserve what works**: Always maintain the core elements of the successful baseline
- **Risk-first mentality**: Prioritize drawdown reduction over return maximization
- **Dual-market focus**: Every change must be tested on both US30 and BTCUSD
- **One change per iteration**: Maintain systematic testing approach
- **Rollback readiness**: If a change doesn't improve within 2-3 iterations, try a different direction

## Success Metrics
- Target: Achieve score > 4.0 within next 15 iterations
- Minimum acceptable: Any improvement above 3.0991
- Risk tolerance: Max drawdown should not exceed current baseline levels
- Consistency: Strategies should perform reasonably on both markets, not just one