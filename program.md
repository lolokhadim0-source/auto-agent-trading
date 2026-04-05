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

## Phase 1: Foundation (iterations 1-20)
- Start with simple moving average strategies, vary periods
- Test different indicator combinations (RSI, MACD, Bollinger Bands)
- Establish a solid baseline that works on both markets
- Focus on avoiding large drawdowns

## Phase 2: Refinement (iterations 21-50)
- Add risk management: stop losses, take profits, trailing stops
- Experiment with position sizing (ATR-based, volatility-adjusted)
- Try trend-following vs mean-reversion approaches
- Consider multi-indicator confirmation signals

## Phase 3: Advanced (iterations 51+)
- Multi-timeframe analysis if applicable
- Regime detection (trending vs ranging markets)
- Adaptive parameters that adjust to market conditions
- Volume-based confirmations (OBV, VWAP)
- Explore Donchian channels, Supertrend, ADX filtering

## Constraints
- Strategy must work on BOTH US30 and BTCUSD (not overfit to one)
- Must maintain the function signature: `strategy(df) -> pd.Series`
- Make ONE focused change per iteration
- Keep code readable and maintainable
