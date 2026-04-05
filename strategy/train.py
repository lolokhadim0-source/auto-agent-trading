"""
Trading Strategy (MUTABLE - the agent modifies THIS file).
Like AutoResearch's train.py, this is the only file the inner loop agent changes.

The strategy function receives a DataFrame with columns: open, high, low, close, volume
and must return a pd.Series of per-bar returns (positive = profit, negative = loss).

Current baseline: Simple Moving Average Crossover
"""

import pandas as pd
import numpy as np
from strategy.indicators import sma, ema, rsi, atr


def strategy(df: pd.DataFrame) -> pd.Series:
    """
    Baseline strategy: SMA crossover with RSI filter.

    Rules:
    - Go long when fast SMA crosses above slow SMA and RSI < 70
    - Go short when fast SMA crosses below slow SMA and RSI > 30
    - Position size: 1 unit (fully invested)
    - No stop loss or take profit (agent should add these)

    Returns:
        pd.Series of per-bar returns
    """
    fast_period = 20
    slow_period = 50

    fast_ma = sma(df["close"], fast_period)
    slow_ma = sma(df["close"], slow_period)
    rsi_val = rsi(df["close"], 14)

    # Signals: 1 = long, -1 = short, 0 = flat
    signal = pd.Series(0, index=df.index)

    # Long when fast > slow and RSI not overbought
    long_condition = (fast_ma > slow_ma) & (rsi_val < 70)
    # Short when fast < slow and RSI not oversold
    short_condition = (fast_ma < slow_ma) & (rsi_val > 30)

    signal[long_condition] = 1
    signal[short_condition] = -1

    # Calculate returns: signal shifted by 1 (trade on next bar) * price change
    price_returns = df["close"].pct_change()
    strategy_returns = signal.shift(1) * price_returns

    # Fill NaN with 0
    strategy_returns = strategy_returns.fillna(0)

    return strategy_returns
