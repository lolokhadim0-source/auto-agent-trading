"""
Trading Strategy (MUTABLE - the agent modifies THIS file).
Like AutoResearch's train.py, this is the only file the inner loop agent changes.

The strategy function receives a DataFrame with columns: open, high, low, close, volume
and must return a pd.Series of per-bar returns (positive = profit, negative = loss).

Current baseline: Simple Moving Average Crossover
"""

import pandas as pd
import numpy as np
from strategy.indicators import sma, ema, rsi, atr, adx, macd, bollinger_bands


def strategy(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum strategy with Bollinger Bands mean reversion overlay.
    
    Primary strategy: Trend following with SMA crossover
    Overlay: Bollinger Band mean reversion in ranging markets
    Risk management: ATR-based stops and position sizing
    
    Returns:
        pd.Series of per-bar returns
    """
    # Primary trend following parameters
    fast_period = 20
    slow_period = 50
    
    # Calculate indicators
    fast_ma = sma(df["close"], fast_period)
    slow_ma = sma(df["close"], slow_period)
    rsi_val = rsi(df["close"], 14)
    atr_val = atr(df, 14)
    adx_val = adx(df, 14)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_middle, bb_lower = bollinger_bands(df["close"], 20, 2.0)
    bb_position = (df["close"] - bb_middle) / (bb_upper - bb_middle)
    
    # MACD for momentum confirmation
    macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
    
    # Multi-regime detection
    trending_market = adx_val > 25
    ranging_market = adx_val <= 25
    
    # Trend following signals (strong trending markets)
    trend_long = (fast_ma > slow_ma) & (rsi_val < 70) & trending_market & (macd_line > macd_signal)
    trend_short = (fast_ma < slow_ma) & (rsi_val > 30) & trending_market & (macd_line < macd_signal)
    
    # Mean reversion signals (ranging markets)
    mean_rev_long = ranging_market & (df["close"] < bb_lower) & (rsi_val < 35)
    mean_rev_short = ranging_market & (df["close"] > bb_upper) & (rsi_val > 65)
    
    # Combined signal logic
    signal = pd.Series(0, index=df.index)
    signal[trend_long | mean_rev_long] = 1
    signal[trend_short | mean_rev_short] = -1
    
    # Enhanced position sizing based on volatility and signal strength
    base_position = 1.0 / (1.0 + 2 * (atr_val / df["close"]))
    
    # Signal strength modifier
    signal_strength = pd.Series(1.0, index=df.index)
    
    # Stronger position for trend following in strong trends
    signal_strength[trend_long | trend_short] = 1.0 + (adx_val - 25) / 100
    
    # Smaller position for mean reversion
    signal_strength[mean_rev_long | mean_rev_short] = 0.6
    
    # Apply position sizing
    position_size = (base_position * signal_strength).clip(0.1, 1.0).fillna(0.5)
    sized_signal = signal * position_size
    
    # Dynamic stop loss based on ATR and strategy type
    entry_price = pd.Series(np.nan, index=df.index)
    current_position = 0
    stop_loss_exits = pd.Series(False, index=df.index)
    strategy_type = pd.Series("", index=df.index)
    
    for i in range(1, len(df)):
        if current_position == 0 and sized_signal.iloc[i] != 0:
            # New position entry
            current_position = sized_signal.iloc[i]
            entry_price.iloc[i] = df["close"].iloc[i]
            
            # Determine strategy type for this trade
            if trending_market.iloc[i]:
                strategy_type.iloc[i] = "trend"
            else:
                strategy_type.iloc[i] = "mean_rev"
                
        elif current_position != 0:
            # Dynamic stop loss based on strategy type
            if strategy_type.iloc[i-1] == "trend":
                stop_pct = 0.04  # 4% stop for trend following
            else:
                stop_pct = 0.02  # 2% stop for mean reversion
            
            # Check stop loss condition
            if current_position > 0:  # Long position
                if df["close"].iloc[i] <= entry_price.iloc[i-1] * (1 - stop_pct):
                    stop_loss_exits.iloc[i] = True
                    current_position = 0
            else:  # Short position  
                if df["close"].iloc[i] >= entry_price.iloc[i-1] * (1 + stop_pct):
                    stop_loss_exits.iloc[i] = True
                    current_position = 0
                    
            # Update tracking variables for existing position
            if current_position != 0:
                entry_price.iloc[i] = entry_price.iloc[i-1]
                strategy_type.iloc[i] = strategy_type.iloc[i-1]
    
    # Apply stop loss exits
    final_signal = sized_signal.copy()
    final_signal[stop_loss_exits] = 0
    
    # Calculate returns
    price_returns = df["close"].pct_change()
    strategy_returns = final_signal.shift(1) * price_returns
    
    return strategy_returns.fillna(0)