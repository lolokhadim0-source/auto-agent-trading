"""
Technical indicators library. Available for use in strategy/train.py.
Uses the `ta` library + custom implementations.
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_vals = atr(df, period)
    plus_di = 100 * ema(plus_dm, period) / atr_vals
    minus_di = 100 * ema(minus_dm, period) / atr_vals
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return ema(dx, period)


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).fillna(0).cumsum()


def donchian_channel(df: pd.DataFrame, period: int = 20):
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend


# === ICT CONCEPTS ===

def fair_value_gap(df: pd.DataFrame) -> pd.Series:
    """Detect Fair Value Gaps (FVG) - ICT concept.
    Bullish FVG: current low > 2-bars-ago high (gap up)
    Bearish FVG: current high < 2-bars-ago low (gap down)
    Returns: 1=bullish FVG, -1=bearish FVG, 0=none
    """
    fvg = pd.Series(0, index=df.index)
    fvg[df["low"] > df["high"].shift(2)] = 1   # Bullish FVG
    fvg[df["high"] < df["low"].shift(2)] = -1  # Bearish FVG
    return fvg


def order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Detect Order Blocks - ICT concept.
    Bullish OB: last bearish candle before a strong bullish move
    Bearish OB: last bullish candle before a strong bearish move
    Returns: 1=at bullish OB, -1=at bearish OB, 0=none
    """
    close = df["close"]
    op = df["open"]
    high = df["high"]
    low = df["low"]

    ob = pd.Series(0, index=df.index)
    returns_fwd = close.pct_change(3).shift(-3)  # 3-bar forward return

    for i in range(lookback, len(df) - 3):
        # Bearish candle followed by strong bullish move = bullish OB
        if close.iloc[i] < op.iloc[i] and returns_fwd.iloc[i] > 0.02:
            # Check if price revisits this level
            ob_level = low.iloc[i]
            for j in range(i + 3, min(i + lookback, len(df))):
                if low.iloc[j] <= ob_level * 1.005 and low.iloc[j] >= ob_level * 0.995:
                    ob.iloc[j] = 1
                    break
        # Bullish candle followed by strong bearish move = bearish OB
        if close.iloc[i] > op.iloc[i] and returns_fwd.iloc[i] < -0.02:
            ob_level = high.iloc[i]
            for j in range(i + 3, min(i + lookback, len(df))):
                if high.iloc[j] >= ob_level * 0.995 and high.iloc[j] <= ob_level * 1.005:
                    ob.iloc[j] = -1
                    break

    return ob


def break_of_structure(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Detect Break of Structure (BOS) - ICT concept.
    Bullish BOS: price breaks above recent swing high
    Bearish BOS: price breaks below recent swing low
    Returns: 1=bullish BOS, -1=bearish BOS, 0=none
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    swing_high = high.rolling(period).max()
    swing_low = low.rolling(period).min()

    bos = pd.Series(0, index=df.index)
    bos[(close > swing_high.shift(1)) & (close.shift(1) <= swing_high.shift(2))] = 1   # Bullish BOS
    bos[(close < swing_low.shift(1)) & (close.shift(1) >= swing_low.shift(2))] = -1    # Bearish BOS
    return bos


def liquidity_sweep(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Detect Liquidity Sweeps - ICT concept.
    Price briefly breaks a key level then reverses (stop hunt).
    Bullish sweep: breaks below swing low then closes above it
    Bearish sweep: breaks above swing high then closes below it
    Returns: 1=bullish sweep (buy signal), -1=bearish sweep (sell signal)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    swing_high = high.rolling(period).max().shift(1)
    swing_low = low.rolling(period).min().shift(1)

    sweep = pd.Series(0, index=df.index)
    # Bullish: wick below swing low but close above = stop hunt complete, go long
    sweep[(low < swing_low) & (close > swing_low)] = 1
    # Bearish: wick above swing high but close below = stop hunt complete, go short
    sweep[(high > swing_high) & (close < swing_high)] = -1
    return sweep


def displacement(df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Detect Displacement candles - ICT concept.
    Large body candle with strong momentum (body > threshold * ATR).
    Returns: 1=bullish displacement, -1=bearish displacement, 0=none
    """
    body = (df["close"] - df["open"]).abs()
    atr_val = atr(df, 14)
    disp = pd.Series(0, index=df.index)
    disp[(df["close"] > df["open"]) & (body > threshold * atr_val)] = 1   # Bullish
    disp[(df["close"] < df["open"]) & (body > threshold * atr_val)] = -1  # Bearish
    return disp
