"""
Trading Strategy (MUTABLE - the agent modifies THIS file).
MEGA STRATEGY: Combines every known working strategy approach.
Uses GPU ML + price action + indicators + economic data.

The strategy function receives:
  df: DataFrame with columns open, high, low, close, volume
  context: dict with economic data, news calendar, market/timeframe info
Returns: pd.Series of per-bar returns
"""

import pandas as pd
import numpy as np
from strategy.indicators import sma, ema, rsi, atr, adx, macd, bollinger_bands, stochastic, donchian_channel, obv, vwap


def strategy(df: pd.DataFrame, context: dict = None) -> pd.Series:
    """
    Combined multi-strategy system:
    1. Trend Following (SMA/EMA crossover + ADX filter)
    2. Mean Reversion (Bollinger Bands + RSI extremes)
    3. Breakout (Donchian Channel breaks)
    4. Momentum (MACD + Stochastic)
    5. Volume Confirmation (OBV divergence)
    6. GPU ML Prediction (LSTM when available)
    7. Economic Regime Filter (Fed rate, VIX context)

    Each sub-strategy votes. Combined signal = weighted consensus.
    """
    n = len(df)
    if n < 60:
        return pd.Series(0.0, index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # === INDICATORS ===
    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    ema_9 = ema(close, 9)
    ema_21 = ema(close, 21)
    rsi_14 = rsi(close, 14)
    atr_14 = atr(df, 14)
    adx_14 = adx(df, 14)
    macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2.0)
    stoch_k, stoch_d = stochastic(df, 14, 3)
    dc_upper, dc_middle, dc_lower = donchian_channel(df, 20)
    obv_val = obv(df)

    # === STRATEGY 1: TREND FOLLOWING ===
    trend_score = pd.Series(0.0, index=df.index)
    trend_score[sma_20 > sma_50] += 1.0
    trend_score[sma_20 < sma_50] -= 1.0
    trend_score[ema_9 > ema_21] += 0.5
    trend_score[ema_9 < ema_21] -= 0.5
    # Only trust trend signals when ADX confirms trend
    trend_score = trend_score * (adx_14 > 20).astype(float)

    # === STRATEGY 2: MEAN REVERSION ===
    mr_score = pd.Series(0.0, index=df.index)
    mr_score[close < bb_lower] += 1.5  # Oversold at lower band
    mr_score[close > bb_upper] -= 1.5  # Overbought at upper band
    mr_score[rsi_14 < 30] += 1.0       # RSI oversold
    mr_score[rsi_14 > 70] -= 1.0       # RSI overbought
    # Only trust mean reversion in ranging markets
    mr_score = mr_score * (adx_14 <= 25).astype(float)

    # === STRATEGY 3: BREAKOUT (DONCHIAN) ===
    bo_score = pd.Series(0.0, index=df.index)
    bo_score[close > dc_upper.shift(1)] += 1.5  # Breakout above channel
    bo_score[close < dc_lower.shift(1)] -= 1.5  # Breakdown below channel

    # === STRATEGY 4: MOMENTUM (MACD + STOCHASTIC) ===
    mom_score = pd.Series(0.0, index=df.index)
    mom_score[macd_hist > 0] += 0.5
    mom_score[macd_hist < 0] -= 0.5
    mom_score[(macd_line > macd_signal) & (macd_hist > macd_hist.shift(1))] += 0.5
    mom_score[(macd_line < macd_signal) & (macd_hist < macd_hist.shift(1))] -= 0.5
    mom_score[(stoch_k < 20) & (stoch_k > stoch_d)] += 1.0  # Stoch oversold cross up
    mom_score[(stoch_k > 80) & (stoch_k < stoch_d)] -= 1.0  # Stoch overbought cross down

    # === STRATEGY 5: VOLUME CONFIRMATION ===
    vol_score = pd.Series(0.0, index=df.index)
    obv_sma = sma(obv_val, 20)
    vol_score[obv_val > obv_sma] += 0.3  # Volume supports uptrend
    vol_score[obv_val < obv_sma] -= 0.3  # Volume supports downtrend
    # Volume spike detection
    vol_avg = volume.rolling(20).mean()
    vol_spike = volume > (vol_avg * 1.5)
    vol_score[vol_spike & (close > close.shift(1))] += 0.5
    vol_score[vol_spike & (close < close.shift(1))] -= 0.5

    # === STRATEGY 6: GPU ML PREDICTION ===
    ml_score = pd.Series(0.0, index=df.index)
    try:
        from strategy.ml_model import train_model, predict_signals
        model = train_model(df, lookback=60, epochs=10)
        if model is not None:
            ml_signals = predict_signals(model, df, lookback=60)
            ml_score = ml_signals.astype(float) * 2.0  # Strong weight for ML
    except Exception:
        pass  # ML not available, skip

    # === STRATEGY 7: ECONOMIC REGIME FILTER ===
    regime_multiplier = pd.Series(1.0, index=df.index)
    if context is not None:
        try:
            # Use VIX for volatility regime
            if "fred_vixcls" in context:
                vix = context["fred_vixcls"]
                vix = vix.reindex(df.index, method="ffill")
                if len(vix.columns) > 0:
                    vix_val = vix.iloc[:, 0]
                    # High VIX = reduce position, low VIX = normal
                    regime_multiplier[vix_val > 30] = 0.5
                    regime_multiplier[vix_val > 40] = 0.25

            # Use Fed Funds Rate for macro trend
            if "fred_dff" in context:
                fed = context["fred_dff"]
                fed = fed.reindex(df.index, method="ffill")
                if len(fed.columns) > 0:
                    fed_val = fed.iloc[:, 0]
                    fed_change = fed_val.diff(20)  # 20-bar rate change
                    # Rising rates = bearish bias
                    regime_multiplier[fed_change > 0.5] *= 0.8
                    # Falling rates = bullish bias
                    regime_multiplier[fed_change < -0.5] *= 1.2
        except Exception:
            pass

    # === COMBINE ALL STRATEGIES ===
    # Weighted voting system
    combined = (
        trend_score * 0.25 +    # 25% trend following
        mr_score * 0.15 +       # 15% mean reversion
        bo_score * 0.15 +       # 15% breakout
        mom_score * 0.20 +      # 20% momentum
        vol_score * 0.10 +      # 10% volume
        ml_score * 0.15         # 15% ML prediction
    )

    # Apply economic regime filter
    combined = combined * regime_multiplier

    # Convert to signal: threshold-based
    signal = pd.Series(0, index=df.index)
    signal[combined > 0.5] = 1     # Long
    signal[combined < -0.5] = -1   # Short

    # ATR-based position sizing (smaller in high volatility)
    atr_pct = atr_14 / close
    position_size = (1.0 / (1.0 + 5 * atr_pct)).clip(0.2, 1.0).fillna(0.5)

    # Final sized signal
    sized_signal = signal * position_size

    # Calculate returns
    price_returns = close.pct_change()
    strategy_returns = sized_signal.shift(1) * price_returns

    # Ensure output is a 1D Series (fix the ndarray shape error)
    result = strategy_returns.fillna(0)
    if hasattr(result, 'squeeze'):
        result = result.squeeze()
    return pd.Series(result.values, index=df.index, dtype=float)
