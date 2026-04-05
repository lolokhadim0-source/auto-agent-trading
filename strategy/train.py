"""
Trading Strategy - EVERYTHING CONNECTED.
Combines all known strategies + ICT + ML + news + cross-timeframe + cross-year patterns.
"""

import pandas as pd
import numpy as np
from strategy.indicators import (
    sma, ema, rsi, atr, adx, macd, bollinger_bands, stochastic,
    donchian_channel, obv, vwap, supertrend,
    fair_value_gap, break_of_structure, liquidity_sweep, displacement,
)


def strategy(df: pd.DataFrame, context: dict = None) -> pd.Series:
    n = len(df)
    if n < 60:
        return pd.Series(0.0, index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    op = df["open"]

    # === CORE INDICATORS ===
    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    sma_200 = sma(close, min(200, n // 3)) if n > 200 else sma(close, 50)
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

    # === ICT INDICATORS ===
    fvg = fair_value_gap(df)
    bos = break_of_structure(df, 20)
    sweep = liquidity_sweep(df, 20)
    disp = displacement(df, 2.0)

    # === 1. TREND FOLLOWING ===
    trend_score = pd.Series(0.0, index=df.index)
    trend_score[sma_20 > sma_50] += 1.0
    trend_score[sma_20 < sma_50] -= 1.0
    trend_score[ema_9 > ema_21] += 0.5
    trend_score[ema_9 < ema_21] -= 0.5
    trend_score[close > sma_200] += 0.5  # Long-term trend bias
    trend_score[close < sma_200] -= 0.5
    trend_score = trend_score * (adx_14 > 20).astype(float)

    # === 2. MEAN REVERSION ===
    mr_score = pd.Series(0.0, index=df.index)
    mr_score[close < bb_lower] += 1.5
    mr_score[close > bb_upper] -= 1.5
    mr_score[rsi_14 < 30] += 1.0
    mr_score[rsi_14 > 70] -= 1.0
    mr_score = mr_score * (adx_14 <= 25).astype(float)

    # === 3. BREAKOUT ===
    bo_score = pd.Series(0.0, index=df.index)
    bo_score[close > dc_upper.shift(1)] += 1.5
    bo_score[close < dc_lower.shift(1)] -= 1.5

    # === 4. MOMENTUM ===
    mom_score = pd.Series(0.0, index=df.index)
    mom_score[macd_hist > 0] += 0.5
    mom_score[macd_hist < 0] -= 0.5
    mom_score[(macd_line > macd_signal) & (macd_hist > macd_hist.shift(1))] += 0.5
    mom_score[(macd_line < macd_signal) & (macd_hist < macd_hist.shift(1))] -= 0.5
    mom_score[(stoch_k < 20) & (stoch_k > stoch_d)] += 1.0
    mom_score[(stoch_k > 80) & (stoch_k < stoch_d)] -= 1.0

    # === 5. VOLUME ===
    vol_score = pd.Series(0.0, index=df.index)
    obv_sma = sma(obv_val, 20)
    vol_score[obv_val > obv_sma] += 0.3
    vol_score[obv_val < obv_sma] -= 0.3
    vol_avg = volume.rolling(20).mean()
    vol_spike = volume > (vol_avg * 1.5)
    vol_score[vol_spike & (close > close.shift(1))] += 0.5
    vol_score[vol_spike & (close < close.shift(1))] -= 0.5

    # === 6. ICT SMART MONEY ===
    ict_score = pd.Series(0.0, index=df.index)
    ict_score += fvg * 1.0
    ict_score += bos * 1.5
    ict_score += sweep * 2.0  # Liquidity sweeps = highest conviction
    ict_score += disp * 0.5

    # === 7. GPU ML PREDICTION ===
    ml_score = pd.Series(0.0, index=df.index)
    try:
        from strategy.ml_model import train_model, predict_signals
        model = train_model(df, lookback=60, epochs=10)
        if model is not None:
            ml_signals = predict_signals(model, df, lookback=60)
            ml_score = ml_signals.astype(float) * 2.0
    except Exception:
        pass

    # === 8. CROSS-YEAR PATTERN DETECTION ===
    cross_year = pd.Series(0.0, index=df.index)
    if n > 252:  # Need at least 1 year of data
        # Monthly seasonality: detect recurring patterns by month
        if hasattr(df.index, 'month'):
            monthly_returns = close.pct_change()
            for month in range(1, 13):
                mask = df.index.month == month
                month_avg = monthly_returns[mask].mean()
                if not np.isnan(month_avg):
                    cross_year[mask] += month_avg * 100  # Scale up

        # Year-over-year momentum: compare to same period last year
        yearly_lookback = min(252, n - 1)
        yearly_return = close.pct_change(yearly_lookback)
        cross_year[yearly_return > 0.1] += 0.5   # Up >10% vs last year = bullish
        cross_year[yearly_return < -0.1] -= 0.5   # Down >10% = bearish

        # Detect recurring support/resistance levels across years
        price_levels = close.rolling(252).quantile(0.05)  # 5th percentile = support
        price_ceiling = close.rolling(252).quantile(0.95)  # 95th percentile = resistance
        near_support = (close - price_levels).abs() / close < 0.02
        near_resistance = (close - price_ceiling).abs() / close < 0.02
        cross_year[near_support] += 1.0   # Near yearly support = buy
        cross_year[near_resistance] -= 1.0  # Near yearly resistance = sell

    # === 9. NEWS & ECONOMIC CORRELATION ===
    news_score = pd.Series(0.0, index=df.index)
    regime_multiplier = pd.Series(1.0, index=df.index)

    if context is not None:
        market = context.get("market", "")
        timeframe = context.get("timeframe", "")

        # --- VIX regime filter ---
        try:
            if "fred_vixcls" in context:
                vix = context["fred_vixcls"].reindex(df.index, method="ffill")
                if hasattr(vix, 'iloc') and len(vix) > 0:
                    vix_val = vix.iloc[:, 0] if vix.ndim > 1 else vix
                    regime_multiplier[vix_val > 30] = 0.5
                    regime_multiplier[vix_val > 40] = 0.25
                    regime_multiplier[vix_val < 15] = 1.2  # Low VIX = risk on
        except Exception:
            pass

        # --- Fed Funds Rate trend ---
        try:
            if "fred_dff" in context:
                fed = context["fred_dff"].reindex(df.index, method="ffill")
                if hasattr(fed, 'iloc') and len(fed) > 0:
                    fed_val = fed.iloc[:, 0] if fed.ndim > 1 else fed
                    fed_change = fed_val.diff(20)
                    regime_multiplier[fed_change > 0.5] *= 0.8  # Rising rates = careful
                    regime_multiplier[fed_change < -0.5] *= 1.2  # Falling rates = bullish
        except Exception:
            pass

        # --- Treasury Spread (recession indicator) ---
        try:
            if "fred_t10y2y" in context:
                spread = context["fred_t10y2y"].reindex(df.index, method="ffill")
                if hasattr(spread, 'iloc') and len(spread) > 0:
                    spread_val = spread.iloc[:, 0] if spread.ndim > 1 else spread
                    regime_multiplier[spread_val < 0] *= 0.7  # Inverted yield curve = danger
                    regime_multiplier[spread_val > 1] *= 1.1  # Healthy spread = risk on
        except Exception:
            pass

        # --- CPI / Inflation trend ---
        try:
            if "fred_cpiaucsl" in context:
                cpi = context["fred_cpiaucsl"].reindex(df.index, method="ffill")
                if hasattr(cpi, 'iloc') and len(cpi) > 0:
                    cpi_val = cpi.iloc[:, 0] if cpi.ndim > 1 else cpi
                    cpi_yoy = cpi_val.pct_change(12) * 100  # Year-over-year %
                    regime_multiplier[cpi_yoy > 5] *= 0.8  # High inflation = bearish
                    regime_multiplier[cpi_yoy < 2] *= 1.1  # Low inflation = bullish
        except Exception:
            pass

        # --- Economic Calendar Events ---
        try:
            if "economic_calendar" in context:
                cal = context["economic_calendar"]
                # Count high-impact events near each bar
                for idx in df.index:
                    try:
                        window_start = idx - pd.Timedelta(days=1)
                        window_end = idx + pd.Timedelta(days=1)
                        nearby = cal[(cal.index >= window_start) & (cal.index <= window_end)]
                        if len(nearby) > 0:
                            high_impact = nearby[nearby["impact"] == "high"] if "impact" in nearby.columns else nearby
                            if len(high_impact) > 0:
                                # High impact event nearby — reduce position (uncertainty)
                                regime_multiplier.loc[idx] *= 0.6
                                # Check if actual > previous (positive surprise)
                                if "change" in high_impact.columns:
                                    avg_change = high_impact["change"].mean()
                                    if avg_change > 0:
                                        news_score.loc[idx] += 0.5
                                    else:
                                        news_score.loc[idx] -= 0.5
                    except Exception:
                        continue
        except Exception:
            pass

        # --- Market-specific adjustments ---
        if market == "btcusd":
            # BTC trades 24/7, more volatile — reduce position
            regime_multiplier *= 0.8
        elif market == "us30":
            # US30 more stable during market hours
            regime_multiplier *= 1.0

    # === COMBINE ALL 9 STRATEGIES ===
    combined = (
        trend_score * 0.12 +
        mr_score * 0.08 +
        bo_score * 0.08 +
        mom_score * 0.12 +
        vol_score * 0.05 +
        ict_score * 0.20 +
        ml_score * 0.15 +
        cross_year * 0.10 +
        news_score * 0.10
    )

    # Apply regime filter
    combined = combined * regime_multiplier

    # Signal threshold
    signal = pd.Series(0, index=df.index)
    signal[combined > 0.5] = 1
    signal[combined < -0.5] = -1

    # ATR-based position sizing
    atr_pct = atr_14 / close
    position_size = (1.0 / (1.0 + 5 * atr_pct)).clip(0.2, 1.0).fillna(0.5)
    sized_signal = signal * position_size

    # Calculate returns
    price_returns = close.pct_change()
    strategy_returns = sized_signal.shift(1) * price_returns

    # Ensure 1D output
    result = strategy_returns.fillna(0)
    if hasattr(result, 'squeeze'):
        result = result.squeeze()
    return pd.Series(result.values.flatten(), index=df.index, dtype=float)
