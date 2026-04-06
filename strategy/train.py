"""
Trading Strategy - IMPOSSIBLE MODE.
Timeframe-adaptive, multi-strategy confirmation, conviction-based sizing.
Combines all 9 strategies + ICT + ML + news + cross-year with EVOLVED combination logic.
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

    # === TIMEFRAME ADAPTATION ===
    # Scale all indicator periods based on bar duration
    tf = context.get("timeframe", "1d") if context else "1d"
    tf_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
              "1d": 1440, "1Day": 1440, "1wk": 10080, "1mo": 43200,
              "1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}
    tf_minutes = tf_map.get(tf, 1440)
    # Scale factor: how many bars make ~1 day
    scale = max(1, 1440 // tf_minutes)

    # Adaptive periods - scale to timeframe
    fast_p = max(5, min(scale * 5, n // 4))
    med_p = max(10, min(scale * 13, n // 3))
    slow_p = max(20, min(scale * 50, n // 3))
    trend_p = max(50, min(scale * 100, n // 2))
    rsi_p = max(7, min(scale * 3, n // 5))
    atr_p = max(7, min(scale * 3, n // 5))
    adx_p = max(7, min(scale * 3, n // 5))
    dc_p = max(10, min(scale * 5, n // 4))
    bb_p = max(10, min(scale * 5, n // 4))
    stoch_p = max(7, min(scale * 3, n // 5))

    # === CORE INDICATORS (timeframe-adapted) ===
    sma_fast = sma(close, fast_p)
    sma_med = sma(close, med_p)
    sma_slow = sma(close, slow_p)
    sma_trend = sma(close, trend_p)
    ema_fast = ema(close, max(5, fast_p // 2))
    ema_med = ema(close, fast_p)
    rsi_val = rsi(close, rsi_p)
    atr_val = atr(df, atr_p)
    adx_val = adx(df, adx_p)
    macd_f = max(6, fast_p // 2)
    macd_s = max(13, med_p)
    macd_sig = max(5, fast_p // 3)
    macd_line, macd_signal, macd_hist = macd(close, macd_f, macd_s, macd_sig)
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, bb_p, 2.0)
    stoch_k, stoch_d = stochastic(df, stoch_p, 3)
    dc_upper, dc_middle, dc_lower = donchian_channel(df, dc_p)
    obv_val = obv(df)

    # === ICT INDICATORS (timeframe-adapted) ===
    ict_period = max(10, min(scale * 5, n // 4))
    fvg = fair_value_gap(df)
    bos = break_of_structure(df, ict_period)
    sweep = liquidity_sweep(df, ict_period)
    disp = displacement(df, 2.0)

    # === 1. TREND FOLLOWING (15%) ===
    trend_score = pd.Series(0.0, index=df.index)
    trend_score = trend_score + np.where(sma_fast > sma_med, 1.0, np.where(sma_fast < sma_med, -1.0, 0.0))
    trend_score = trend_score + np.where(ema_fast > ema_med, 0.5, np.where(ema_fast < ema_med, -0.5, 0.0))
    trend_score = trend_score + np.where(close > sma_trend, 0.5, np.where(close < sma_trend, -0.5, 0.0))
    # Only trade trends when ADX confirms
    trend_score = trend_score * (adx_val > 20).astype(float)

    # === 2. MEAN REVERSION (10%) ===
    mr_score = pd.Series(0.0, index=df.index)
    mr_score = mr_score + np.where(close < bb_lower, 1.5, np.where(close > bb_upper, -1.5, 0.0))
    mr_score = mr_score + np.where(rsi_val < 30, 1.0, np.where(rsi_val > 70, -1.0, 0.0))
    # Only mean-revert in ranging markets
    mr_score = mr_score * (adx_val <= 25).astype(float)

    # === 3. BREAKOUT (10%) ===
    bo_score = pd.Series(0.0, index=df.index)
    bo_score = bo_score + np.where(close > dc_upper.shift(1).values, 1.5, np.where(close < dc_lower.shift(1).values, -1.5, 0.0))
    # Volume confirmation for breakouts
    vol_avg = volume.rolling(max(10, fast_p)).mean()
    vol_spike = (volume > vol_avg * 1.3).astype(float)
    bo_score = bo_score * (0.5 + 0.5 * vol_spike)  # Boost breakouts with volume

    # === 4. MOMENTUM (15%) ===
    mom_score = pd.Series(0.0, index=df.index)
    mom_score = mom_score + np.where(macd_hist > 0, 0.5, np.where(macd_hist < 0, -0.5, 0.0))
    mom_accel = macd_hist > macd_hist.shift(1)
    mom_decel = macd_hist < macd_hist.shift(1)
    mom_score = mom_score + np.where((macd_line > macd_signal) & mom_accel, 0.5, 0.0)
    mom_score = mom_score + np.where((macd_line < macd_signal) & mom_decel, -0.5, 0.0)
    mom_score = mom_score + np.where((stoch_k < 20) & (stoch_k > stoch_d), 1.0, 0.0)
    mom_score = mom_score + np.where((stoch_k > 80) & (stoch_k < stoch_d), -1.0, 0.0)

    # === 5. VOLUME (8%) ===
    vol_score = pd.Series(0.0, index=df.index)
    obv_sma_val = sma(obv_val, max(10, fast_p))
    vol_score = vol_score + np.where(obv_val > obv_sma_val, 0.3, np.where(obv_val < obv_sma_val, -0.3, 0.0))
    vol_score = vol_score + np.where(vol_spike.astype(bool) & (close > close.shift(1)), 0.5, 0.0)
    vol_score = vol_score + np.where(vol_spike.astype(bool) & (close < close.shift(1)), -0.5, 0.0)

    # === 6. ICT SMART MONEY (18%) ===
    ict_score = pd.Series(0.0, index=df.index)
    ict_score = ict_score + fvg * 1.0
    ict_score = ict_score + bos * 1.5
    ict_score = ict_score + sweep * 2.0  # Liquidity sweeps = highest conviction
    ict_score = ict_score + disp * 0.5

    # === 7. GPU ML PREDICTION (12%) ===
    ml_score = pd.Series(0.0, index=df.index)
    try:
        from strategy.ml_model import train_model, predict_signals
        model = train_model(df, lookback=60, epochs=10)
        if model is not None:
            ml_signals = predict_signals(model, df, lookback=60)
            ml_score = ml_signals.astype(float) * 2.0
    except Exception:
        pass

    # === 8. CROSS-YEAR PATTERN DETECTION (5%) ===
    cross_year = pd.Series(0.0, index=df.index)
    yearly_bars = max(scale * 252, 252)  # Adapt to timeframe
    if n > yearly_bars:
        if hasattr(df.index, 'month'):
            monthly_returns = close.pct_change()
            for month in range(1, 13):
                mask = df.index.month == month
                month_avg = monthly_returns[mask].mean()
                month_std = monthly_returns[mask].std()
                if not np.isnan(month_avg) and month_std > 0:
                    # Normalize to [-1, 1] range instead of raw scaling
                    cross_year.loc[mask] += np.clip(month_avg / month_std, -1, 1)

        # Year-over-year momentum
        yoy_lookback = min(yearly_bars, n - 1)
        yearly_return = close.pct_change(yoy_lookback)
        cross_year = cross_year + np.where(yearly_return > 0.1, 0.5, np.where(yearly_return < -0.1, -0.5, 0.0))

        # Recurring support/resistance
        q_low = close.rolling(min(yearly_bars, n - 1)).quantile(0.05)
        q_high = close.rolling(min(yearly_bars, n - 1)).quantile(0.95)
        near_support = ((close - q_low).abs() / close < 0.02)
        near_resistance = ((close - q_high).abs() / close < 0.02)
        cross_year = cross_year + np.where(near_support, 1.0, 0.0)
        cross_year = cross_year + np.where(near_resistance, -1.0, 0.0)
    elif n > 60:
        # Short data: use simple momentum
        ret_60 = close.pct_change(min(60, n - 1))
        cross_year = cross_year + np.where(ret_60 > 0.05, 0.3, np.where(ret_60 < -0.05, -0.3, 0.0))

    # === 9. NEWS & ECONOMIC CORRELATION (7%) ===
    news_score = pd.Series(0.0, index=df.index)
    regime_multiplier = pd.Series(1.0, index=df.index)

    if context is not None:
        market = context.get("market", "")

        # --- VIX regime filter ---
        try:
            if "fred_vixcls" in context:
                vix = context["fred_vixcls"].reindex(df.index, method="ffill")
                if hasattr(vix, 'iloc') and len(vix) > 0:
                    vix_val = vix.iloc[:, 0] if vix.ndim > 1 else vix
                    regime_multiplier = regime_multiplier * np.where(vix_val > 30, 0.5, np.where(vix_val > 40, 0.25, np.where(vix_val < 15, 1.2, 1.0)))
        except Exception:
            pass

        # --- Fed Funds Rate trend ---
        try:
            if "fred_dff" in context:
                fed = context["fred_dff"].reindex(df.index, method="ffill")
                if hasattr(fed, 'iloc') and len(fed) > 0:
                    fed_val = fed.iloc[:, 0] if fed.ndim > 1 else fed
                    fed_change = fed_val.diff(20)
                    regime_multiplier = regime_multiplier * np.where(fed_change > 0.5, 0.8, np.where(fed_change < -0.5, 1.2, 1.0))
        except Exception:
            pass

        # --- Treasury Spread ---
        try:
            if "fred_t10y2y" in context:
                spread = context["fred_t10y2y"].reindex(df.index, method="ffill")
                if hasattr(spread, 'iloc') and len(spread) > 0:
                    spread_val = spread.iloc[:, 0] if spread.ndim > 1 else spread
                    regime_multiplier = regime_multiplier * np.where(spread_val < 0, 0.7, np.where(spread_val > 1, 1.1, 1.0))
        except Exception:
            pass

        # --- CPI / Inflation ---
        try:
            if "fred_cpiaucsl" in context:
                cpi = context["fred_cpiaucsl"].reindex(df.index, method="ffill")
                if hasattr(cpi, 'iloc') and len(cpi) > 0:
                    cpi_val = cpi.iloc[:, 0] if cpi.ndim > 1 else cpi
                    cpi_yoy = cpi_val.diff(12) / cpi_val.shift(12) * 100
                    regime_multiplier = regime_multiplier * np.where(cpi_yoy > 5, 0.8, np.where(cpi_yoy < 2, 1.1, 1.0))
        except Exception:
            pass

        # --- Economic Calendar Events ---
        try:
            if "economic_calendar" in context:
                cal = context["economic_calendar"]
                high_cal = cal[cal["impact"] == "high"] if "impact" in cal.columns else cal
                if len(high_cal) > 0:
                    event_marks = pd.Series(1.0, index=high_cal.index)
                    combined_idx = pd.concat([pd.Series(0.0, index=df.index), event_marks])
                    combined_idx = combined_idx[~combined_idx.index.duplicated(keep='last')].sort_index()
                    near_event = combined_idx.rolling('2D').sum().reindex(df.index).fillna(0)
                    regime_multiplier = regime_multiplier * np.where(near_event > 0, 0.6, 1.0)

                    if "change" in high_cal.columns:
                        pos = pd.Series(0.5, index=high_cal[high_cal["change"] > 0].index)
                        neg = pd.Series(-0.5, index=high_cal[high_cal["change"] < 0].index)
                        sentiment = pd.concat([pos, neg]).sort_index()
                        sentiment = sentiment.reindex(df.index.union(sentiment.index)).sort_index()
                        sentiment = sentiment.ffill(limit=48).reindex(df.index).fillna(0)
                        news_score = news_score + sentiment
        except Exception:
            pass

        # Market-specific
        if market == "btcusd":
            regime_multiplier = regime_multiplier * 0.85
        elif market == "us30":
            regime_multiplier = regime_multiplier * 1.0

    # ============================================================
    # === STRATEGY EVOLUTION: MULTI-STRATEGY CONFIRMATION ===
    # ============================================================
    # Count how many strategies agree on direction (bullish vs bearish)
    bull_count = (
        (trend_score > 0).astype(float) +
        (mr_score > 0).astype(float) +
        (bo_score > 0).astype(float) +
        (mom_score > 0).astype(float) +
        (vol_score > 0).astype(float) +
        (ict_score > 0).astype(float) +
        (ml_score > 0).astype(float) +
        (cross_year > 0).astype(float) +
        (news_score > 0).astype(float)
    )
    bear_count = (
        (trend_score < 0).astype(float) +
        (mr_score < 0).astype(float) +
        (bo_score < 0).astype(float) +
        (mom_score < 0).astype(float) +
        (vol_score < 0).astype(float) +
        (ict_score < 0).astype(float) +
        (ml_score < 0).astype(float) +
        (cross_year < 0).astype(float) +
        (news_score < 0).astype(float)
    )

    # Confirmation multiplier: need 3+ strategies agreeing
    confirmation = pd.Series(0.5, index=df.index)  # Base: weak signal
    confirmation = confirmation + np.where(bull_count >= 3, 0.5, 0.0)  # 3+ bull = normal
    confirmation = confirmation + np.where(bull_count >= 5, 0.5, 0.0)  # 5+ bull = strong
    confirmation = confirmation + np.where(bull_count >= 7, 0.5, 0.0)  # 7+ bull = super signal
    confirmation = confirmation + np.where(bear_count >= 3, 0.5, 0.0)
    confirmation = confirmation + np.where(bear_count >= 5, 0.5, 0.0)
    confirmation = confirmation + np.where(bear_count >= 7, 0.5, 0.0)

    # === SUPER SIGNALS: ICT + Trend + Volume aligned ===
    super_bull = ((ict_score > 0) & (trend_score > 0) & (vol_score > 0)).astype(float)
    super_bear = ((ict_score < 0) & (trend_score < 0) & (vol_score < 0)).astype(float)
    super_signal = super_bull - super_bear

    # === COMBINE ALL 9 STRATEGIES ===
    combined = (
        trend_score * 0.15 +
        mr_score * 0.10 +
        bo_score * 0.10 +
        mom_score * 0.15 +
        vol_score * 0.08 +
        ict_score * 0.18 +
        ml_score * 0.12 +
        cross_year * 0.05 +
        news_score * 0.07
    )

    # Add super signal bonus (when ICT + Trend + Volume all agree)
    combined = combined + super_signal * 0.5

    # Apply confirmation multiplier (high conviction = amplified signal)
    combined = combined * confirmation

    # Apply regime filter
    combined = combined * regime_multiplier

    # === CONVICTION-BASED SIGNAL & SIZING ===
    abs_combined = combined.abs()
    signal = pd.Series(0.0, index=df.index)
    signal = signal + np.where(combined > 0.3, 1.0, 0.0)
    signal = signal + np.where(combined < -0.3, -1.0, 0.0)

    # Graduated position sizing by conviction
    atr_pct = atr_val / close
    base_size = (1.0 / (1.0 + 5 * atr_pct)).clip(0.2, 1.0).fillna(0.5)

    # Scale position with conviction level
    conviction_scale = pd.Series(0.6, index=df.index)  # Default: 60%
    conviction_scale = conviction_scale + np.where(abs_combined > 0.5, 0.2, 0.0)   # Medium: 80%
    conviction_scale = conviction_scale + np.where(abs_combined > 1.0, 0.2, 0.0)   # High: 100%
    conviction_scale = conviction_scale + np.where(abs_combined > 2.0, 0.3, 0.0)   # Very high: 130%
    conviction_scale = np.clip(conviction_scale, 0.3, 1.5)

    sized_signal = signal * base_size * conviction_scale

    # Calculate returns
    price_returns = close.pct_change()
    strategy_returns = sized_signal.shift(1) * price_returns

    # Ensure 1D output with df.index
    result = strategy_returns.fillna(0)
    if hasattr(result, 'squeeze'):
        result = result.squeeze()
    return pd.Series(result.values.flatten(), index=df.index, dtype=float)
