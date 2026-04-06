"""
Backtesting engine (IMMUTABLE - the agent must NOT modify this file).
Like AutoResearch's prepare.py, this provides the fixed evaluation framework.
Loads data, runs a strategy, computes metrics.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, JOBS_DIR

log = logging.getLogger(__name__)


def load_data(market: str, timeframe: str) -> pd.DataFrame:
    """Load market data from parquet files."""
    data_dir = DATA_DIR / market
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory: {data_dir}")

    # Find matching parquet file
    candidates = list(data_dir.glob(f"*{timeframe}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No data file for {market}/{timeframe}")

    df = pd.read_parquet(candidates[0])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_metrics(returns: pd.Series) -> dict:
    """Compute comprehensive trading metrics from a return series."""
    if returns.empty or returns.isna().all():
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "avg_return_per_trade": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
            "composite_score": -999.0,
        }

    # Filter to actual trades (non-zero returns)
    trades = returns[returns != 0]
    num_trades = len(trades)

    # Total return (cumulative)
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0

    # Annualized Sharpe (assume 252 trading days)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # Win rate
    wins = trades[trades > 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(trades[trades < 0].sum()) if len(trades[trades < 0]) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

    # Avg return per trade
    avg_ret = trades.mean() if num_trades > 0 else 0.0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Calmar ratio
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

    # Composite score: weighted combination the agent can try to maximize
    # This is the PRIMARY metric for the experiment loop
    # IMPORTANT: Cap each component so high-frequency datasets (millions of bars)
    # can't produce insane scores via compounding (e.g. BTC 1m = 26000x return)
    composite = (
        0.30 * float(np.clip(sharpe, -5, 5))
        + 0.25 * float(np.clip(total_return * 10, -10, 30))   # cap return contribution
        + 0.20 * float(np.clip((1 + max_dd) * 5, 0, 5))      # drawdown already bounded
        + 0.15 * float(np.clip(profit_factor, 0, 5))          # cap profit factor
        + 0.10 * win_rate * 5                                  # win_rate already 0-1
    )

    return {
        "total_return": round(float(total_return), 6),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "num_trades": int(num_trades),
        "avg_return_per_trade": round(float(avg_ret), 6),
        "volatility": round(float(volatility), 4),
        "calmar_ratio": round(float(calmar), 4),
        "composite_score": round(float(composite), 4),
    }


def load_economic_context() -> dict:
    """Load all economic/news data into a dict for strategy use."""
    context = {}

    # Load FRED economic series
    econ_dir = DATA_DIR / "economic"
    if econ_dir.exists():
        for pfile in econ_dir.glob("*.parquet"):
            try:
                context[f"fred_{pfile.stem}"] = pd.read_parquet(pfile)
            except Exception:
                pass

    # Load economic calendar (Forex Factory style events)
    news_dir = DATA_DIR / "news"
    if news_dir.exists():
        cal_path = news_dir / "economic_calendar.parquet"
        if cal_path.exists():
            try:
                context["economic_calendar"] = pd.read_parquet(cal_path)
            except Exception:
                pass

    return context


def run_backtest(strategy_func, market: str, timeframe: str) -> dict:
    """
    Run a backtest for a given strategy function on specified market/timeframe.

    Args:
        strategy_func: A callable that accepts either:
            - strategy(df: pd.DataFrame) -> pd.Series
            - strategy(df: pd.DataFrame, context: dict) -> pd.Series
        market: "us30" or "btcusd"
        timeframe: timeframe string matching data file

    Returns:
        dict with all metrics
    """
    df = load_data(market, timeframe)
    log.info(f"Backtesting on {market}/{timeframe}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Try calling with context first, fall back to just df
    import inspect
    sig = inspect.signature(strategy_func)
    if len(sig.parameters) >= 2:
        context = load_economic_context()
        context["market"] = market
        context["timeframe"] = timeframe
        returns = strategy_func(df, context)
    else:
        returns = strategy_func(df)

    # Ensure returns is a clean 1D Series (fix shape errors)
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.array(returns).flatten(), index=df.index[:len(returns)])
    if returns.ndim > 1:
        returns = returns.squeeze()
    returns = pd.Series(returns.values.flatten(), index=df.index[:len(returns)], dtype=float)

    metrics = compute_metrics(returns)
    metrics["market"] = market
    metrics["timeframe"] = timeframe
    metrics["data_start"] = str(df.index[0])
    metrics["data_end"] = str(df.index[-1])
    metrics["num_bars"] = len(df)

    return metrics


def run_full_evaluation(strategy_func, fast_mode: bool = True) -> dict:
    """
    Run the strategy across all available data files and aggregate results.
    Returns a summary dict with per-market scores and an overall composite.

    fast_mode: Skip 1-minute data files (>1M rows) to speed up iterations.
               Set False for final validation runs.
    """
    # Skip these huge files in fast mode (saves ~12 min per iteration)
    SKIP_IN_FAST_MODE = {"binance_1m", "alpaca_1Min"}

    results = {}
    all_scores = []

    for market in ["us30", "btcusd"]:
        market_dir = DATA_DIR / market
        if not market_dir.exists():
            continue
        for pfile in sorted(market_dir.glob("*.parquet")):
            tf_key = pfile.stem  # e.g. "yfinance_1d" or "binance_1h"
            if fast_mode and tf_key in SKIP_IN_FAST_MODE:
                log.info(f"  {market}/{tf_key}: SKIPPED (fast mode)")
                continue
            try:
                metrics = run_backtest(strategy_func, market, tf_key)
                results[f"{market}/{tf_key}"] = metrics
                all_scores.append(metrics["composite_score"])
                log.info(f"  {market}/{tf_key}: composite={metrics['composite_score']:.4f}, "
                         f"sharpe={metrics['sharpe_ratio']:.4f}, return={metrics['total_return']:.4f}")
            except Exception as e:
                log.warning(f"  {market}/{tf_key}: FAILED - {e}")
                results[f"{market}/{tf_key}"] = {"error": str(e), "composite_score": -999}
                all_scores.append(-999)

    # Use median so BTC 1m's massive score doesn't drown out weak US30 timeframes
    overall = float(np.median(all_scores)) if all_scores else -999.0
    results["_overall_composite"] = round(overall, 4)
    results["_num_evaluated"] = len(all_scores)
    results["_timestamp"] = datetime.now().isoformat()

    return results


def save_result(results: dict, job_name: str = "latest"):
    """Save backtest results to jobs directory."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = JOBS_DIR / f"{job_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test: import and run the current strategy
    from strategy.train import strategy
    results = run_full_evaluation(strategy)
    print(f"\nOverall composite score: {results['_overall_composite']}")
    save_result(results)
