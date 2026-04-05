"""
Inner Loop: AutoResearch-style experiment runner.
Reads strategy/train.py, asks LLM to modify it, backtests, keeps improvements.
"""

import sys
import json
import shutil
import logging
import traceback
from datetime import datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    ANTHROPIC_API_KEY, STRATEGY_DIR, JOBS_DIR,
    LLM_MODEL, LLM_MAX_TOKENS, MAX_INNER_ITERATIONS,
)
from strategy.backtest import run_full_evaluation, save_result

log = logging.getLogger(__name__)

TRAIN_FILE = STRATEGY_DIR / "train.py"
INDICATORS_FILE = STRATEGY_DIR / "indicators.py"
BEST_SCORE_FILE = JOBS_DIR / "best_score.json"


def load_best_score() -> dict:
    if BEST_SCORE_FILE.exists():
        with open(BEST_SCORE_FILE) as f:
            return json.load(f)
    return {"score": -999.0, "iteration": 0}


def save_best_score(score: float, iteration: int, results: dict):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"score": score, "iteration": iteration, "timestamp": datetime.now().isoformat(), "results": results}
    with open(BEST_SCORE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def get_experiment_history() -> str:
    """Load recent experiment logs for context."""
    history_file = JOBS_DIR / "experiment_history.jsonl"
    if not history_file.exists():
        return "No previous experiments."

    lines = history_file.read_text().strip().split("\n")
    recent = lines[-20:]  # last 20 experiments
    return "\n".join(recent)


def log_experiment(iteration: int, score: float, improved: bool, description: str):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    history_file = JOBS_DIR / "experiment_history.jsonl"
    entry = {
        "iteration": iteration,
        "score": score,
        "improved": improved,
        "description": description,
        "timestamp": datetime.now().isoformat(),
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ask_llm_for_modification(current_code: str, indicators_code: str, best_score: float, history: str, program: str) -> str:
    """Ask Claude to propose a modification to train.py."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are an autonomous trading strategy optimizer. Your job is to modify the trading strategy code to improve its backtesting performance.

## Current strategy code (strategy/train.py - the ONLY file you can modify):
```python
{current_code}
```

## Available indicators (strategy/indicators.py - read only, use these):
```python
{indicators_code}
```

## Current best composite score: {best_score}

## Recent experiment history:
{history}

## Research directives:
{program}

## Your task:
Propose a MODIFIED version of strategy/train.py that you believe will achieve a higher composite score.
The composite score rewards: Sharpe ratio (30%), total return (25%), low drawdown (20%), profit factor (15%), win rate (10%).

## Available economic/news data (optional second argument):
If your strategy function accepts a second argument `context: dict`, it will receive:
- context["fred_dff"] - Federal Funds Rate (daily, decades of history)
- context["fred_t10y2y"] - 10Y-2Y Treasury Spread (daily)
- context["fred_dtwexbgs"] - Trade-weighted USD Index (daily)
- context["fred_unrate"] - Unemployment Rate (monthly)
- context["fred_cpiaucsl"] - CPI (monthly)
- context["economic_calendar"] - 45K+ economic events with columns: event, country, impact (high/medium/low), actual, previous, change
- context["market"] - "us30" or "btcusd"
- context["timeframe"] - the current timeframe string
Each value is a pandas DataFrame indexed by date. Merge with df.index using pd.merge_asof() or reindex.

## GPU-accelerated ML Model (strategy/ml_model.py):
You can import and use the LSTM model for GPU-accelerated price prediction:
```python
from strategy.ml_model import train_model, predict_signals
model = train_model(df, lookback=60, epochs=20)  # Trains on GPU automatically
ml_signals = predict_signals(model, df, lookback=60)  # Returns 1=long, -1=short, 0=flat
```
- train_model() trains an LSTM neural network on the GPU (NVIDIA A100-SXM4-80GB) using OHLCV data
- predict_signals() generates buy/sell signals from the trained model
- You can combine ML signals with technical indicators for confirmation
- The model learns patterns from price action, volume, and volatility automatically

Rules:
1. You MUST return the complete modified train.py file content
2. The strategy() function signature can be EITHER: strategy(df: pd.DataFrame) -> pd.Series OR strategy(df: pd.DataFrame, context: dict) -> pd.Series
3. You can import from strategy.indicators (sma, ema, rsi, macd, bollinger_bands, atr, stochastic, adx, vwap, obv, donchian_channel, supertrend)
4. You can use pandas and numpy
5. Make ONE focused change at a time (don't rewrite everything)
6. Consider what worked and what didn't in the experiment history
7. Be creative: try different indicators, timeframe analysis, risk management, position sizing, AND economic data correlation

Return ONLY the complete Python code for train.py, nothing else. No markdown, no explanation, just the code."""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    code = response.content[0].text.strip()

    # Strip markdown code fences if present
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def run_experiment(iteration: int, program: str, notify_func=None) -> dict:
    """Run a single experiment iteration."""
    best = load_best_score()
    best_score = best["score"]

    log.info(f"{'='*60}")
    log.info(f"Experiment iteration {iteration} | Best score: {best_score}")
    log.info(f"{'='*60}")

    # Read current files
    current_code = load_file(TRAIN_FILE)
    indicators_code = load_file(INDICATORS_FILE)
    history = get_experiment_history()

    # Backup current train.py
    backup_path = JOBS_DIR / f"train_backup_iter{iteration}.py"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAIN_FILE, backup_path)

    # Ask LLM for modification
    log.info("Asking LLM for strategy modification...")
    try:
        new_code = ask_llm_for_modification(current_code, indicators_code, best_score, history, program)
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        log_experiment(iteration, best_score, False, f"LLM error: {e}")
        return {"iteration": iteration, "improved": False, "error": str(e)}

    # Write new code
    save_file(TRAIN_FILE, new_code)
    log.info("New strategy code written, running backtest...")

    # Run backtest with new code
    try:
        # Reload the module
        if "strategy.train" in sys.modules:
            del sys.modules["strategy.train"]
        from strategy.train import strategy
        results = run_full_evaluation(strategy, fast_mode=False)
        new_score = results["_overall_composite"]
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        traceback.print_exc()
        # Revert
        save_file(TRAIN_FILE, current_code)
        log_experiment(iteration, best_score, False, f"Backtest error: {e}")
        return {"iteration": iteration, "improved": False, "error": str(e)}

    log.info(f"New score: {new_score} | Best score: {best_score}")

    if new_score > best_score:
        log.info(f"IMPROVEMENT! {best_score} -> {new_score} (+{new_score - best_score:.4f})")
        save_best_score(new_score, iteration, results)
        save_result(results, f"best_iter{iteration}")
        log_experiment(iteration, new_score, True, f"Improved from {best_score} to {new_score}")

        if notify_func:
            notify_func(
                f"Iteration {iteration}: NEW BEST!\n"
                f"Score: {best_score:.4f} -> {new_score:.4f}\n"
                f"Sharpe: {results.get('us30/yfinance_1d', {}).get('sharpe_ratio', 'N/A')}\n"
                f"Return: {results.get('us30/yfinance_1d', {}).get('total_return', 'N/A')}"
            )

        return {"iteration": iteration, "improved": True, "old_score": best_score, "new_score": new_score}
    else:
        log.info(f"No improvement ({new_score} <= {best_score}), reverting.")
        save_file(TRAIN_FILE, current_code)
        log_experiment(iteration, new_score, False, f"No improvement: {new_score} vs {best_score}")

        return {"iteration": iteration, "improved": False, "old_score": best_score, "new_score": new_score}


def run_inner_loop(max_iterations: int = None, notify_func=None):
    """Run the full inner experiment loop."""
    if max_iterations is None:
        max_iterations = MAX_INNER_ITERATIONS

    program_path = Path(__file__).parent.parent / "program.md"
    program = program_path.read_text(encoding="utf-8") if program_path.exists() else "Optimize the trading strategy for maximum composite score."

    best = load_best_score()
    start_iter = best.get("iteration", 0) + 1

    log.info(f"Starting inner loop from iteration {start_iter}, max {max_iterations}")

    for i in range(start_iter, start_iter + max_iterations):
        try:
            result = run_experiment(i, program, notify_func)
            log.info(f"Iteration {i} result: {result}")
        except Exception as e:
            log.error(f"Iteration {i} crashed: {e}")
            traceback.print_exc()
            continue

    log.info("Inner loop complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_inner_loop(max_iterations=5)
