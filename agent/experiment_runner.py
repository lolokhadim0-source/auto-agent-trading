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

# ── Code Validation ──────────────────────────────────────────────
# Enforces that LLM output actually keeps ALL required components.
# Without this, the LLM can silently drop strategies across iterations.

REQUIRED_CHECKS = [
    # (description, substring that MUST appear in the code)
    ("Trend Following strategy",    "trend_score"),
    ("Mean Reversion strategy",     "mr_score"),
    ("Breakout strategy",           "bo_score"),
    ("Momentum strategy",           "mom_score"),
    ("Volume strategy",             "vol_score"),
    ("ICT Smart Money strategy",    "ict_score"),
    ("ML/GPU strategy",             "ml_score"),
    ("Cross-Year strategy",         "cross_year"),
    ("News/Economic strategy",      "news_score"),
    ("ML model import",             "train_model"),
    ("ML predict import",           "predict_signals"),
    ("fair_value_gap indicator",    "fair_value_gap"),
    ("break_of_structure indicator","break_of_structure"),
    ("liquidity_sweep indicator",   "liquidity_sweep"),
    ("Economic context usage",      "context"),
    ("Strategy signature",          "def strategy(df"),
    ("Returns pd.Series",          "pd.Series"),
]


def validate_strategy_code(code: str) -> tuple[bool, list[str]]:
    """Validate that LLM-generated code contains ALL required components.
    Also checks for syntax errors (truncated LLM output).
    Returns (is_valid, list_of_missing_items).
    """
    missing = []

    # 1. Syntax check — catches truncated LLM output before wasting a backtest
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        missing.append(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
        return (False, missing)

    # 2. Required component checks
    for desc, keyword in REQUIRED_CHECKS:
        if keyword not in code:
            missing.append(desc)
    return (len(missing) == 0, missing)


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

## STRATEGY EVOLUTION:
Study the experiment history. Identify which changes IMPROVED the score and which FAILED.
- DOUBLE DOWN on what works — if a change improved the score, push it further
- COMBINE winning signals into hybrid "super signals" (e.g., ICT sweep + Volume spike + Trend = 3x weight)
- CREATE your own evolved strategy logic on TOP of the 9 base strategies
- The 9 strategies are ingredients — YOUR JOB is to find the best recipe
- ADAPT all indicator periods to the timeframe (5m needs different periods than 1d)

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

## Available ICT indicators (strategy/indicators.py):
You can import: fair_value_gap, order_blocks, break_of_structure, liquidity_sweep, displacement
Example: `from strategy.indicators import fair_value_gap, break_of_structure, liquidity_sweep`

## Cross-Year Pattern Analysis:
Use df.index to find the current month/quarter, then compare price action to the same period in prior years.
Example: rolling 252-bar (1 year) returns, seasonal momentum, anniversary reactions.

Rules:
1. You MUST return the complete modified train.py file content
2. The strategy() MUST accept (df, context=None) as arguments
3. You can import from strategy.indicators (sma, ema, rsi, macd, bollinger_bands, atr, stochastic, adx, vwap, obv, donchian_channel, supertrend, fair_value_gap, order_blocks, break_of_structure, liquidity_sweep, displacement)
4. You can use pandas and numpy
5. Make ONE focused improvement at a time — do NOT remove existing strategies
6. Consider what worked and what didn't in the experiment history
7. **CRITICAL: ALL 9 strategies must remain in the code** (Trend, MeanReversion, Breakout, Momentum, Volume, ICT, ML, Economic, CrossYear). You may tune weights and improve individual strategies but NEVER remove any.
8. **CRITICAL: You MUST use the GPU ML model** (train_model + predict_signals)
9. **CRITICAL: You MUST use economic context data** (FRED data + economic_calendar from context dict)
10. **CRITICAL: You MUST use ICT indicators** (at least fair_value_gap + break_of_structure + liquidity_sweep)
11. Use ONLY vectorized pandas/numpy operations — NO Python for-loops over DataFrame rows
12. Handle edge cases: short DataFrames (<100 bars), missing context, NaN values
13. **CRITICAL BUG PREVENTION**: All boolean masks and Series MUST use df.index as their index. NEVER create a Series with integer/float index and try to apply it to df. Always do: `pd.Series(0.0, index=df.index)` — NOT `pd.Series(0.0, index=range(len(df)))`. Reindex any economic data to df.index BEFORE using it in boolean operations.
14. **CRITICAL**: The final return MUST be `pd.Series(result.values.flatten(), index=df.index, dtype=float)` — ALWAYS use df.index, never the result's own index.

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

    # Validate LLM output has ALL required components
    is_valid, missing = validate_strategy_code(new_code)
    if not is_valid:
        msg = f"LLM code REJECTED — missing: {', '.join(missing)}"
        log.warning(msg)
        log_experiment(iteration, best_score, False, msg)
        return {"iteration": iteration, "improved": False, "error": msg}

    # Write new code
    save_file(TRAIN_FILE, new_code)
    log.info("New strategy code written (passed validation), running backtest...")

    # Run backtest with new code
    try:
        # Reload the module
        if "strategy.train" in sys.modules:
            del sys.modules["strategy.train"]
        from strategy.train import strategy
        results = run_full_evaluation(strategy, fast_mode=True)
        new_score = results["_overall_composite"]
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        traceback.print_exc()
        # Revert
        save_file(TRAIN_FILE, current_code)
        log_experiment(iteration, best_score, False, f"Backtest error: {e}")
        return {"iteration": iteration, "improved": False, "error": str(e)}

    log.info(f"New score: {new_score} | Best score: {best_score}")

    # Collect error details so LLM can learn from failures
    errors = []
    for key, val in results.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "error" in val:
            err_msg = str(val["error"])[:100]  # Truncate long errors
            errors.append(f"{key}: {err_msg}")
    error_summary = "; ".join(errors[:5]) if errors else "no errors"

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
        log_experiment(iteration, new_score, False, f"No improvement: {new_score} vs {best_score}. Errors: {error_summary}")

        return {"iteration": iteration, "improved": False, "old_score": best_score, "new_score": new_score}


def run_inner_loop(max_iterations: int = None, notify_func=None):
    """Run the full inner experiment loop."""
    if max_iterations is None:
        max_iterations = MAX_INNER_ITERATIONS

    program_path = Path(__file__).parent.parent / "program.md"

    best = load_best_score()
    start_iter = best.get("iteration", 0) + 1

    log.info(f"Starting inner loop from iteration {start_iter}, max {max_iterations}")

    for i in range(start_iter, start_iter + max_iterations):
        # Re-read program.md each iteration so meta-agent updates are picked up
        program = program_path.read_text(encoding="utf-8") if program_path.exists() else "Optimize the trading strategy for maximum composite score."
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
