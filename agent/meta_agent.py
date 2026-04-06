"""
Outer Loop: AutoAgent-style meta-agent.
Analyzes experiment results, updates program.md research directions,
and potentially modifies the experiment runner itself.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    ANTHROPIC_API_KEY, JOBS_DIR, LLM_MODEL, LLM_MAX_TOKENS,
    OUTER_LOOP_EVERY_N,
)

log = logging.getLogger(__name__)

PROGRAM_FILE = Path(__file__).parent.parent / "program.md"
HISTORY_FILE = JOBS_DIR / "experiment_history.jsonl"
META_LOG_FILE = JOBS_DIR / "meta_agent_log.jsonl"


def load_experiment_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def load_program() -> str:
    if PROGRAM_FILE.exists():
        return PROGRAM_FILE.read_text(encoding="utf-8")
    return ""


def save_program(content: str):
    PROGRAM_FILE.write_text(content, encoding="utf-8")


def log_meta_action(action: str, details: str):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"action": action, "details": details, "timestamp": datetime.now().isoformat()}
    with open(META_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def analyze_and_update(notify_func=None):
    """
    Meta-agent analyzes experiment history and updates research directions.
    This is the outer loop that runs every N inner iterations.
    """
    history = load_experiment_history()
    if not history:
        log.info("No experiment history yet, skipping meta-analysis.")
        return

    current_program = load_program()

    # Prepare analysis prompt
    recent = history[-50:]  # Last 50 experiments
    improvements = [h for h in recent if h.get("improved")]
    failures = [h for h in recent if not h.get("improved")]

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are a meta-agent overseeing an autonomous trading strategy optimization system.

## How the system works:
- An inner loop agent repeatedly modifies strategy/train.py (a trading strategy)
- Each modification is backtested on US30 (Dow Jones) and BTCUSD markets
- The composite score rewards: Sharpe ratio (30%), total return (25%), low drawdown (20%), profit factor (15%), win rate (10%)
- Changes that improve the score are kept; others are reverted

## Current research directives (program.md):
{current_program}

## Experiment history summary:
- Total experiments: {len(recent)}
- Improvements: {len(improvements)}
- Failures: {len(failures)}
- Success rate: {len(improvements)/len(recent)*100:.1f}%

## Recent improvements:
{json.dumps(improvements[-10:], indent=2)}

## Recent failures:
{json.dumps(failures[-10:], indent=2)}

## Your task:
Analyze the experiment history and write an UPDATED program.md that will guide the inner loop agent to better strategies.

Consider:
1. What types of changes led to improvements? Do more of those.
2. What types of changes failed? Avoid those directions.
3. Are there unexplored areas (new indicators, risk management, multi-timeframe analysis)?
4. Should the agent focus on specific markets or timeframes?
5. Are there diminishing returns? Should the agent try fundamentally different approaches?

Return ONLY the updated program.md content. Keep it focused and actionable.
Start with "# Research Directives" as the heading."""

    log.info("Meta-agent analyzing experiment history...")
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        new_program = response.content[0].text.strip()
    except Exception as e:
        log.error(f"Meta-agent LLM call failed: {e}")
        log_meta_action("error", str(e))
        return

    # Save updated program
    save_program(new_program)
    log_meta_action("update_program", f"Updated program.md based on {len(recent)} experiments")
    log.info("Program.md updated with new research directions.")

    if notify_func:
        notify_func(
            f"Meta-agent update:\n"
            f"Analyzed {len(recent)} experiments ({len(improvements)} improvements)\n"
            f"Updated research directives in program.md"
        )


def should_run(current_iteration: int) -> bool:
    """Check if the meta-agent should run based on iteration count."""
    return current_iteration > 0 and current_iteration % OUTER_LOOP_EVERY_N == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyze_and_update()
