"""
Main Agent: Orchestrates the nested AutoAgent + AutoResearch loops.
Outer loop (meta-agent) runs every N iterations.
Inner loop (experiment runner) runs continuously.
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MAX_INNER_ITERATIONS, OUTER_LOOP_EVERY_N
from agent.experiment_runner import run_experiment, load_best_score
from agent.meta_agent import analyze_and_update
from telegram_bot.notifier import send_notification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "jobs" / "agent.log", mode="a"),
    ]
)
log = logging.getLogger("agent")


def run(max_iterations: int = None, notify: bool = True):
    """Run the full nested agent loop."""
    if max_iterations is None:
        max_iterations = MAX_INNER_ITERATIONS

    notify_func = send_notification if notify else None
    program_path = Path(__file__).parent.parent / "program.md"
    program = program_path.read_text(encoding="utf-8") if program_path.exists() else ""

    best = load_best_score()
    start_iter = best.get("iteration", 0) + 1

    log.info(f"Starting autonomous trading agent")
    log.info(f"  Iterations: {start_iter} to {start_iter + max_iterations - 1}")
    log.info(f"  Meta-agent runs every {OUTER_LOOP_EVERY_N} iterations")

    if notify_func:
        notify_func(f"Agent started! Running {max_iterations} iterations from #{start_iter}")

    for i in range(start_iter, start_iter + max_iterations):
        log.info(f"\n{'='*60}")
        log.info(f"ITERATION {i}")
        log.info(f"{'='*60}")

        # Inner loop: run experiment
        try:
            result = run_experiment(i, program, notify_func)
        except Exception as e:
            log.error(f"Experiment {i} failed: {e}")
            continue

        # Outer loop: meta-agent analysis
        if i % OUTER_LOOP_EVERY_N == 0:
            log.info(f"\n--- Meta-agent analysis (every {OUTER_LOOP_EVERY_N} iterations) ---")
            try:
                analyze_and_update(notify_func)
                # Reload program after meta-agent updates it
                if program_path.exists():
                    program = program_path.read_text(encoding="utf-8")
            except Exception as e:
                log.error(f"Meta-agent failed: {e}")

    final_best = load_best_score()
    log.info(f"\nAgent complete. Final best score: {final_best['score']}")

    if notify_func:
        notify_func(f"Agent complete!\nFinal best score: {final_best['score']}\nTotal iterations: {max_iterations}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Trading Agent")
    parser.add_argument("-n", "--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--no-notify", action="store_true", help="Disable Telegram notifications")
    args = parser.parse_args()

    (Path(__file__).parent.parent / "jobs").mkdir(parents=True, exist_ok=True)
    run(max_iterations=args.iterations, notify=not args.no_notify)


if __name__ == "__main__":
    main()
