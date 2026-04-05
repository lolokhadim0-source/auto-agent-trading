"""
Standalone runner script for autonomous agent execution.
Can be scheduled via Windows Task Scheduler or cron.
Usage: python run_agent.py [--iterations N]
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import run


def main():
    parser = argparse.ArgumentParser(description="Run autonomous trading agent")
    parser.add_argument("-n", "--iterations", type=int, default=50,
                        help="Number of iterations (default: 50)")
    parser.add_argument("--no-notify", action="store_true",
                        help="Disable Telegram notifications")
    args = parser.parse_args()

    Path("jobs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("jobs/agent.log", mode="a"),
        ]
    )

    run(max_iterations=args.iterations, notify=not args.no_notify)


if __name__ == "__main__":
    main()
