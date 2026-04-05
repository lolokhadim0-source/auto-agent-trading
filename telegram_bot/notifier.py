"""
Telegram notification bot for trading agent experiment results.
"""

import sys
import logging
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)


async def _send_message_async(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured, skipping notification")
        return

    from telegram import Bot

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"🤖 Trading Agent\n{'─'*20}\n{text}",
        parse_mode=None,
    )


def send_notification(text: str):
    """Send a Telegram notification (sync wrapper)."""
    try:
        asyncio.run(_send_message_async(text))
        log.info(f"Telegram notification sent: {text[:50]}...")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def send_test():
    """Send a test message to verify Telegram setup."""
    send_notification("Test message - Trading Agent is connected!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    send_test()
