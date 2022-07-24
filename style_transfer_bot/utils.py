import logging
import sys

from style_transfer_bot.config import LOG_LEVEL


def get_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
