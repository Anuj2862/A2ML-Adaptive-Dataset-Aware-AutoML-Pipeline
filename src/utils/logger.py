import logging
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)

def get_logger(name: str = "A2ML") -> logging.Logger:
    """Returns a configured logger that writes to both console and logs/a2ml.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
                             datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(f"logs/a2ml_{datetime.now().strftime('%Y%m%d')}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
