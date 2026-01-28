"""Logging configuration."""

import sys
from loguru import logger
from pathlib import Path

from src.config import config


def setup_logging(log_file: Path = None, level: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (uses config default if None)
        level: Log level (uses config default if None)
    """
    log_file = log_file or config.LOG_FILE
    level = level or config.LOG_LEVEL
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation="10 MB",
        retention="1 week"
    )
    
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")


# Initialize logging on import
setup_logging()
