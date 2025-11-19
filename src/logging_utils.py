"""Logging utilities for the trading bot."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import re


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    max_log_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_log_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_path / "trading_bot.log"
    max_bytes = max_log_size_mb * 1024 * 1024
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class SecretFilter(logging.Filter):
    """Filter to redact secrets from log messages."""
    
    # Patterns to match API keys, secrets, etc.
    SECRET_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'api_key="REDACTED"'),
        (r'api[_-]?secret["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'api_secret="REDACTED"'),
        (r'secret["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'secret="REDACTED"'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to redact secrets."""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.SECRET_PATTERNS:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        
        if hasattr(record, 'args') and record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, replacement in self.SECRET_PATTERNS:
                        args[i] = re.sub(pattern, replacement, arg, flags=re.IGNORECASE)
            record.args = tuple(args)
        
        return True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to 'trading_bot')
    
    Returns:
        Logger instance
    """
    logger_name = name if name else "trading_bot"
    logger = logging.getLogger(logger_name)
    
    # Add secret filter if not already added
    if not any(isinstance(f, SecretFilter) for f in logger.filters):
        logger.addFilter(SecretFilter())
    
    return logger

