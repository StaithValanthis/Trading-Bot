"""
Logging utilities for the trading bot.

This module provides a centralized, production-grade logging system that:
- Configures the root logger so all loggers inherit handlers
- Writes to both files (rotating) and stderr (for systemd/journald)
- Uses UTC timestamps
- Redacts secrets from logs
- Ensures immediate log flushing for systemd
"""

import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
import re


class UTCFormatter(logging.Formatter):
    """Formatter that uses UTC time."""
    
    converter = time.gmtime  # Use UTC time


class SecretFilter(logging.Filter):
    """Filter to redact secrets from log messages."""
    
    # Patterns to match API keys, secrets, webhook URLs, etc.
    SECRET_PATTERNS = [
        # API keys (e.g., api_key="...", apiKey: "...", BYBIT_API_KEY=...)
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'api_key="REDACTED"'),
        (r'api[_-]?secret["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'api_secret="REDACTED"'),
        (r'secret["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})["\']?', r'secret="REDACTED"'),
        # Webhook URLs
        (r'(webhook[_-]?url["\']?\s*[:=]\s*["\']?https?://[^\s"\'<>]+)', r'webhook_url="REDACTED"'),
        # Full API keys/secrets in messages (standalone long strings)
        (r'\b([A-Za-z0-9]{32,})\b', lambda m: 'REDACTED' if len(m.group(1)) >= 32 else m.group(1)),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to redact secrets."""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.SECRET_PATTERNS:
                if callable(replacement):
                    msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
                else:
                    msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        
        if hasattr(record, 'args') and record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, replacement in self.SECRET_PATTERNS:
                        if callable(replacement):
                            args[i] = re.sub(pattern, replacement, arg, flags=re.IGNORECASE)
                        else:
                            args[i] = re.sub(pattern, replacement, arg, flags=re.IGNORECASE)
            record.args = tuple(args)
        
        return True


# Global flag to track if logging is initialized
_logging_initialized = False


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    max_log_size_mb: int = 10,
    backup_count: int = 5,
    service_name: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Set up centralized logging configuration for the entire application.
    
    This function configures the ROOT logger, so all loggers created with
    logging.getLogger(__name__) will inherit the handlers automatically.
    
    Args:
        log_dir: Directory for log files (relative to current working directory)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_log_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
        service_name: Optional service name for separate log files (e.g., "live", "optimizer")
        force: If True, reinitialize logging even if already initialized
    
    Returns:
        None (configures root logger globally)
    """
    global _logging_initialized
    
    if _logging_initialized and not force:
        # Logging already initialized, just update level if needed
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger (this affects all loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Prevent propagation to avoid duplicate logs (we handle everything here)
    root_logger.propagate = False
    
    # Create secret filter
    secret_filter = SecretFilter()
    
    # Console handler (stderr for systemd/journald)
    # Use stderr instead of stdout to ensure unbuffered output and proper systemd integration
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    console_formatter = UTCFormatter(
        '%(asctime)s [%(levelname)8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(secret_filter)
    # Force unbuffered output for systemd
    console_handler.stream.reconfigure(line_buffering=True)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    # Use service-specific filename if provided, otherwise generic
    if service_name:
        log_filename = f"bot-{service_name}.log"
    else:
        log_filename = "bot.log"
    
    log_file = log_path / log_filename
    max_bytes = max_log_size_mb * 1024 * 1024
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',  # Ensure UTF-8 encoding
    )
    file_handler.setLevel(logging.DEBUG)  # File logs everything (DEBUG and above)
    file_formatter = UTCFormatter(
        '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(secret_filter)
    root_logger.addHandler(file_handler)
    
    # Also configure a named logger for backwards compatibility
    named_logger = logging.getLogger("trading_bot")
    named_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    named_logger.propagate = True  # Propagate to root
    
    # Log initialization
    root_logger.info("=" * 80)
    root_logger.info("Logging system initialized")
    root_logger.info(f"  Log level: {level.upper()}")
    root_logger.info(f"  Log directory: {log_path.absolute()}")
    root_logger.info(f"  Log file: {log_file.absolute()}")
    root_logger.info(f"  Service name: {service_name or 'default'}")
    root_logger.info("=" * 80)
    
    # Force flush to ensure logs appear immediately
    sys.stderr.flush()
    
    _logging_initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    This function creates a logger that will automatically inherit handlers
    from the root logger configured by setup_logging().
    
    Args:
        name: Logger name (defaults to calling module's __name__)
              If None and called from a module, use that module's __name__
    
    Returns:
        Logger instance
    """
    if name is None:
        # Try to get the calling module's name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'trading_bot')
        else:
            name = "trading_bot"
    
    logger = logging.getLogger(name)
    
    # Add secret filter if not already added (for this logger)
    if not any(isinstance(f, SecretFilter) for f in logger.filters):
        logger.addFilter(SecretFilter())
    
    return logger


def flush_logs() -> None:
    """Force flush all log handlers."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.flush()
    sys.stderr.flush()
    sys.stdout.flush()


# Initialize basic logging at module import time (before setup_logging is called)
# This ensures early log messages don't get lost
_temp_handler = logging.StreamHandler(sys.stderr)
_temp_handler.setFormatter(UTCFormatter('%(asctime)s [%(levelname)8s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S UTC'))
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above until proper setup
    handlers=[_temp_handler],
    force=True,  # Override any existing basicConfig
)
