"""Utility functions for the trading bot."""

from typing import Union


def parse_timeframe_to_hours(timeframe: str) -> float:
    """
    Parse a timeframe string (e.g., '1m', '5m', '1h', '4h', '1d') to its duration in hours.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '15m', '30m', '1h', '4h', '1d')
    
    Returns:
        Duration in hours (0.0 if invalid)
    
    Examples:
        >>> parse_timeframe_to_hours('1h')
        1.0
        >>> parse_timeframe_to_hours('4h')
        4.0
        >>> parse_timeframe_to_hours('1d')
        24.0
        >>> parse_timeframe_to_hours('15m')
        0.25
    """
    if not timeframe:
        return 0.0
    
    timeframe = timeframe.lower().strip()
    
    if timeframe.endswith('m'):
        try:
            minutes = int(timeframe[:-1])
            return minutes / 60.0
        except ValueError:
            return 0.0
    elif timeframe.endswith('h'):
        try:
            hours = int(timeframe[:-1])
            return float(hours)
        except ValueError:
            return 0.0
    elif timeframe.endswith('d'):
        try:
            days = int(timeframe[:-1])
            return float(days * 24)
        except ValueError:
            return 0.0
    else:
        return 0.0

