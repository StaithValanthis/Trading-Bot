"""OHLCV data storage in SQLite."""

import sqlite3
import pandas as pd
import math
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from ..logging_utils import get_logger

logger = get_logger(__name__)


class OHLCVStore:
    """SQLite-based OHLCV data storage."""
    
    def __init__(self, db_path: str):
        """
        Initialize OHLCV store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # OHLCV table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
            """
        )

        # Create index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp
            ON ohlcv(symbol, timeframe, timestamp)
            """
        )

        conn.commit()
        conn.close()
    
    def store_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        ohlcv_data: List[List]
    ):
        """
        Store OHLCV data in database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            ohlcv_data: List of [timestamp, open, high, low, close, volume]
        """
        if not ohlcv_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for candle in ohlcv_data:
                timestamp, open_price, high, low, close, volume = candle
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timeframe, timestamp, open_price, high, low, close, volume))
            
            conn.commit()
            self.logger.debug(f"Stored {len(ohlcv_data)} candles for {symbol} {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error storing OHLCV data: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            since: Minimum timestamp in milliseconds
            limit: Maximum number of rows to return (returns most recent candles)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Data is sorted by timestamp ASC (oldest first) for proper time series processing
        """
        conn = sqlite3.connect(self.db_path)
        
        # CRITICAL: When using LIMIT, we need to get the MOST RECENT candles
        # So we order DESC first, apply LIMIT, then sort ASC for proper time series order
        if limit:
            # Get the most recent N candles
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM (
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            
            query += """
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                ORDER BY timestamp ASC
            """
            params.append(limit)
        else:
            # No limit - get all data sorted ASC
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            
            query += " ORDER BY timestamp ASC"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                # Ensure sorted by datetime ASC (oldest to newest)
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting OHLCV data: {e}")
            conn.close()
            raise
    
    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[int]:
        """
        Get the latest timestamp for a symbol/timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
        
        Returns:
            Latest timestamp in milliseconds, or None if no data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT MAX(timestamp)
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else None
            
        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {e}")
            conn.close()
            return None
    
    def get_earliest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[int]:
        """
        Get the earliest timestamp for a symbol/timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
        
        Returns:
            Earliest timestamp in milliseconds, or None if no data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT MIN(timestamp)
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else None
            
        except Exception as e:
            self.logger.error(f"Error getting earliest timestamp: {e}")
            conn.close()
            return None
    
    def delete_old_data(self, days_to_keep: int = 365):
        """
        Delete OHLCV data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_timestamp = int((datetime.now().timestamp() - days_to_keep * 86400) * 1000)
        
        try:
            cursor.execute("""
                DELETE FROM ohlcv
                WHERE timestamp < ?
            """, (cutoff_timestamp,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Deleted {deleted} old OHLCV records")
            
        except Exception as e:
            self.logger.error(f"Error deleting old data: {e}")
            conn.rollback()
            conn.close()
    
    def check_health(
        self,
        symbol: str,
        timeframe: str,
        required_days: int,
        max_gap_pct: float = 5.0
    ) -> dict:
        """
        Check OHLCV data health for a symbol/timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            required_days: Minimum calendar days of history required
            max_gap_pct: Maximum allowed gap percentage (default: 5.0)
        
        Returns:
            Dictionary with health metrics:
            {
                'symbol': str,
                'timeframe': str,
                'required_days': int,
                'history_days': float,
                'actual_bars': int,
                'expected_bars': int,
                'gap_pct': float,
                'earliest_timestamp': int or None,
                'latest_timestamp': int or None,
                'is_healthy': bool,
                'needs_backfill': bool,
                'missing_ranges': List[Tuple[int, int]],  # List of (start_ms, end_ms) gaps
                'issues': List[str]  # List of issue descriptions
            }
        """
        from ..utils import parse_timeframe_to_hours
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'required_days': required_days,
            'history_days': 0.0,
            'actual_bars': 0,
            'expected_bars': 0,
            'gap_pct': 0.0,
            'earliest_timestamp': None,
            'latest_timestamp': None,
            'is_healthy': False,
            'needs_backfill': False,
            'missing_ranges': [],
            'issues': []
        }
        
        try:
            # Get all data for this symbol/timeframe
            df = self.get_ohlcv(symbol, timeframe)
            
            if df.empty:
                result['issues'].append('no_data')
                result['needs_backfill'] = True
                return result
            
            # Calculate metrics
            hours_per_bar = parse_timeframe_to_hours(timeframe)
            if hours_per_bar <= 0:
                result['issues'].append('invalid_timeframe')
                return result
            
            candles_per_day = max(24.0 / hours_per_bar, 1.0)
            required_bars = int(math.ceil(required_days * candles_per_day))
            
            # Get timestamps
            earliest_ts = df.index[0]
            latest_ts = df.index[-1]
            
            # Convert to datetime if needed
            if hasattr(earliest_ts, 'timestamp'):
                earliest_dt = earliest_ts.to_pydatetime() if hasattr(earliest_ts, 'to_pydatetime') else earliest_ts
                latest_dt = latest_ts.to_pydatetime() if hasattr(latest_ts, 'to_pydatetime') else latest_ts
            else:
                earliest_dt = earliest_ts
                latest_dt = latest_ts
            
            # Ensure UTC-aware
            from datetime import timezone
            if earliest_dt.tzinfo is None:
                earliest_dt = earliest_dt.replace(tzinfo=timezone.utc)
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            
            # Calculate history days
            history_days = (latest_dt - earliest_dt).total_seconds() / 86400.0
            actual_bars = len(df)
            expected_bars = max(int(history_days * candles_per_day), 1)
            gap_pct = (1 - actual_bars / expected_bars) * 100 if expected_bars > 0 else 100.0
            
            # Convert timestamps to milliseconds
            if hasattr(earliest_ts, 'timestamp'):
                earliest_timestamp_ms = int(earliest_ts.timestamp() * 1000)
            else:
                earliest_timestamp_ms = int(earliest_dt.timestamp() * 1000)
            
            if hasattr(latest_ts, 'timestamp'):
                latest_timestamp_ms = int(latest_ts.timestamp() * 1000)
            else:
                latest_timestamp_ms = int(latest_dt.timestamp() * 1000)
            
            # Detect missing ranges (gaps)
            missing_ranges = []
            if len(df) > 1:
                timestamps = df.index
                ms_per_bar = hours_per_bar * 3600 * 1000
                
                for i in range(len(timestamps) - 1):
                    current_ts = timestamps[i]
                    next_ts = timestamps[i + 1]
                    
                    # Convert to milliseconds
                    if hasattr(current_ts, 'timestamp'):
                        current_ms = int(current_ts.timestamp() * 1000)
                    elif hasattr(current_ts, 'to_pydatetime'):
                        current_ms = int(current_ts.to_pydatetime().timestamp() * 1000)
                    else:
                        # Assume it's already a timestamp in some form
                        try:
                            current_ms = int(pd.Timestamp(current_ts).timestamp() * 1000)
                        except:
                            continue
                    
                    if hasattr(next_ts, 'timestamp'):
                        next_ms = int(next_ts.timestamp() * 1000)
                    elif hasattr(next_ts, 'to_pydatetime'):
                        next_ms = int(next_ts.to_pydatetime().timestamp() * 1000)
                    else:
                        try:
                            next_ms = int(pd.Timestamp(next_ts).timestamp() * 1000)
                        except:
                            continue
                    
                    # Check if gap is larger than expected (more than 1.5x the bar duration)
                    expected_next_ms = current_ms + ms_per_bar
                    if next_ms > expected_next_ms + (ms_per_bar * 0.5):  # Allow 0.5 bar tolerance
                        missing_ranges.append((expected_next_ms, next_ms))
            
            # Update result
            result['history_days'] = history_days
            result['actual_bars'] = actual_bars
            result['expected_bars'] = expected_bars
            result['gap_pct'] = gap_pct
            result['earliest_timestamp'] = earliest_timestamp_ms
            result['latest_timestamp'] = latest_timestamp_ms
            result['missing_ranges'] = missing_ranges
            
            # Determine health
            if history_days < required_days:
                result['issues'].append('insufficient_history')
                result['needs_backfill'] = True
            
            if gap_pct > max_gap_pct:
                result['issues'].append('too_many_gaps')
                result['needs_backfill'] = True
            
            if not result['issues']:
                result['is_healthy'] = True
            
        except Exception as e:
            self.logger.error(f"Error checking health for {symbol} {timeframe}: {e}", exc_info=True)
            result['issues'].append(f'check_error: {str(e)}')
            result['needs_backfill'] = True
        
        return result

