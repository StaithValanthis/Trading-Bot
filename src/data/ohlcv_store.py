"""OHLCV data storage in SQLite."""

import sqlite3
import pandas as pd
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

