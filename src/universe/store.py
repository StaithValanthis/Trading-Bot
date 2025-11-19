"""Universe membership persistence layer."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, date
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


class UniverseStore:
    """SQLite-based universe membership storage."""
    
    def __init__(self, db_path: str):
        """
        Initialize universe store.
        
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
        
        # Universe history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                volume_24h REAL,
                open_interest REAL,
                metadata TEXT,
                UNIQUE(date, symbol, action)
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_history_date_symbol
            ON universe_history(date, symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_history_symbol
            ON universe_history(symbol)
        """)
        
        # Symbol metadata cache (for fast lookups)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metadata (
                symbol TEXT PRIMARY KEY,
                last_updated INTEGER NOT NULL,
                listing_date TEXT,
                status TEXT,
                category TEXT,
                quote_coin TEXT,
                min_price REAL,
                max_price REAL,
                metadata TEXT
            )
        """)
        
        # Warm-up tracking (for new listings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS warmup_tracking (
                symbol TEXT PRIMARY KEY,
                first_seen_date TEXT NOT NULL,
                warmup_start_date TEXT NOT NULL,
                eligible_date TEXT,
                last_check_date TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_universe(self, as_of_date: Optional[date] = None) -> Set[str]:
        """
        Get universe membership as of a specific date.
        
        Args:
            as_of_date: Date to query (default: latest)
        
        Returns:
            Set of symbol names in universe
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if as_of_date is None:
            # Get latest date
            cursor.execute("SELECT MAX(date) FROM universe_history")
            result = cursor.fetchone()
            if not result or not result[0]:
                conn.close()
                return set()
            as_of_date_str = result[0]
        else:
            as_of_date_str = as_of_date.isoformat()
        
        # Get all symbols with action='added' or 'kept' up to this date
        # and no subsequent 'removed' action
        cursor.execute("""
            SELECT DISTINCT symbol
            FROM universe_history
            WHERE date <= ? AND action IN ('added', 'kept')
            AND symbol NOT IN (
                SELECT symbol
                FROM universe_history
                WHERE date <= ? AND action = 'removed'
                AND date > (
                    SELECT MAX(date)
                    FROM universe_history
                    WHERE symbol = universe_history.symbol
                    AND date <= ?
                    AND action IN ('added', 'kept')
                )
            )
        """, (as_of_date_str, as_of_date_str, as_of_date_str))
        
        symbols = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        return symbols
    
    def get_current_universe(self) -> Set[str]:
        """Get current (latest) universe."""
        return self.get_universe()
    
    def log_universe_snapshot(
        self,
        snapshot_date: date,
        universe: Set[str],
        changes: Dict[str, Dict]  # symbol -> {action, reason, volume_24h, metadata}
    ):
        """
        Log a universe snapshot with changes.
        
        Args:
            snapshot_date: Date of snapshot
            universe: Set of symbols in universe
            changes: Dictionary of changes (additions/removals)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_str = snapshot_date.isoformat()
        
        try:
            # Log kept symbols (for completeness)
            for symbol in universe:
                if symbol not in changes:
                    # Symbol kept (no change)
                    cursor.execute("""
                        INSERT OR REPLACE INTO universe_history
                        (date, symbol, action, reason, volume_24h, metadata)
                        VALUES (?, ?, 'kept', 'no_change', NULL, NULL)
                    """, (date_str, symbol))
            
            # Log changes (additions/removals)
            for symbol, change_info in changes.items():
                action = change_info.get('action', 'unknown')
                reason = change_info.get('reason', '')
                volume_24h = change_info.get('volume_24h')
                open_interest = change_info.get('open_interest')
                metadata = json.dumps(change_info.get('metadata', {})) if change_info.get('metadata') else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO universe_history
                    (date, symbol, action, reason, volume_24h, open_interest, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (date_str, symbol, action, reason, volume_24h, open_interest, metadata))
            
            conn.commit()
            self.logger.debug(f"Logged universe snapshot for {date_str}: {len(universe)} symbols, {len(changes)} changes")
            
        except Exception as e:
            self.logger.error(f"Error logging universe snapshot: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_history(self, symbol: str) -> List[Dict]:
        """
        Get history of a symbol's universe membership.
        
        Args:
            symbol: Symbol name
        
        Returns:
            List of history records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, action, reason, volume_24h, open_interest, metadata
            FROM universe_history
            WHERE symbol = ?
            ORDER BY date ASC
        """, (symbol,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            date_str, action, reason, volume_24h, open_interest, metadata_json = row
            history.append({
                'date': date_str,
                'action': action,
                'reason': reason,
                'volume_24h': volume_24h,
                'open_interest': open_interest,
                'metadata': json.loads(metadata_json) if metadata_json else {}
            })
        
        return history
    
    def get_changes(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, List[Dict]]:
        """
        Get all universe changes in a date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary mapping symbol to list of changes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, date, action, reason, volume_24h, open_interest, metadata
            FROM universe_history
            WHERE date >= ? AND date <= ? AND action != 'kept'
            ORDER BY symbol, date ASC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        changes = {}
        for row in rows:
            symbol, date_str, action, reason, volume_24h, open_interest, metadata_json = row
            if symbol not in changes:
                changes[symbol] = []
            
            changes[symbol].append({
                'date': date_str,
                'action': action,
                'reason': reason,
                'volume_24h': volume_24h,
                'open_interest': open_interest,
                'metadata': json.loads(metadata_json) if metadata_json else {}
            })
        
        return changes
    
    def update_symbol_metadata(
        self,
        symbol: str,
        listing_date: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        quote_coin: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Update cached symbol metadata.
        
        Args:
            symbol: Symbol name
            listing_date: Listing date (ISO format)
            status: Contract status
            category: Contract category
            quote_coin: Quote currency
            min_price: Minimum price
            max_price: Maximum price
            metadata: Additional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        last_updated = int(datetime.now().timestamp())
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO symbol_metadata
            (symbol, last_updated, listing_date, status, category, quote_coin, min_price, max_price, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, last_updated, listing_date, status, category, quote_coin, min_price, max_price, metadata_json))
        
        conn.commit()
        conn.close()
    
    def get_symbol_metadata(self, symbol: str) -> Optional[Dict]:
        """
        Get cached symbol metadata.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Metadata dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT listing_date, status, category, quote_coin, min_price, max_price, metadata
            FROM symbol_metadata
            WHERE symbol = ?
        """, (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        listing_date, status, category, quote_coin, min_price, max_price, metadata_json = row
        
        return {
            'listing_date': listing_date,
            'status': status,
            'category': category,
            'quote_coin': quote_coin,
            'min_price': min_price,
            'max_price': max_price,
            'metadata': json.loads(metadata_json) if metadata_json else {}
        }
    
    def track_warmup(
        self,
        symbol: str,
        first_seen_date: date,
        warmup_start_date: date
    ):
        """
        Track warm-up period for a new listing.
        
        Args:
            symbol: Symbol name
            first_seen_date: First date symbol was seen
            warmup_start_date: Start of warm-up period
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO warmup_tracking
            (symbol, first_seen_date, warmup_start_date, last_check_date)
            VALUES (?, ?, ?, ?)
        """, (
            symbol,
            first_seen_date.isoformat(),
            warmup_start_date.isoformat(),
            datetime.now().date().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_warmup_status(self, symbol: str) -> Optional[Dict]:
        """
        Get warm-up tracking status for a symbol.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Warm-up status dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT first_seen_date, warmup_start_date, eligible_date, last_check_date
            FROM warmup_tracking
            WHERE symbol = ?
        """, (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        first_seen_date, warmup_start_date, eligible_date, last_check_date = row
        
        return {
            'first_seen_date': first_seen_date,
            'warmup_start_date': warmup_start_date,
            'eligible_date': eligible_date,
            'last_check_date': last_check_date
        }
    
    def mark_warmup_eligible(self, symbol: str, eligible_date: date):
        """
        Mark symbol as eligible after warm-up period.
        
        Args:
            symbol: Symbol name
            eligible_date: Date symbol became eligible
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE warmup_tracking
            SET eligible_date = ?, last_check_date = ?
            WHERE symbol = ?
        """, (eligible_date.isoformat(), datetime.now().date().isoformat(), symbol))
        
        conn.commit()
        conn.close()

