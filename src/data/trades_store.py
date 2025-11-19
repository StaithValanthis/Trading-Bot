"""Trade storage in SQLite."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """Round-trip trade record."""

    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    reason: str


class TradesStore:
    """SQLite-backed trade storage."""

    def __init__(self, db_path: str):
        """
        Initialize trades store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize trades table if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                pnl REAL NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_exit_time
            ON trades(symbol, exit_time)
            """
        )

        conn.commit()
        conn.close()

    def log_trade(self, trade: TradeRecord) -> None:
        """Insert a trade record into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO trades
                (symbol, side, size, entry_price, exit_price,
                 entry_time, exit_time, pnl, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.symbol,
                    trade.side,
                    trade.size,
                    trade.entry_price,
                    trade.exit_price,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.pnl,
                    trade.reason,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging trade for {trade.symbol}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_trades_between(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict]:
        """
        Get trades closed between start and end (by exit_time).

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of trade dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT symbol, side, size, entry_price, exit_price,
                       entry_time, exit_time, pnl, reason
                FROM trades
                WHERE exit_time >= ? AND exit_time <= ?
                ORDER BY exit_time ASC
                """,
                (start.isoformat(), end.isoformat()),
            )
            rows = cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error querying trades between {start} and {end}: {e}")
            conn.close()
            return []

        conn.close()

        trades: List[Dict] = []
        for (
            symbol,
            side,
            size,
            entry_price,
            exit_price,
            entry_time,
            exit_time,
            pnl,
            reason,
        ) in rows:
            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": datetime.fromisoformat(entry_time),
                    "exit_time": datetime.fromisoformat(exit_time),
                    "pnl": pnl,
                    "reason": reason,
                }
            )

        return trades


