"""Order execution logging in SQLite."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OrderRecord:
    symbol: str
    side: str           # 'buy' or 'sell'
    size: float
    price: Optional[float]
    order_type: str     # 'market' or 'limit'
    reason: str         # e.g. 'reconcile', 'close_position'
    timestamp: datetime


class OrdersStore:
    """SQLite-backed order execution log."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                price REAL,
                order_type TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_orders_symbol_created_at
            ON orders(symbol, created_at)
            """
        )

        conn.commit()
        conn.close()

    def log_order(self, order: OrderRecord) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO orders (symbol, side, size, price, order_type, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.symbol,
                    order.side,
                    order.size,
                    order.price,
                    order.order_type,
                    order.reason,
                    order.timestamp.isoformat(),
                ),
            )
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging order for {order.symbol}: {e}")
            conn.rollback()
        finally:
            conn.close()


