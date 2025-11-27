"""SQLite-backed storage for optimizer runs and results.

This module provides a thin abstraction around a few tables that track:
  - Optimization runs (per timeframe / symbol universe / date range)
  - Per-parameter-set results for each run
  - Best-known parameters per timeframe/universe

The goal is to make long-running optimizations resumable and their results
queryable, without changing the core optimization logic.
"""

import json
import sqlite3
import time
import hashlib
from typing import Dict, List, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)


class OptimizerStore:
    """Helper for persisting optimizer runs/results in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        """Create optimizer tables if they don't already exist."""
        conn = self._connect()
        cur = conn.cursor()

        # Runs table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_runs (
                run_id TEXT PRIMARY KEY,
                timeframe TEXT NOT NULL,
                symbols_hash TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                config_version TEXT,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )

        # Per-parameter-set results
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_param_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                param_index INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                sharpe_is REAL,
                sharpe_oos REAL,
                dd_is REAL,
                dd_oos REAL,
                trades_is INTEGER,
                trades_oos INTEGER,
                return_pct_is REAL,
                return_pct_oos REAL,
                accepted INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES optimizer_runs (run_id)
            )
            """
        )
        
        # Add return_pct columns if they don't exist (for existing DBs)
        try:
            cur.execute("ALTER TABLE optimizer_param_results ADD COLUMN return_pct_is REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cur.execute("ALTER TABLE optimizer_param_results ADD COLUMN return_pct_oos REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Best parameters per timeframe/universe
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_best_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT NOT NULL,
                symbols_hash TEXT NOT NULL,
                config_version TEXT,
                params_json TEXT NOT NULL,
                sharpe_oos REAL,
                updated_at INTEGER NOT NULL,
                UNIQUE (timeframe, symbols_hash)
            )
            """
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def hash_symbols(symbols: List[str]) -> str:
        """Create a stable hash for a symbol universe."""
        normalized = ",".join(sorted(symbols))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def create_run(
        self,
        run_id: str,
        timeframe: str,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        config_version: Optional[str],
    ):
        """Insert a new optimization run record."""
        now = int(time.time())
        symbols_hash = self.hash_symbols(symbols)
        symbols_json = json.dumps(symbols)

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO optimizer_runs
            (run_id, timeframe, symbols_hash, symbols_json,
             start_date, end_date, config_version, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timeframe,
                symbols_hash,
                symbols_json,
                start_date,
                end_date,
                config_version,
                "running",
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()

    def update_run_status(self, run_id: str, status: str):
        """Update the status of an optimization run (e.g. 'completed', 'failed')."""
        now = int(time.time())
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE optimizer_runs
            SET status = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (status, now, run_id),
        )
        conn.commit()
        conn.close()

    def add_param_result(
        self,
        run_id: str,
        param_index: int,
        params: Dict,
        sharpe_is: float,
        sharpe_oos: float,
        dd_is: float,
        dd_oos: float,
        trades_is: int,
        trades_oos: int,
        accepted: bool,
        return_pct_is: Optional[float] = None,
        return_pct_oos: Optional[float] = None,
    ):
        """Store metrics for a single parameter set within a run."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO optimizer_param_results
            (run_id, param_index, params_json, sharpe_is, sharpe_oos,
             dd_is, dd_oos, trades_is, trades_oos, return_pct_is, return_pct_oos,
             accepted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                param_index,
                json.dumps(params),
                sharpe_is,
                sharpe_oos,
                dd_is,
                dd_oos,
                trades_is,
                trades_oos,
                return_pct_is,
                return_pct_oos,
                1 if accepted else 0,
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()

    def upsert_best_parameters(
        self,
        timeframe: str,
        symbols: List[str],
        config_version: Optional[str],
        params: Dict,
        sharpe_oos: float,
    ):
        """Store or update the best-known parameters for a timeframe/universe."""
        symbols_hash = self.hash_symbols(symbols)
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO optimizer_best_parameters
            (timeframe, symbols_hash, config_version, params_json, sharpe_oos, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(timeframe, symbols_hash) DO UPDATE SET
                params_json = excluded.params_json,
                sharpe_oos = excluded.sharpe_oos,
                updated_at = excluded.updated_at
            """,
            (
                timeframe,
                symbols_hash,
                config_version,
                json.dumps(params),
                sharpe_oos,
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()

    def get_best_parameters(
        self, timeframe: str, symbols: List[str]
    ) -> Optional[Dict]:
        """Fetch best parameters for timeframe/universe if present."""
        symbols_hash = self.hash_symbols(symbols)
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT params_json, sharpe_oos
            FROM optimizer_best_parameters
            WHERE timeframe = ? AND symbols_hash = ?
            """,
            (timeframe, symbols_hash),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        params_json, sharpe_oos = row
        try:
            params = json.loads(params_json)
        except Exception:
            logger.warning("Failed to parse params_json from optimizer_best_parameters")
            return None
        return {"params": params, "sharpe_oos": sharpe_oos}

    def get_top_historical_parameters(
        self,
        timeframe: Optional[str] = None,
        min_oos_sharpe: float = 0.5,
        min_trades_oos: int = 10,
        top_n: int = 10,
        days_lookback: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get top-ranked historical parameter sets across all runs.
        
        This allows the optimizer to include proven performers as seeds
        in new optimization runs, ensuring continuity across time periods.
        
        Args:
            timeframe: Optional timeframe filter (None = all timeframes)
            min_oos_sharpe: Minimum out-of-sample Sharpe ratio to include
            min_trades_oos: Minimum out-of-sample trades to include
            top_n: Number of top performers to return
            days_lookback: Optional limit to recent runs only (None = all time)
        
        Returns:
            List of dicts with params, sharpe_oos, dd_oos, and composite_score
        """
        conn = self._connect()
        cur = conn.cursor()
        
        # Build query to aggregate parameter performance across all runs
        # We group by parameter set (params_json) and compute aggregate stats
        where_conditions = ["sharpe_oos >= ?", "trades_oos >= ?", "accepted = 1"]
        query_params = [min_oos_sharpe, min_trades_oos]
        
        if timeframe:
            # Join with runs table to filter by timeframe
            where_conditions.append("r.timeframe = ?")
            query_params.append(timeframe)
        
        if days_lookback:
            cutoff_timestamp = int(time.time()) - (days_lookback * 86400)
            where_conditions.append("r.created_at >= ?")
            query_params.append(cutoff_timestamp)
        
        where_clause = " AND " + " AND ".join(where_conditions)
        
        query = f"""
            SELECT 
                pr.params_json,
                AVG(pr.sharpe_oos) as avg_sharpe_oos,
                AVG(pr.sharpe_is) as avg_sharpe_is,
                AVG(pr.dd_oos) as avg_dd_oos,
                AVG(pr.dd_is) as avg_dd_is,
                MIN(pr.trades_oos) as min_trades_oos,
                COUNT(*) as run_count,
                MAX(r.created_at) as last_seen_at
            FROM optimizer_param_results pr
            JOIN optimizer_runs r ON pr.run_id = r.run_id
            WHERE {where_clause.replace('r.timeframe', 'r.timeframe').replace('r.created_at', 'r.created_at')}
            GROUP BY pr.params_json
            HAVING COUNT(*) >= 1
            ORDER BY avg_sharpe_oos DESC, min_trades_oos DESC
            LIMIT ?
        """
        
        # Fix the query to properly reference joined table
        # Build dynamic WHERE clause
        query_params_final = []
        where_parts = []
        
        if timeframe:
            where_parts.append("r.timeframe = ?")
            query_params_final.append(timeframe)
        
        if days_lookback:
            cutoff_timestamp = int(time.time()) - (days_lookback * 86400)
            where_parts.append("r.created_at >= ?")
            query_params_final.append(cutoff_timestamp)
        
        where_parts.extend([
            "pr.sharpe_oos >= ?",
            "pr.trades_oos >= ?",
            "pr.accepted = 1"
        ])
        query_params_final.extend([min_oos_sharpe, min_trades_oos])
        
        where_clause = " AND ".join(where_parts)
        query_params_final.append(top_n)
        
        query = f"""
            SELECT 
                pr.params_json,
                AVG(pr.sharpe_oos) as avg_sharpe_oos,
                AVG(pr.sharpe_is) as avg_sharpe_is,
                AVG(pr.dd_oos) as avg_dd_oos,
                AVG(pr.dd_is) as avg_dd_is,
                AVG(COALESCE(pr.return_pct_oos, 0)) as avg_return_pct_oos,
                AVG(COALESCE(pr.return_pct_is, 0)) as avg_return_pct_is,
                MIN(pr.trades_oos) as min_trades_oos,
                COUNT(*) as run_count,
                MAX(r.created_at) as last_seen_at
            FROM optimizer_param_results pr
            JOIN optimizer_runs r ON pr.run_id = r.run_id
            WHERE {where_clause}
            GROUP BY pr.params_json
            HAVING COUNT(*) >= 1
            ORDER BY avg_sharpe_oos DESC, min_trades_oos DESC
            LIMIT ?
        """
        
        cur.execute(query, query_params_final)
        
        results = []
        for row in cur.fetchall():
            params_json, avg_sharpe_oos, avg_sharpe_is, avg_dd_oos, avg_dd_is, avg_return_pct_oos, avg_return_pct_is, min_trades_oos, run_count, last_seen_at = row
            
            try:
                params = json.loads(params_json)
            except Exception:
                logger.warning(f"Failed to parse params_json: {params_json[:100]}")
                continue
            
            # Composite score: prioritize OOS Sharpe, penalize large drawdowns, reward robustness (run_count)
            # Also consider returns, but Sharpe is the primary metric
            consistency_bonus = max(0, (avg_sharpe_is - avg_sharpe_oos) * 0.5) if avg_sharpe_is > avg_sharpe_oos else 0
            robustness_bonus = min(run_count / 5.0, 1.0) * 0.5  # Reward robustness (capped at 1.0)
            return_bonus = max(0, avg_return_pct_oos / 100.0) * 0.3  # Reward positive returns (normalized)
            
            composite_score = (
                avg_sharpe_oos * 2.0  # OOS performance is most important
                - abs(avg_dd_oos) * 0.1  # Penalize large drawdowns (dd is negative)
                + consistency_bonus  # Reward consistency (IS vs OOS)
                + robustness_bonus  # Reward robustness across multiple runs
                + return_bonus  # Reward positive returns
            )
            
            results.append({
                "params": params,
                "avg_sharpe_oos": avg_sharpe_oos,
                "avg_sharpe_is": avg_sharpe_is,
                "avg_dd_oos": avg_dd_oos,
                "avg_dd_is": avg_dd_is,
                "avg_return_pct_oos": avg_return_pct_oos,
                "avg_return_pct_is": avg_return_pct_is,
                "min_trades_oos": min_trades_oos,
                "run_count": run_count,
                "last_seen_at": last_seen_at,
                "composite_score": composite_score,
            })
        
        conn.close()
        
        # Sort by composite score
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_n]


