"""Parameter optimization with walk-forward analysis."""

import json
import sqlite3
import random
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..config import BotConfig, TrendStrategyConfig, CrossSectionalStrategyConfig
from ..backtest.backtester import Backtester
from ..data.ohlcv_store import OHLCVStore
from ..logging_utils import get_logger

logger = get_logger(__name__)


class Optimizer:
    """Optimize strategy parameters using walk-forward analysis."""
    
    def __init__(self, config: BotConfig, ohlcv_store: OHLCVStore):
        """
        Initialize optimizer.
        
        Args:
            config: Bot configuration
            ohlcv_store: OHLCV data store
        """
        self.config = config
        self.store = ohlcv_store
        self.logger = get_logger(__name__)
        self.backtester = Backtester(config)
    
    def optimize(
        self,
        symbols: List[str],
        timeframe: str
    ) -> Dict:
        """
        Run parameter optimization.
        
        Args:
            symbols: List of symbols to optimize on
            timeframe: Timeframe (e.g., '1h')
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting optimization for {len(symbols)} symbols")
        
        # Load historical data
        lookback_months = self.config.optimizer.lookback_months
        lookback_days = lookback_months * 30
        
        since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        symbol_data = {}
        symbols_with_insufficient_data = []
        symbols_with_no_data = []
        
        for symbol in symbols:
            try:
                df = self.store.get_ohlcv(symbol, timeframe, since=since)
                if df.empty:
                    symbols_with_no_data.append(symbol)
                    self.logger.warning(f"No data found for {symbol} at timeframe {timeframe}")
                    # Check what timeframes are available
                    try:
                        import sqlite3
                        conn = sqlite3.connect(self.store.db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT DISTINCT timeframe FROM ohlcv WHERE symbol = ?",
                            (symbol,)
                        )
                        available_timeframes = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        if available_timeframes:
                            self.logger.info(
                                f"Available timeframes for {symbol}: {available_timeframes}. "
                                f"Optimizer requires {timeframe}"
                            )
                    except Exception:
                        pass
                elif len(df) < 200:  # Need at least 200 bars
                    symbols_with_insufficient_data.append((symbol, len(df)))
                    self.logger.warning(
                        f"Insufficient data for {symbol}: {len(df)} bars (need at least 200)"
                    )
                else:
                    symbol_data[symbol] = df
                    self.logger.info(f"Loaded {len(df)} candles for {symbol} at {timeframe}")
            except Exception as e:
                symbols_with_no_data.append(symbol)
                self.logger.warning(f"Error loading data for {symbol}: {e}")
                continue
        
        if not symbol_data:
            error_msg = f"No data available for optimization at timeframe {timeframe}.\n"
            if symbols_with_no_data:
                error_msg += f"Symbols with no data: {', '.join(symbols_with_no_data)}\n"
            if symbols_with_insufficient_data:
                insufficient_list = [f"{s} ({n} bars)" for s, n in symbols_with_insufficient_data]
                error_msg += f"Symbols with insufficient data (<200 bars): {', '.join(insufficient_list)}\n"
            error_msg += f"\nTo fix:\n"
            error_msg += f"1. Download data first: Let the live bot run or manually download data\n"
            error_msg += f"2. Check timeframe: Ensure data exists for {timeframe}\n"
            error_msg += f"3. Check symbols: Verify {', '.join(symbols)} exist in database\n"
            self.logger.error(error_msg)
            return {'error': error_msg}
        
        # Generate parameter sets
        if self.config.optimizer.search_method == 'random':
            param_sets = self._generate_random_params(
                self.config.optimizer.n_trials,
                self.config.optimizer.param_ranges
            )
        elif self.config.optimizer.search_method == 'grid':
            param_sets = self._generate_grid_params(
                self.config.optimizer.param_ranges
            )
        else:
            # Default to random
            param_sets = self._generate_random_params(
                self.config.optimizer.n_trials,
                self.config.optimizer.param_ranges
            )
        
        self.logger.info(f"Testing {len(param_sets)} parameter sets")
        
        # Walk-forward analysis
        walk_forward_window_days = self.config.optimizer.walk_forward_window_days
        results = []
        
        for i, params in enumerate(param_sets):
            try:
                # Create config with these parameters
                test_config = self._create_test_config(params)

                # Run walk-forward backtests
                walk_forward_results = self._walk_forward_backtest(
                    symbol_data,
                    test_config,
                    walk_forward_window_days,
                )

                if walk_forward_results:
                    # Split into in-sample and out-of-sample windows (e.g. 70% / 30%)
                    n_windows = len(walk_forward_results)
                    split_idx = max(1, int(n_windows * 0.7))
                    is_results = walk_forward_results[:split_idx]
                    oos_results = walk_forward_results[split_idx:] or walk_forward_results

                    def _agg(results_list):
                        return (
                            np.mean([r["total_return_pct"] for r in results_list]),
                            np.mean([r["sharpe_ratio"] for r in results_list]),
                            np.mean([r["max_drawdown_pct"] for r in results_list]),
                            np.mean([r["total_trades"] for r in results_list]),
                            min([r["total_trades"] for r in results_list]),
                        )

                    avg_return_is, avg_sharpe_is, avg_dd_is, avg_trades_is, min_trades_is = _agg(
                        is_results
                    )
                    avg_return_oos, avg_sharpe_oos, avg_dd_oos, avg_trades_oos, min_trades_oos = _agg(
                        oos_results
                    )

                    # Combined stats (for backwards compatibility)
                    avg_return = np.mean(
                        [r["total_return_pct"] for r in walk_forward_results]
                    )
                    avg_sharpe = np.mean(
                        [r["sharpe_ratio"] for r in walk_forward_results]
                    )
                    avg_dd = np.mean(
                        [r["max_drawdown_pct"] for r in walk_forward_results]
                    )
                    avg_trades = np.mean(
                        [r["total_trades"] for r in walk_forward_results]
                    )
                    min_trades = min(
                        [r["total_trades"] for r in walk_forward_results]
                    )

                    # Check if meets criteria (both IS and OOS)
                    passes_is = (
                        min_trades_is >= self.config.optimizer.min_trades
                        and avg_sharpe_is >= self.config.optimizer.min_sharpe_ratio
                        and avg_dd_is >= self.config.optimizer.max_drawdown_pct
                    )
                    # Allow slightly looser constraints for OOS
                    passes_oos = (
                        min_trades_oos >= self.config.optimizer.min_trades
                        and avg_sharpe_oos
                        >= self.config.optimizer.min_sharpe_ratio * 0.7
                        and avg_dd_oos
                        >= self.config.optimizer.max_drawdown_pct * 1.2
                    )

                    if passes_is and passes_oos:
                        results.append(
                            {
                                "params": params,
                                "avg_return_pct": avg_return,
                                "avg_sharpe": avg_sharpe,
                                "avg_drawdown_pct": avg_dd,
                                "avg_trades": avg_trades,
                                "min_trades": min_trades,
                                "walk_forward_results": walk_forward_results,
                                "avg_sharpe_is": avg_sharpe_is,
                                "avg_sharpe_oos": avg_sharpe_oos,
                                "avg_dd_is": avg_dd_is,
                                "avg_dd_oos": avg_dd_oos,
                            }
                        )

                    if (i + 1) % 10 == 0:
                        self.logger.info(
                            f"Tested {i + 1}/{len(param_sets)} parameter sets"
                        )

            except Exception as e:
                self.logger.warning(f"Error testing parameter set {i}: {e}")
                continue
        
        if not results:
            self.logger.warning("No parameter sets met optimization criteria")
            return {'error': 'No valid parameter sets found'}
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        best_result = results[0]
        self.logger.info(
            f"Best parameters: Sharpe={best_result['avg_sharpe']:.2f}, "
            f"Return={best_result['avg_return_pct']:.2f}%, "
            f"DD={best_result['avg_drawdown_pct']:.2f}%"
        )
        
        return {
            'best_params': best_result['params'],
            'best_metrics': {
                'avg_return_pct': best_result['avg_return_pct'],
                'avg_sharpe': best_result['avg_sharpe'],
                'avg_drawdown_pct': best_result['avg_drawdown_pct'],
                'avg_trades': best_result['avg_trades']
            },
            'all_results': results[:10]  # Top 10
        }
    
    def _generate_random_params(self, n_trials: int, param_ranges: Dict) -> List[Dict]:
        """Generate random parameter sets."""
        param_sets = []
        
        for _ in range(n_trials):
            params = {}
            for param_name, values in param_ranges.items():
                params[param_name] = random.choice(values)
            param_sets.append(params)
        
        return param_sets
    
    def _generate_grid_params(self, param_ranges: Dict) -> List[Dict]:
        """Generate grid search parameter sets."""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        param_sets = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            param_sets.append(params)
        
        return param_sets
    
    def _create_test_config(self, params: Dict) -> BotConfig:
        """Create a test config with given parameters."""
        import copy
        
        test_config = copy.deepcopy(self.config)
        
        # Update trend parameters
        if 'ma_short' in params:
            test_config.strategy.trend.ma_short = params['ma_short']
        if 'ma_long' in params:
            test_config.strategy.trend.ma_long = params['ma_long']
        if 'momentum_lookback' in params:
            test_config.strategy.trend.momentum_lookback = params['momentum_lookback']
        if 'atr_stop_multiplier' in params:
            test_config.strategy.trend.atr_stop_multiplier = params['atr_stop_multiplier']
        
        # Update cross-sectional parameters
        if 'top_k' in params:
            test_config.strategy.cross_sectional.top_k = params['top_k']
        
        return test_config
    
    def _walk_forward_backtest(
        self,
        symbol_data: Dict,
        test_config: BotConfig,
        window_days: int
    ) -> List[Dict]:
        """Run walk-forward backtests."""
        from ..backtest.backtester import Backtester
        
        backtester = Backtester(test_config)
        results = []
        
        # Get date range
        all_dates = []
        for df in symbol_data.values():
            if not df.empty:
                all_dates.extend(df.index.tolist())
        
        if not all_dates:
            return []
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Create windows
        current_date = start_date
        windows = []
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=window_days)
            if window_end > end_date:
                window_end = end_date
            
            windows.append((current_date, window_end))
            current_date = window_end
        
        # Backtest each window
        for window_start, window_end in windows:
            # Filter data to window
            window_data = {}
            for symbol, df in symbol_data.items():
                mask = (df.index >= window_start) & (df.index <= window_end)
                window_df = df[mask].copy()
                if not window_df.empty and len(window_df) > 100:
                    window_data[symbol] = window_df
            
            if not window_data:
                continue
            
            try:
                result = backtester.backtest(window_data)
                if 'error' not in result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in walk-forward window {window_start} to {window_end}: {e}")
                continue
        
        return results
    
    def compare_with_current(self, best_params: Dict) -> Dict:
        """
        Compare best parameters with current config.
        
        Returns:
            Dictionary with comparison and recommendation
        """
        current_params = {
            'ma_short': self.config.strategy.trend.ma_short,
            'ma_long': self.config.strategy.trend.ma_long,
            'momentum_lookback': self.config.strategy.trend.momentum_lookback,
            'atr_stop_multiplier': self.config.strategy.trend.atr_stop_multiplier,
            'top_k': self.config.strategy.cross_sectional.top_k
        }
        
        # Check if best params differ from current
        params_changed = {}
        for key, value in best_params.items():
            if key in current_params and current_params[key] != value:
                params_changed[key] = {
                    'old': current_params[key],
                    'new': value
                }
        
        if not params_changed:
            return {
                'should_update': False,
                'reason': 'Best parameters match current config'
            }
        
        return {
            'should_update': True,
            'params_changed': params_changed,
            'recommendation': 'Update config with new parameters'
        }
    
    def save_optimization_result(self, result: Dict, db_path: str):
        """Save optimization result to database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    best_params TEXT NOT NULL,
                    best_metrics TEXT NOT NULL,
                    should_update BOOLEAN,
                    params_changed TEXT
                )
            """)
            
            # Insert result
            timestamp = int(datetime.now().timestamp())
            best_params_json = json.dumps(result.get('best_params', {}))
            best_metrics_json = json.dumps(result.get('best_metrics', {}))
            params_changed_json = json.dumps(result.get('comparison', {}).get('params_changed', {}))
            
            cursor.execute("""
                INSERT INTO optimization_results
                (timestamp, best_params, best_metrics, should_update, params_changed)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                best_params_json,
                best_metrics_json,
                result.get('comparison', {}).get('should_update', False),
                params_changed_json
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving optimization result: {e}")
            conn.rollback()
            conn.close()


# Import numpy here to avoid circular import
import numpy as np

