"""Parameter optimization with walk-forward analysis."""

import json
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta, date
from pathlib import Path
import uuid
import sqlite3
import numpy as np

from ..config import BotConfig, TrendStrategyConfig, CrossSectionalStrategyConfig
from ..backtest.backtester import Backtester
from ..data.ohlcv_store import OHLCVStore
from .store import OptimizerStore
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
        # OptimizerStore shares the same SQLite DB as OHLCV/universe
        self.optimizer_store = OptimizerStore(ohlcv_store.db_path)
    
    def optimize(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        universe_history: Optional[Dict[date, Set[str]]] = None,
    ) -> Dict:
        """
        Run parameter optimization.
        
        Args:
            symbols: List of symbols to optimize on
            timeframe: Timeframe (e.g., '1h')
            start_date: Optional start date for data (if None, uses lookback_months from now)
            end_date: Optional end date for data (if None, uses now)
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting optimization for {len(symbols)} symbols at timeframe {timeframe}")
        
        # Store the optimization timeframe to ensure all test configs use it
        self.optimization_timeframe = timeframe

        # Create a persistent run record for this optimization
        run_id = uuid.uuid4().hex
        start_date_str = start_date.isoformat() if start_date else None
        end_date_str = end_date.isoformat() if end_date else None
        try:
            self.optimizer_store.create_run(
                run_id=run_id,
                timeframe=timeframe,
                symbols=symbols,
                start_date=start_date_str,
                end_date=end_date_str,
                config_version=getattr(self.config, "config_version", None),
            )
        except Exception as e:
            # Non-fatal: optimization can still proceed without DB tracking
            self.logger.warning(f"Could not create optimizer run record: {e}")
            run_id = None
        
        # Load historical data
        if start_date is not None:
            since = int(start_date.timestamp() * 1000)
        else:
            lookback_months = self.config.optimizer.lookback_months
            lookback_days = lookback_months * 30
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        if end_date is not None:
            until = int(end_date.timestamp() * 1000)
        else:
            until = None
        
        symbol_data = {}
        symbols_with_insufficient_data = []
        symbols_with_no_data = []
        
        for symbol in symbols:
            try:
                # First, try to get data without since filter to check what exists
                # This helps with debugging when data exists but not in the requested range
                df_all = self.store.get_ohlcv(symbol, timeframe)
                
                # Now get data with the requested date range
                df = self.store.get_ohlcv(symbol, timeframe, since=since)
                
                # Filter by end_date if provided
                if end_date is not None and not df.empty:
                    # df.index is DatetimeIndex, filter directly by datetime
                    df = df[df.index <= end_date]
                
                # If no data in requested range but data exists, log helpful info
                if df.empty and not df_all.empty:
                    earliest_in_db = df_all.index[0]
                    latest_in_db = df_all.index[-1]
                    requested_start = datetime.fromtimestamp(since / 1000) if since else None
                    self.logger.warning(
                        f"No data for {symbol} {timeframe} in requested range. "
                        f"Data in DB: {earliest_in_db.date()} to {latest_in_db.date()}. "
                        f"Requested: {requested_start.date() if requested_start else 'any'} to {end_date.date() if end_date else 'now'}"
                    )
                
                if df.empty:
                    symbols_with_no_data.append(symbol)
                    
                    # Check what data exists for this symbol/timeframe
                    try:
                        df_all = self.store.get_ohlcv(symbol, timeframe)
                        if not df_all.empty:
                            earliest = df_all.index[0]
                            latest = df_all.index[-1]
                            self.logger.warning(
                                f"No data found for {symbol} at timeframe {timeframe} in requested range. "
                                f"Data exists in DB: {earliest.date()} to {latest.date()} ({len(df_all)} bars). "
                                f"Requested: {datetime.fromtimestamp(since / 1000).date() if since else 'any'} to {end_date.date() if end_date else 'now'}"
                            )
                        else:
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
                    except Exception as e:
                        self.logger.warning(f"No data found for {symbol} at timeframe {timeframe}: {e}")
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
        rng = random
        seed = getattr(self.config.optimizer, "random_seed", None)
        if seed is not None:
            rng = random.Random(seed)

        param_ranges = self.config.optimizer.param_ranges
        sample_method = getattr(self.config.optimizer, "sample_method", "uniform")
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= max(1, len(values))

        if (
            self.config.optimizer.search_method == 'random'
            and total_combinations > 0
            and getattr(self.config.optimizer, "coverage_warning_threshold", 1e-4) > 0
        ):
            coverage = self.config.optimizer.n_trials / total_combinations
            if coverage < self.config.optimizer.coverage_warning_threshold:
                self.logger.warning(
                    "Optimizer sampling coverage is very low (%.6f%% of parameter space). "
                    "Consider increasing n_trials or narrowing param_ranges.",
                    coverage * 100,
                )

        if self.config.optimizer.search_method == 'random':
            param_sets = self._generate_random_params(
                self.config.optimizer.n_trials,
                param_ranges,
                rng=rng,
                method=sample_method,
            )
        elif self.config.optimizer.search_method == 'grid':
            param_sets = self._generate_grid_params(
                param_ranges
            )
        else:
            # Default to random
            param_sets = self._generate_random_params(
                self.config.optimizer.n_trials,
                param_ranges,
                rng=rng,
                method=sample_method,
            )
        
        # Always include current config parameters as the first test (baseline)
        current_params = self._get_current_params()
        
        # Only add if not already in the list (avoids duplicate testing)
        if current_params not in param_sets:
            param_sets.insert(0, current_params)
            self.logger.info(f"Added current config parameters as baseline (test #0)")
        
        # Include top historical performers as seeds (hall of fame)
        # This ensures proven parameter sets continue to be tested over time
        try:
            top_historical = self.optimizer_store.get_top_historical_parameters(
                timeframe=timeframe,
                min_oos_sharpe=0.5,  # Only include decent performers
                min_trades_oos=10,  # Need sufficient trades for reliability
                top_n=5,  # Include top 5 historical performers
                days_lookback=180,  # Last 6 months
            )
            
            seeds_added = 0
            for hist_param in top_historical:
                hist_params = hist_param["params"]
                # Only add if not already in the list and not the current config
                if hist_params not in param_sets and hist_params != current_params:
                    param_sets.append(hist_params)
                    seeds_added += 1
                    self.logger.info(
                        f"Added top historical performer (Sharpe OOS: {hist_param['avg_sharpe_oos']:.2f}, "
                        f"seen in {hist_param['run_count']} runs): {hist_params}"
                    )
            
            if seeds_added > 0:
                self.logger.info(f"Added {seeds_added} top historical parameter sets as seeds")
        except Exception as e:
            # Non-fatal: optimization can proceed without historical seeds
            self.logger.warning(f"Could not load historical top performers: {e}")
        
        self.logger.info(f"Testing {len(param_sets)} parameter sets (current config baseline + historical seeds + new candidates)")
        
        # Calculate total data available and adjust criteria if limited
        total_bars = sum(len(df) for df in symbol_data.values())
        if symbol_data:
            avg_bars_per_symbol = total_bars / len(symbol_data)
            if 'h' in timeframe:
                hours_per_bar = int(timeframe.replace('h', ''))
            elif 'd' in timeframe:
                hours_per_bar = 24 * int(timeframe.replace('d', ''))
            else:
                hours_per_bar = 1
            total_days = (avg_bars_per_symbol * hours_per_bar) / 24.0
        else:
            total_days = 0
        
        # Relax criteria if data is limited (less than 6 months)
        min_trades_threshold = self.config.optimizer.min_trades
        min_sharpe_threshold = self.config.optimizer.min_sharpe_ratio
        max_dd_threshold = self.config.optimizer.max_drawdown_pct
        
        if total_days < 180:  # Less than 6 months
            # Relax criteria proportionally
            relaxation_factor = max(0.5, total_days / 180.0)  # Between 50% and 100%
            min_trades_threshold = max(5, int(self.config.optimizer.min_trades * relaxation_factor))
            min_sharpe_threshold = self.config.optimizer.min_sharpe_ratio * relaxation_factor
            max_dd_threshold = self.config.optimizer.max_drawdown_pct * (1.0 / relaxation_factor)  # More lenient on drawdown
            
            self.logger.info(
                f"Limited data detected (~{total_days:.0f} days). Relaxing criteria: "
                f"min_trades={min_trades_threshold}, min_sharpe={min_sharpe_threshold:.2f}, "
                f"max_dd={max_dd_threshold:.1f}%"
            )
        
        # Walk-forward analysis
        walk_forward_window_days = self.config.optimizer.walk_forward_window_days
        results = []
        
        import time as time_module
        start_time = time_module.time()
        total_days = (max(df.index[-1] for df in symbol_data.values()) - min(df.index[0] for df in symbol_data.values())).days if symbol_data else 0
        estimated_windows = int(total_days / walk_forward_window_days) if walk_forward_window_days > 0 else 0
        
        self.logger.info(
            f"Walk-forward optimization: {len(param_sets)} parameter sets, "
            f"~{estimated_windows} windows per set ({walk_forward_window_days}-day windows over ~{total_days} days), "
            f"estimated time: ~{estimated_windows * len(param_sets) * 2 / 60:.1f} minutes (assuming ~2s per window)"
        )
        
        best_oos_sharpe = None
        best_oos_params = None
        current_config_metrics = None  # Track performance of current config

        collected_all_results = []

        for i, params in enumerate(param_sets):
            try:
                param_start_time = time_module.time()
                
                # Create config with these parameters
                test_config = self._create_test_config(params, timeframe)

                # Run walk-forward backtests
                self.logger.debug(f"Parameter set {i+1}/{len(param_sets)}: Starting walk-forward backtest...")
                walk_forward_results = self._walk_forward_backtest(
                    symbol_data,
                    test_config,
                    walk_forward_window_days,
                    universe_history=universe_history,
                )
                param_elapsed = time_module.time() - param_start_time
                
                # Log progress for all sets (but more detail for first few and every 10th)
                if i < 3 or (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Parameter set {i+1}/{len(param_sets)} completed in {param_elapsed:.1f}s "
                        f"({len(walk_forward_results)} windows, avg {param_elapsed/len(walk_forward_results):.2f}s/window) "
                        f"| ETA: ~{((time_module.time() - start_time) / (i + 1) * (len(param_sets) - i - 1)) / 60:.1f} min remaining"
                    )

                if walk_forward_results:
                    # Split into in-sample and out-of-sample windows (e.g. 70% / 30%)
                    n_windows = len(walk_forward_results)
                    split_idx = max(1, int(n_windows * 0.7))
                    is_results = walk_forward_results[:split_idx]
                    oos_results = walk_forward_results[split_idx:] or walk_forward_results
                    oos_eval = oos_results
                    if len(oos_results) > 1:
                        oos_eval = oos_results[1:]
                    if not oos_eval:
                        oos_eval = oos_results

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
                        oos_eval
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
                    # Note: drawdowns are negative, so >= means "better" (less negative)
                    passes_is = (
                        min_trades_is >= min_trades_threshold
                        and avg_sharpe_is >= min_sharpe_threshold
                        and avg_dd_is >= max_dd_threshold
                    )
                    # Allow slightly looser constraints for OOS
                    passes_oos = (
                        min_trades_oos >= min_trades_threshold
                        and avg_sharpe_oos >= min_sharpe_threshold * 0.7
                        and avg_dd_oos >= max_dd_threshold * 1.2
                    )
                    
                    # Log failed criteria (first failed set and every 10th after)
                    if (not passes_is or not passes_oos) and (i == 0 or (i + 1) % 10 == 0):
                        self.logger.info(
                            f"Parameter set {i+1} failed criteria: "
                            f"IS: trades={min_trades_is:.0f}/{min_trades_threshold:.0f}, "
                            f"sharpe={avg_sharpe_is:.2f}/{min_sharpe_threshold:.2f}, "
                            f"dd={avg_dd_is:.2f}%/{max_dd_threshold:.1f}%, "
                            f"OOS: trades={min_trades_oos:.0f}/{min_trades_threshold:.0f}, "
                            f"sharpe={avg_sharpe_oos:.2f}/{min_sharpe_threshold * 0.7:.2f}, "
                            f"dd={avg_dd_oos:.2f}%/{max_dd_threshold * 1.2:.1f}%"
                        )

                    # Store all results with sufficient trades for ranking
                    # We'll rank them and pick the best even if none meet strict criteria
                    positive_windows = sum(1 for r in walk_forward_results if r.get("total_return_pct", 0) > 0)
                    window_success_ratio = positive_windows / len(walk_forward_results) if walk_forward_results else 0.0

                    all_results = {
                        "params": params,
                        "avg_return_pct": avg_return,
                        "avg_sharpe": avg_sharpe,
                        "avg_drawdown_pct": avg_dd,
                        "avg_trades": avg_trades,
                        "min_trades": min_trades,
                        "min_trades_oos": min_trades_oos,
                        "walk_forward_results": walk_forward_results,
                        "avg_sharpe_is": avg_sharpe_is,
                        "avg_sharpe_oos": avg_sharpe_oos,
                        "avg_dd_is": avg_dd_is,
                        "avg_dd_oos": avg_dd_oos,
                        "passes_is": passes_is,
                        "passes_oos": passes_oos,
                        "passes_all": passes_is and passes_oos,
                        "positive_window_ratio": window_success_ratio,
                    }
                    
                    # Store results based on priority:
                    # 1. Passes all criteria (preferred)
                    # 2. Passes OOS criteria (good for generalization)
                    # 3. Has sufficient trades for fallback ranking (with limited data)
                    already_added = False
                    
                    if passes_is and passes_oos:
                        results.append(all_results)
                        already_added = True
                        if (i == 0 or (i + 1) % 10 == 0):
                            self.logger.info(
                                f"Parameter set {i+1} ✓ PASSED all criteria: "
                                f"IS sharpe={avg_sharpe_is:.2f}, OOS sharpe={avg_sharpe_oos:.2f}"
                            )
                    elif total_days < 180 and passes_oos and min_trades_oos >= 5:
                        results.append(all_results)
                        already_added = True
                        if (i == 0 or (i + 1) % 10 == 0):
                            self.logger.info(
                                f"Parameter set {i+1} ✓ ACCEPTED based on OOS performance "
                                f"(OOS: sharpe={avg_sharpe_oos:.2f}, trades={min_trades_oos:.0f}) "
                                f"even though IS failed (IS: sharpe={avg_sharpe_is:.2f}, limited data: ~{total_days:.0f} days)"
                            )
                    
                    # Always collect for fallback ranking and diagnostics
                    collected_all_results.append(all_results)

                    # For limited data, also store ALL results with sufficient trades for fallback ranking
                    # This ensures we always have something to return, even if nothing meets strict criteria
                    if not already_added and total_days < 180 and min_trades_oos >= 5:
                        results.append(all_results)
                        if (i == 0 or (i + 1) % 10 == 0):
                            self.logger.info(
                                f"Parameter set {i+1} ✓ stored for fallback ranking "
                                f"(OOS: sharpe={avg_sharpe_oos:.2f}, trades={min_trades_oos:.0f}, "
                                f"limited data: {total_days:.0f} days)"
                            )

                    # Persist per-parameter-set metrics to DB (best-effort)
                    try:
                        if run_id is not None:
                            # Calculate return percentages for IS and OOS
                            return_pct_is = np.mean([r["total_return_pct"] for r in is_results]) if is_results else None
                            return_pct_oos = np.mean([r["total_return_pct"] for r in oos_eval]) if oos_eval else None
                            
                            self.optimizer_store.add_param_result(
                                run_id=run_id,
                                param_index=i + 1,
                                params=params,
                                sharpe_is=avg_sharpe_is,
                                sharpe_oos=avg_sharpe_oos,
                                dd_is=avg_dd_is,
                                dd_oos=avg_dd_oos,
                                trades_is=int(min_trades_is),
                                trades_oos=int(min_trades_oos),
                                accepted=passes_is and passes_oos,
                                return_pct_is=return_pct_is,
                                return_pct_oos=return_pct_oos,
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to persist param result for set {i+1}: {e}")

                    # Track best OOS Sharpe for best-parameters registry
                    if passes_oos and (best_oos_sharpe is None or avg_sharpe_oos > best_oos_sharpe):
                        best_oos_sharpe = avg_sharpe_oos
                        best_oos_params = params
                    
                    # If this is the current config (test #0), store its metrics for comparison
                    if i == 0 and params == self._get_current_params():
                        current_config_metrics = {
                            'avg_return_pct': avg_return,
                            'avg_sharpe': avg_sharpe,
                            'avg_drawdown_pct': avg_dd,
                            'avg_trades': avg_trades,
                            'avg_sharpe_oos': avg_sharpe_oos,
                            'avg_sharpe_is': avg_sharpe_is
                        }
                        self.logger.info(
                            f"Current config baseline performance: Sharpe={avg_sharpe:.2f} "
                            f"(IS={avg_sharpe_is:.2f}, OOS={avg_sharpe_oos:.2f}), "
                            f"Return={avg_return:+.2f}%, DD={avg_dd:.2f}%"
                        )
            except Exception as e:
                self.logger.warning(f"Error testing parameter set {i}: {e}")
                continue

        # Mark run status and update best-parameter registry
        if run_id is not None:
            try:
                self.optimizer_store.update_run_status(run_id, "completed")
            except Exception as e:
                self.logger.warning(f"Failed to update optimizer run status: {e}")

            if best_oos_sharpe is not None and best_oos_params is not None:
                try:
                    self.optimizer_store.upsert_best_parameters(
                        timeframe=timeframe,
                        symbols=list(symbol_data.keys()),
                        config_version=getattr(self.config, "config_version", None),
                        params=best_oos_params,
                        sharpe_oos=best_oos_sharpe,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to upsert best parameters: {e}")
        
        # Sort results - prioritize those that passed all criteria, then by OOS Sharpe
        if results:
            results.sort(
                key=lambda x: (
                    0 if x.get('passes_all', False) else 1,
                    x.get('avg_sharpe_oos', x.get('avg_sharpe', -999)),
                    x.get('avg_dd_oos', x.get('avg_drawdown_pct', -999)),
                ),
                reverse=True,
            )
            
            best_result = results[0]
            
            # Log what we're using
            if best_result.get('passes_all', False):
                self.logger.info(
                    f"Best parameters (passed all criteria): Sharpe={best_result['avg_sharpe']:.2f}, "
                    f"OOS Sharpe={best_result['avg_sharpe_oos']:.2f}, "
                    f"Return={best_result['avg_return_pct']:.2f}%, "
                    f"DD={best_result['avg_drawdown_pct']:.2f}%"
                )
            else:
                self.logger.warning(
                    f"Best parameters (relaxed criteria due to limited data ~{total_days:.0f} days): "
                    f"OOS Sharpe={best_result['avg_sharpe_oos']:.2f}, "
                    f"IS Sharpe={best_result['avg_sharpe_is']:.2f}, "
                    f"Return={best_result['avg_return_pct']:.2f}%, "
                    f"DD={best_result['avg_drawdown_pct']:.2f}%. "
                    f"Consider using more historical data for better optimization."
                )
            
            return {
                'best_params': best_result['params'],
                'best_metrics': {
                    'avg_return_pct': best_result['avg_return_pct'],
                    'avg_sharpe': best_result['avg_sharpe'],
                    'avg_drawdown_pct': best_result['avg_drawdown_pct'],
                    'avg_trades': best_result['avg_trades'],
                    'avg_sharpe_oos': best_result.get('avg_sharpe_oos', best_result['avg_sharpe']),
                    'avg_sharpe_is': best_result.get('avg_sharpe_is', best_result['avg_sharpe'])
                },
                'current_config_metrics': current_config_metrics,  # Include for comparison
                'all_results': results[:10],  # Top 10
                'warning': None if best_result.get('passes_all', False) else 'Used best available parameters - strict criteria not met'
            }
        
        # If no parameter set passed strict criteria, fall back to best available
        if not results and collected_all_results:
            collected_all_results.sort(
                key=lambda x: (
                    x.get('avg_sharpe_oos', x.get('avg_sharpe', -999)),
                    x.get('avg_dd_oos', x.get('avg_drawdown_pct', -999)),
                ),
                reverse=True,
            )
            fallback = collected_all_results[0]
            self.logger.warning(
                "No parameter sets met strict criteria; using best available: "
                "OOS Sharpe=%.2f, trades=%d. Consider expanding data or relaxing thresholds.",
                fallback.get('avg_sharpe_oos', fallback.get('avg_sharpe', 0)),
                fallback.get('min_trades', 0)
            )
            return {
                'best_params': fallback['params'],
                'best_metrics': {
                    'avg_return_pct': fallback['avg_return_pct'],
                    'avg_sharpe': fallback['avg_sharpe'],
                    'avg_drawdown_pct': fallback['avg_drawdown_pct'],
                    'avg_trades': fallback['avg_trades']
                },
                'warning': 'Used best available parameters - strict criteria not met'
            }
        
        self.logger.warning(
            f"No parameter sets met optimization criteria after testing {len(param_sets)} sets. "
            f"Criteria: min_trades={min_trades_threshold}, "
            f"min_sharpe={min_sharpe_threshold:.2f}, "
            f"max_dd={max_dd_threshold:.1f}%. "
            f"Available data: ~{total_days:.0f} days. "
            f"Try relaxing criteria or using more historical data."
        )
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
    
    def _generate_random_params(
        self,
        n_trials: int,
        param_ranges: Dict,
        rng=random,
        method: str = "uniform",
    ) -> List[Dict]:
        """Generate random parameter sets."""
        param_sets = []
        
        if method == "latin" and param_ranges:
            param_names = list(param_ranges.keys())
            buckets: Dict[str, List[Any]] = {}
            for name, values in param_ranges.items():
                if not values:
                    buckets[name] = [None] * n_trials
                    continue
                repeated = []
                while len(repeated) < n_trials:
                    repeated.extend(values)
                repeated = repeated[:n_trials]
                rng.shuffle(repeated)
                buckets[name] = repeated
            for i in range(n_trials):
                params = {name: buckets[name][i] for name in param_names}
                param_sets.append(params)
        else:
            for _ in range(n_trials):
                params = {}
                for param_name, values in param_ranges.items():
                    params[param_name] = rng.choice(values)
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
    
    def _create_test_config(self, params: Dict, timeframe: str) -> BotConfig:
        """
        Create a test config with given parameters.
        
        Args:
            params: Parameter dictionary to test
            timeframe: Timeframe to use for optimization (from config.yaml)
        
        Returns:
            BotConfig with updated parameters and timeframe
        """
        import copy
        
        test_config = copy.deepcopy(self.config)
        
        # Explicitly set the timeframe from config.yaml
        test_config.exchange.timeframe = timeframe
        
        # Update trend parameters
        if 'ma_short' in params:
            test_config.strategy.trend.ma_short = params['ma_short']
        if 'ma_long' in params:
            test_config.strategy.trend.ma_long = params['ma_long']
        if 'momentum_lookback' in params:
            test_config.strategy.trend.momentum_lookback = params['momentum_lookback']
        if 'atr_stop_multiplier' in params:
            test_config.strategy.trend.atr_stop_multiplier = params['atr_stop_multiplier']
        if 'atr_period' in params:
            test_config.strategy.trend.atr_period = params['atr_period']
        
        # Update cross-sectional parameters
        if 'top_k' in params:
            test_config.strategy.cross_sectional.top_k = params['top_k']
        if 'ranking_window' in params:
            test_config.strategy.cross_sectional.ranking_window = params['ranking_window']
        if 'exit_band' in params:
            test_config.strategy.cross_sectional.exit_band = params['exit_band']
        if 'rebalance_frequency_hours' in params:
            test_config.strategy.cross_sectional.rebalance_frequency_hours = params['rebalance_frequency_hours']
        
        return test_config
    
    def _walk_forward_backtest(
        self,
        symbol_data: Dict,
        test_config: BotConfig,
        window_days: int,
        universe_history: Optional[Dict[date, Set[str]]] = None,
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
            
            window_universe = None
            if universe_history:
                window_universe = {
                    day: members
                    for day, members in universe_history.items()
                    if window_start.date() <= day <= window_end.date()
                }
            
            try:
                result = backtester.backtest(
                    window_data,
                    taker_fee=test_config.exchange.taker_fee,
                    funding_rate_per_8h=getattr(test_config.exchange, "funding_rate_per_8h", 0.0),
                    universe_history=window_universe,
                    stop_slippage_bps=test_config.risk.stop_slippage_bps,
                    tp_slippage_bps=test_config.risk.tp_slippage_bps,
                )
                if 'error' not in result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in walk-forward window {window_start} to {window_end}: {e}")
                continue
        
        return results
    
    def _get_current_params(self) -> Dict:
        """Extract current config parameters."""
        params = {
            'ma_short': self.config.strategy.trend.ma_short,
            'ma_long': self.config.strategy.trend.ma_long,
            'momentum_lookback': self.config.strategy.trend.momentum_lookback,
            'atr_stop_multiplier': self.config.strategy.trend.atr_stop_multiplier,
            'top_k': self.config.strategy.cross_sectional.top_k,
            'ranking_window': self.config.strategy.cross_sectional.ranking_window
        }
        # Add optional parameters if they exist in config
        if hasattr(self.config.strategy.trend, 'atr_period'):
            params['atr_period'] = self.config.strategy.trend.atr_period
        if hasattr(self.config.strategy.cross_sectional, 'exit_band'):
            params['exit_band'] = self.config.strategy.cross_sectional.exit_band
        if hasattr(self.config.strategy.cross_sectional, 'rebalance_frequency_hours'):
            params['rebalance_frequency_hours'] = self.config.strategy.cross_sectional.rebalance_frequency_hours
        return params
    
    def compare_with_current(
        self, 
        best_params: Dict, 
        best_metrics: Dict = None,
        current_metrics: Dict = None
    ) -> Dict:
        """
        Compare best parameters with current config, including performance metrics.
        
        Args:
            best_params: Best parameters found by optimizer
            best_metrics: Performance metrics for best parameters (optional)
            current_metrics: Performance metrics for current config (optional)
        
        Returns:
            Dictionary with comparison and recommendation
        """
        current_params = self._get_current_params()
        
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
                'reason': 'Best parameters match current config',
                'params_changed': {},
                'performance_comparison': None
            }
        
        # Build recommendation with performance comparison if available
        recommendation = 'Update config with new parameters'
        performance_comparison = None
        
        if current_metrics and best_metrics:
            # Calculate performance improvements
            sharpe_improvement = best_metrics.get('avg_sharpe', 0) - current_metrics.get('avg_sharpe', 0)
            return_improvement = best_metrics.get('avg_return_pct', 0) - current_metrics.get('avg_return_pct', 0)
            dd_improvement = best_metrics.get('avg_drawdown_pct', 0) - current_metrics.get('avg_drawdown_pct', 0)
            
            performance_comparison = {
                'current': {
                    'sharpe': current_metrics.get('avg_sharpe', 0),
                    'return_pct': current_metrics.get('avg_return_pct', 0),
                    'drawdown_pct': current_metrics.get('avg_drawdown_pct', 0),
                    'trades': current_metrics.get('avg_trades', 0)
                },
                'best': {
                    'sharpe': best_metrics.get('avg_sharpe', 0),
                    'return_pct': best_metrics.get('avg_return_pct', 0),
                    'drawdown_pct': best_metrics.get('avg_drawdown_pct', 0),
                    'trades': best_metrics.get('avg_trades', 0)
                },
                'improvements': {
                    'sharpe': sharpe_improvement,
                    'return_pct': return_improvement,
                    'drawdown_pct': dd_improvement
                }
            }
            
            # Build recommendation based on performance
            improvements = []
            if sharpe_improvement > 0.1:
                improvements.append(f"Sharpe +{sharpe_improvement:.2f}")
            if return_improvement > 0:
                improvements.append(f"Return +{return_improvement:.2f}%")
            if dd_improvement > 0:  # Less negative is better
                improvements.append(f"Drawdown {dd_improvement:+.2f}%")
            
            if improvements:
                recommendation = f"Update recommended: {', '.join(improvements)}"
            elif sharpe_improvement < -0.1:
                recommendation = "CAUTION: New params have lower Sharpe. Current config may be better."
            else:
                recommendation = "Update suggested, but performance difference is minimal"
        
        return {
            'should_update': True,
            'params_changed': params_changed,
            'recommendation': recommendation,
            'performance_comparison': performance_comparison
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
                    params_changed TEXT,
                    performance_comparison TEXT
                )
            """)
            
            # Add performance_comparison column if it doesn't exist (for existing DBs)
            try:
                cursor.execute("ALTER TABLE optimization_results ADD COLUMN performance_comparison TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Insert result
            timestamp = int(datetime.now().timestamp())
            best_params_json = json.dumps(result.get('best_params', {}))
            best_metrics_json = json.dumps(result.get('best_metrics', {}))
            comparison = result.get('comparison', {})
            params_changed_json = json.dumps(comparison.get('params_changed', {}))
            performance_comparison_json = json.dumps(comparison.get('performance_comparison'))
            
            cursor.execute("""
                INSERT INTO optimization_results
                (timestamp, best_params, best_metrics, should_update, params_changed, performance_comparison)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                best_params_json,
                best_metrics_json,
                comparison.get('should_update', False),
                params_changed_json,
                performance_comparison_json
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving optimization result: {e}")
            conn.rollback()
            conn.close()


# Import numpy here to avoid circular import
import numpy as np

