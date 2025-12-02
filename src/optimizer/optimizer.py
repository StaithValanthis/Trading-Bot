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
from ..utils import parse_timeframe_to_hours
import pandas as pd

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
        
        # Multiple-testing-aware Sharpe threshold
        # Using a more conservative correction: log-based but capped to avoid over-penalizing
        base_min_sharpe = self.config.optimizer.min_sharpe_ratio
        n_trials_cfg = self.config.optimizer.n_trials or 1
        adjusted_min_sharpe = base_min_sharpe
        try:
            # More conservative correction: smaller multiplier and cap at +0.2
            # For 300 trials: 0.75 + 0.05 * log10(30) ≈ 0.75 + 0.05 * 1.477 ≈ 0.82
            # For 100 trials: 0.75 + 0.05 * log10(10) ≈ 0.75 + 0.05 * 1.0 ≈ 0.80
            correction = 0.05 * math.log10(max(1.0, n_trials_cfg / 10.0))
            adjusted_min_sharpe = base_min_sharpe + min(correction, 0.2)  # Cap at +0.2
        except Exception:
            # Fallback to base if anything goes wrong
            adjusted_min_sharpe = base_min_sharpe

        self.logger.info(
            "Optimizer Sharpe thresholds - base: %.2f, adjusted for %d trials: %.2f",
            base_min_sharpe,
            n_trials_cfg,
            adjusted_min_sharpe,
        )
        
        # Log walk-forward settings
        embargo_days = getattr(self.config.optimizer, "embargo_days", 0)
        walk_forward_folds = getattr(self.config.optimizer, "walk_forward_folds", 1)
        self.logger.info(
            "Walk-forward settings: folds=%d, embargo_days=%d, purge=enabled (computed per parameter set)",
            walk_forward_folds,
            embargo_days,
        )

        # Relax criteria if data is limited (less than 6 months)
        min_trades_threshold = self.config.optimizer.min_trades
        # Start from the multiple-testing-adjusted Sharpe threshold
        min_sharpe_threshold = adjusted_min_sharpe
        max_dd_threshold = self.config.optimizer.max_drawdown_pct
        
        if total_days < 180:  # Less than 6 months
            # Relax criteria proportionally
            relaxation_factor = max(0.5, total_days / 180.0)  # Between 50% and 100%
            min_trades_threshold = max(
                5, int(self.config.optimizer.min_trades * relaxation_factor)
            )
            # Apply relaxation on top of multiple-testing adjustment
            min_sharpe_threshold = adjusted_min_sharpe * relaxation_factor
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
        num_pass_all = 0  # For summary logging

        for i, params in enumerate(param_sets):
            try:
                param_start_time = time_module.time()
                
                # Log parameter variation for first few sets to verify exploration
                if i < 3:
                    self.logger.info(
                        f"Parameter set {i+1}/{len(param_sets)} params: "
                        f"ma_short={params.get('ma_short')}, ma_long={params.get('ma_long')}, "
                        f"momentum={params.get('momentum_lookback')}, top_k={params.get('top_k')}, "
                        f"ranking={params.get('ranking_window')}, rebalance={params.get('rebalance_frequency_hours')}"
                    )
                
                # Create config with these parameters
                test_config = self._create_test_config(params, timeframe)

                # Run walk-forward backtests
                self.logger.debug(f"Parameter set {i+1}/{len(param_sets)}: Starting walk-forward backtest...")
                walk_forward_results = self._walk_forward_backtest(
                    symbol_data,
                    test_config,
                    walk_forward_window_days,
                    universe_history=universe_history,
                    params=params,
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
                    # Split into in-sample and out-of-sample windows
                    # Results now have 'is_oos' and 'fold_index' tags from _walk_forward_backtest
                    is_results = [r for r in walk_forward_results if r.get('is_oos') == 'is']
                    oos_results = [r for r in walk_forward_results if r.get('is_oos') == 'oos']
                    
                    # Fallback: if no tags, use old 70/30 split for backwards compatibility
                    if not is_results and not oos_results:
                        n_windows = len(walk_forward_results)
                        split_idx = max(1, int(n_windows * 0.7))
                        is_results = walk_forward_results[:split_idx]
                        oos_results = walk_forward_results[split_idx:] or walk_forward_results
                    
                    # Use all OOS results for evaluation (across all folds)
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
                    
                    # Diagnostic logging for first parameter set
                    if i == 0:
                        self.logger.info(
                            f"Parameter set 1 diagnostic: IS windows={len(is_results)}, OOS windows={len(oos_eval)}, "
                            f"IS avg_trades={avg_trades_is:.1f}, OOS avg_trades={avg_trades_oos:.1f}, "
                            f"IS trades per window: {[r.get('total_trades', 0) for r in is_results[:3]]}, "
                            f"OOS trades per window: {[r.get('total_trades', 0) for r in oos_eval[:3]]}"
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
                    # For crypto strategies, OOS performance is more important than IS
                    passes_is = (
                        min_trades_is >= min_trades_threshold
                        and avg_sharpe_is >= min_sharpe_threshold
                        and avg_dd_is >= max_dd_threshold
                    )
                    # OOS criteria: slightly looser Sharpe (0.7x), but also allow if OOS Sharpe is excellent (>1.2)
                    # This accounts for cases where IS underperforms but OOS is strong (better generalization)
                    oos_sharpe_min = min_sharpe_threshold * 0.7
                    oos_sharpe_excellent = 1.2  # If OOS Sharpe > 1.2, accept even if IS is slightly below threshold
                    passes_oos = (
                        min_trades_oos >= min_trades_threshold
                        and (
                            avg_sharpe_oos >= oos_sharpe_min
                            or (avg_sharpe_oos >= oos_sharpe_excellent and avg_sharpe_is >= min_sharpe_threshold * 0.6)
                        )
                        and avg_dd_oos >= max_dd_threshold * 1.2
                    )
                    
                    # Log failed criteria (first failed set and every 10th after)
                    if (not passes_is or not passes_oos) and (i == 0 or (i + 1) % 10 == 0):
                        fail_reasons = []
                        if min_trades_is < min_trades_threshold:
                            fail_reasons.append(
                                f"IS min_trades {min_trades_is:.0f} < threshold {min_trades_threshold:.0f}"
                            )
                        if avg_sharpe_is < min_sharpe_threshold:
                            fail_reasons.append(
                                f"IS Sharpe {avg_sharpe_is:.2f} < threshold {min_sharpe_threshold:.2f}"
                            )
                        if avg_dd_is < max_dd_threshold:
                            fail_reasons.append(
                                f"IS DD {avg_dd_is:.2f}% < threshold {max_dd_threshold:.1f}%"
                            )
                        if min_trades_oos < min_trades_threshold:
                            fail_reasons.append(
                                f"OOS min_trades {min_trades_oos:.0f} < threshold {min_trades_threshold:.0f}"
                            )
                        if avg_sharpe_oos < min_sharpe_threshold * 0.7:
                            fail_reasons.append(
                                f"OOS Sharpe {avg_sharpe_oos:.2f} < relaxed threshold {min_sharpe_threshold * 0.7:.2f}"
                            )
                        if avg_dd_oos < max_dd_threshold * 1.2:
                            fail_reasons.append(
                                f"OOS DD {avg_dd_oos:.2f}% < relaxed threshold {max_dd_threshold * 1.2:.1f}%"
                            )
                        self.logger.info(
                            "Parameter set %d failed criteria: %s",
                            i + 1,
                            "; ".join(fail_reasons) if fail_reasons else "unspecified",
                        )

                    # Store all results with sufficient trades for ranking
                    # We'll rank them and pick the best even if none meet strict criteria
                    # Robustness metrics are evaluated on OOS windows
                    positive_oos_windows = sum(
                        1 for r in oos_results if r.get("total_return_pct", 0) > 0
                    )
                    oos_window_count = len(oos_results)
                    positive_window_ratio = (
                        positive_oos_windows / oos_window_count
                        if oos_window_count
                        else 0.0
                    )
                    oos_sharpes = [
                        r["sharpe_ratio"]
                        for r in oos_results
                        if "sharpe_ratio" in r
                    ]
                    oos_sharpe_std = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
                    worst_oos_dd = min(
                        (r["max_drawdown_pct"] for r in oos_results), default=0.0
                    )

                    # Enforce robustness filters on top of baseline OOS criteria
                    min_positive_windows_cfg = getattr(
                        self.config.optimizer, "min_positive_windows", 0.5
                    )
                    max_oos_sharpe_std_cfg = getattr(
                        self.config.optimizer, "max_oos_sharpe_std", None
                    )
                    max_worst_oos_dd_cfg = getattr(
                        self.config.optimizer, "max_worst_oos_dd", None
                    )

                    if passes_oos and oos_window_count > 0:
                        if positive_window_ratio < min_positive_windows_cfg:
                            self.logger.info(
                                "Parameter set %d rejected on OOS robustness: "
                                "only %.1f%% of OOS windows positive (min %.1f%%)",
                                i + 1,
                                positive_window_ratio * 100,
                                min_positive_windows_cfg * 100,
                            )
                            passes_oos = False
                        if (
                            passes_oos
                            and max_oos_sharpe_std_cfg is not None
                            and oos_sharpe_std > max_oos_sharpe_std_cfg
                        ):
                            self.logger.info(
                                "Parameter set %d rejected on OOS robustness: "
                                "OOS Sharpe std %.2f > max %.2f",
                                i + 1,
                                oos_sharpe_std,
                                max_oos_sharpe_std_cfg,
                            )
                            passes_oos = False
                        if (
                            passes_oos
                            and max_worst_oos_dd_cfg is not None
                            and worst_oos_dd < max_worst_oos_dd_cfg
                        ):
                            self.logger.info(
                                "Parameter set %d rejected on OOS robustness: "
                                "worst OOS drawdown %.2f%% < max allowed %.2f%%",
                                i + 1,
                                worst_oos_dd,
                                max_worst_oos_dd_cfg,
                            )
                            passes_oos = False

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
                        "positive_window_ratio": positive_window_ratio,
                        "oos_sharpe_std": oos_sharpe_std,
                        "worst_oos_dd": worst_oos_dd,
                    }
                    
                    # Store results based on priority:
                    # 1. Passes all criteria (preferred)
                    # 2. Passes OOS criteria (good for generalization)
                    # 3. Has sufficient trades for fallback ranking (with limited data)
                    already_added = False
                    
                    if passes_is and passes_oos:
                        results.append(all_results)
                        num_pass_all += 1
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

            # Summary logging
            embargo_days = getattr(self.config.optimizer, "embargo_days", 0)
            walk_forward_folds = getattr(self.config.optimizer, "walk_forward_folds", 1)
            self.logger.info(
                "Optimizer summary: tested %d parameter sets, %d passed all criteria, %d total candidates stored "
                "(folds=%d, embargo=%d days, purge=enabled)",
                len(param_sets),
                num_pass_all,
                len(results),
                walk_forward_folds,
                embargo_days,
            )
            top_n = min(3, len(results))
            for rank, res in enumerate(results[:top_n], start=1):
                self.logger.info(
                    "Top %d: OOS Sharpe=%.2f, OOS DD=%.2f%%, pos_win_ratio=%.1f%%, params=%s",
                    rank,
                    res.get("avg_sharpe_oos", res.get("avg_sharpe", 0.0)),
                    res.get("avg_dd_oos", res.get("avg_drawdown_pct", 0.0)),
                    res.get("positive_window_ratio", 0.0) * 100.0,
                    res.get("params"),
                )

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
    
    def _compute_purge_bars(self, params: Dict[str, Any], timeframe: str) -> int:
        """
        Compute purge length in bars based on maximum lookback used by the strategy.
        
        This prevents data leakage from overlapping lookback periods (e.g., MA, momentum, ranking).
        
        Args:
            params: Parameter dictionary containing strategy lookback parameters
            timeframe: Trading timeframe (e.g., '1h', '4h')
        
        Returns:
            Purge length in bars (with safety margin)
        """
        hours_per_bar = parse_timeframe_to_hours(timeframe)
        if hours_per_bar <= 0:
            return 0
        
        # Extract lookback parameters (in bars)
        ma_long = params.get('ma_long', 100)
        momentum_lookback = params.get('momentum_lookback', 24)
        ranking_window = params.get('ranking_window', 18)
        
        # Find maximum lookback
        max_lookback_bars = max(ma_long, momentum_lookback, ranking_window, 1)
        
        # Add safety margin (10 bars or ~10% of max lookback, whichever is larger)
        safety_margin = max(10, int(max_lookback_bars * 0.1))
        purge_bars = max_lookback_bars + safety_margin
        
        return purge_bars
    
    def _walk_forward_backtest(
        self,
        symbol_data: Dict,
        test_config: BotConfig,
        window_days: int,
        universe_history: Optional[Dict[date, Set[str]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Run walk-forward backtests with purged cross-validation, embargo, and multiple folds.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            test_config: Bot configuration for this parameter set
            window_days: Size of each walk-forward window in days
            universe_history: Optional historical universe membership
            params: Parameter dictionary (for computing purge period)
        
        Returns:
            List of backtest result dictionaries (one per window, aggregated across folds)
        """
        from ..backtest.backtester import Backtester
        
        backtester = Backtester(test_config)
        
        # Get date range
        all_dates = []
        for df in symbol_data.values():
            if not df.empty:
                all_dates.extend(df.index.tolist())
        
        if not all_dates:
            return []
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Create non-overlapping windows
        current_date = start_date
        windows = []
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=window_days)
            if window_end > end_date:
                window_end = end_date
            
            windows.append((current_date, window_end))
            current_date = window_end
        
        if not windows:
            return []
        
        # Compute purge period (in bars and days)
        timeframe = test_config.exchange.timeframe
        purge_bars = 0
        purge_days = 0.0
        if params:
            purge_bars = self._compute_purge_bars(params, timeframe)
            hours_per_bar = parse_timeframe_to_hours(timeframe)
            if hours_per_bar > 0:
                purge_days = (purge_bars * hours_per_bar) / 24.0
        
        embargo_days = getattr(self.config.optimizer, "embargo_days", 0)
        walk_forward_folds = getattr(self.config.optimizer, "walk_forward_folds", 1)
        
        # Log purge/embargo settings (once per parameter set)
        if params and len(windows) > 0:
            self.logger.debug(
                f"Walk-forward settings: purge_bars={purge_bars} (~{purge_days:.1f} days), "
                f"embargo_days={embargo_days}, folds={walk_forward_folds}, windows={len(windows)}"
            )
        
        # Collect results from all folds
        all_fold_results = []
        
        # Generate folds
        n_windows = len(windows)
        if walk_forward_folds <= 1:
            # Single fold: use 70/30 split (current behavior)
            split_idx = max(1, int(n_windows * 0.7))
            fold_splits = [(split_idx,)]
        else:
            # Multiple folds: shift the IS/OOS boundary
            base_split = max(1, int(n_windows * 0.7))
            fold_splits = []
            for fold_idx in range(walk_forward_folds):
                # Shift boundary by 1-2 windows per fold (ensure we have both IS and OOS)
                shift = fold_idx * max(1, n_windows // (walk_forward_folds * 2))
                split_idx = min(base_split + shift, n_windows - 1)
                if split_idx > 0 and split_idx < n_windows:
                    fold_splits.append((split_idx,))
        
        # Process each fold
        for fold_idx, (split_idx,) in enumerate(fold_splits):
            is_windows = windows[:split_idx]
            oos_windows = windows[split_idx:]
            
            # Apply embargo: skip OOS windows that start within embargo_days of last IS window
            if embargo_days > 0 and is_windows and oos_windows:
                last_is_end = is_windows[-1][1]
                embargo_cutoff = last_is_end + timedelta(days=embargo_days)
                oos_windows = [
                    (w_start, w_end) for w_start, w_end in oos_windows
                    if w_start >= embargo_cutoff
                ]
            
            if not oos_windows:
                # No valid OOS windows after embargo, skip this fold
                continue
            
            # Backtest IS windows (no purge needed for IS)
            is_results = []
            for window_start, window_end in is_windows:
                window_result = self._backtest_window(
                    backtester, symbol_data, window_start, window_end,
                    universe_history, test_config, None  # No purge for IS
                )
                if window_result:
                    is_results.append(window_result)
            
            # Backtest OOS windows (with purge)
            oos_results = []
            for window_start, window_end in oos_windows:
                # Purge training data that overlaps with OOS lookback
                purge_start = None
                if purge_days > 0:
                    purge_start = window_start - timedelta(days=purge_days)
                
                window_result = self._backtest_window(
                    backtester, symbol_data, window_start, window_end,
                    universe_history, test_config, purge_start
                )
                if window_result:
                    oos_results.append(window_result)
            
            # Tag results with fold index for aggregation
            for result in is_results:
                result['fold_index'] = fold_idx
                result['is_oos'] = 'is'
            for result in oos_results:
                result['fold_index'] = fold_idx
                result['is_oos'] = 'oos'
            
            all_fold_results.extend(is_results)
            all_fold_results.extend(oos_results)
        
        # If no folds produced results, return empty list
        if not all_fold_results:
            return []
        
        # Aggregate results: for backwards compatibility, return one result per unique window
        # (averaging across folds if a window appears in multiple folds)
        window_results_map = {}
        for result in all_fold_results:
            # Use a window key (we'll need to track which window this came from)
            # For now, we'll just return all results and let the caller aggregate
            pass
        
        # Return all results (caller will aggregate by IS/OOS across folds)
        return all_fold_results
    
    def _backtest_window(
        self,
        backtester: 'Backtester',
        symbol_data: Dict,
        window_start: datetime,
        window_end: datetime,
        universe_history: Optional[Dict[date, Set[str]]],
        test_config: BotConfig,
        purge_start: Optional[datetime],
    ) -> Optional[Dict]:
        """
        Backtest a single window with optional purge period.
        
        For OOS windows, purge_start indicates the start of the purge period (data before
        window_start that would be visible to the strategy's lookback).
        
        Args:
            backtester: Backtester instance
            symbol_data: Full symbol data dictionary (may contain data beyond the window)
            window_start: Window start datetime
            window_end: Window end datetime
            universe_history: Historical universe membership
            test_config: Bot configuration
            purge_start: Optional purge period start (for OOS: exclude data from [purge_start, window_start))
        
        Returns:
            Backtest result dict or None if window is invalid
        """
        # Filter data to window
        window_data = {}
        for symbol, df in symbol_data.items():
            # Get data for the window
            mask = (df.index >= window_start) & (df.index <= window_end)
            window_df = df[mask].copy()
            
            # For OOS windows with purge: include lookback data but exclude purge period
            # The backtester needs lookback to compute indicators, but we must avoid
            # data leakage from the purge period
            if purge_start is not None:
                # Include data before purge_start for lookback (safe from leakage)
                lookback_df = df[df.index < purge_start].copy()
                # Combine safe lookback + window data
                if not lookback_df.empty:
                    window_df = pd.concat([lookback_df, window_df]).sort_index()
                # If no safe lookback, just use window data (backtester may fail or use defaults)
            else:
                # For IS windows: include all data before window_start for lookback
                lookback_df = df[df.index < window_start].copy()
                if not lookback_df.empty:
                    window_df = pd.concat([lookback_df, window_df]).sort_index()
            
            # Reduced minimum bar requirement: need at least ma_long + some buffer for signals
            # The 100-bar minimum was too strict for short windows
            min_required_bars = max(
                test_config.strategy.trend.ma_long + 10,  # ma_long + small buffer
                50  # Absolute minimum for any meaningful backtest
            )
            
            if not window_df.empty and len(window_df) >= min_required_bars:
                window_data[symbol] = window_df
        
        if not window_data:
            return None
        
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
                return result
            else:
                self.logger.debug(f"Window {window_start.date()} returned error: {result.get('error')}")
        except Exception as e:
            self.logger.warning(f"Error in walk-forward window {window_start} to {window_end}: {e}", exc_info=True)
        
        return None
    
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
    
    def optimize_funding_strategy(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        universe_history: Optional[Dict[date, Set[str]]] = None,
    ) -> Dict:
        """
        Optimize funding opportunity strategy parameters using walk-forward analysis.
        
        Args:
            symbols: List of symbols to optimize on
            timeframe: Timeframe (e.g., '1h')
            start_date: Optional start date for data
            end_date: Optional end date for data
            universe_history: Optional historical universe membership
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("="*80)
        self.logger.info("=== FUNDING STRATEGY OPTIMIZATION START ===")
        self.logger.info("="*80)
        
        funding_config = self.config.optimizer.funding
        
        if not funding_config.enabled:
            self.logger.info("Funding optimization is disabled in config. Skipping.")
            return {'skipped': True, 'reason': 'disabled'}
        
        if not self.config.strategy.funding_opportunity.enabled:
            self.logger.warning("Funding opportunity strategy is disabled in config. Cannot optimize.")
            return {'error': 'Funding strategy not enabled in config'}
        
        # Load historical data (same as main optimizer)
        if start_date is not None:
            since = int(start_date.timestamp() * 1000)
        else:
            lookback_days = funding_config.lookback_months * 30
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        if end_date is not None:
            until = int(end_date.timestamp() * 1000)
        else:
            until = None
        
        symbol_data = {}
        for symbol in symbols:
            try:
                df = self.store.get_ohlcv(symbol, timeframe, since=since)
                if end_date is not None and not df.empty:
                    df = df[df.index <= end_date]
                if not df.empty and len(df) >= 200:
                    symbol_data[symbol] = df
            except Exception as e:
                self.logger.warning(f"Error loading data for {symbol}: {e}")
                continue
        
        if not symbol_data:
            return {'error': 'No data available for funding optimization'}
        
        # Generate funding parameter sets
        rng = random
        seed = funding_config.random_seed
        if seed is not None:
            rng = random.Random(seed)
        
        param_sets = self._generate_funding_param_candidates(
            funding_config.trials,
            funding_config.param_ranges,
            rng=rng,
            method=funding_config.sample_method
        )
        
        # Add current config as baseline
        current_funding_params = self._get_current_funding_params()
        if current_funding_params not in param_sets:
            param_sets.insert(0, current_funding_params)
        
        self.logger.info(f"Testing {len(param_sets)} funding parameter sets")
        
        # Walk-forward analysis
        window_days = funding_config.walk_forward_window_days
        results = []
        
        import time as time_module
        start_time = time_module.time()
        
        for i, params in enumerate(param_sets):
            try:
                # Create test config with funding parameters
                test_config = self._create_funding_test_config(params, timeframe)
                
                # Run walk-forward backtests
                walk_forward_results = self._walk_forward_backtest(
                    symbol_data,
                    test_config,
                    window_days,
                    universe_history=universe_history,
                    params=None,  # Funding params don't affect purge (no lookback)
                )
                
                if not walk_forward_results:
                    continue
                
                # Split IS/OOS
                n_windows = len(walk_forward_results)
                split_idx = max(1, int(n_windows * 0.7))
                is_results = walk_forward_results[:split_idx]
                oos_results = walk_forward_results[split_idx:] or walk_forward_results
                
                # Aggregate metrics
                def _agg(results_list):
                    if not results_list:
                        return 0.0, 0.0, 0.0, 0, 0
                    return (
                        np.mean([r["total_return_pct"] for r in results_list]),
                        np.mean([r["sharpe_ratio"] for r in results_list]),
                        np.mean([r["max_drawdown_pct"] for r in results_list]),
                        np.mean([r["total_trades"] for r in results_list]),
                        min([r["total_trades"] for r in results_list]),
                    )
                
                avg_return_is, avg_sharpe_is, avg_dd_is, avg_trades_is, min_trades_is = _agg(is_results)
                avg_return_oos, avg_sharpe_oos, avg_dd_oos, avg_trades_oos, min_trades_oos = _agg(oos_results)
                
                # Extract funding-specific metrics
                funding_trades_oos = []
                funding_pnl_oos = 0.0
                total_pnl_oos = 0.0
                
                for r in oos_results:
                    funding_metrics = r.get('funding_metrics', {})
                    funding_trades_oos.append(funding_metrics.get('total_funding_trades', 0))
                    # Estimate funding PnL from trades (if available in result)
                    # For now, we'll use total return as proxy
                    total_pnl_oos += r.get("total_return_pct", 0.0)
                
                avg_funding_trades_oos = np.mean(funding_trades_oos) if funding_trades_oos else 0.0
                min_funding_trades_oos = min(funding_trades_oos) if funding_trades_oos else 0.0
                
                # Calculate funding PnL share (approximate from funding trade count)
                # This is a simplified metric - in production, track actual funding PnL separately
                funding_pnl_share = min_funding_trades_oos / max(avg_trades_oos, 1.0) if avg_trades_oos > 0 else 0.0
                
                # Check criteria
                passes_oos = (
                    min_funding_trades_oos >= funding_config.min_trades
                    and avg_sharpe_oos >= funding_config.min_sharpe
                    and avg_dd_oos >= funding_config.max_dd
                    and funding_pnl_share >= funding_config.min_funding_pnl_share
                )
                
                result = {
                    "params": params,
                    "avg_return_pct": np.mean([r["total_return_pct"] for r in walk_forward_results]),
                    "avg_sharpe": np.mean([r["sharpe_ratio"] for r in walk_forward_results]),
                    "avg_drawdown_pct": np.mean([r["max_drawdown_pct"] for r in walk_forward_results]),
                    "avg_trades": np.mean([r["total_trades"] for r in walk_forward_results]),
                    "avg_sharpe_oos": avg_sharpe_oos,
                    "avg_dd_oos": avg_dd_oos,
                    "min_funding_trades_oos": min_funding_trades_oos,
                    "funding_pnl_share": funding_pnl_share,
                    "passes_oos": passes_oos,
                }
                
                if passes_oos:
                    results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Funding param set {i+1}/{len(param_sets)}: "
                        f"OOS Sharpe={avg_sharpe_oos:.2f}, funding trades={min_funding_trades_oos:.0f}"
                    )
            except Exception as e:
                self.logger.warning(f"Error testing funding parameter set {i}: {e}")
                continue
        
        # Sort by OOS Sharpe
        if results:
            results.sort(key=lambda x: x['avg_sharpe_oos'], reverse=True)
            best_result = results[0]
            
            self.logger.info(
                f"Funding optimization complete: {len(results)}/{len(param_sets)} passed criteria. "
                f"Best OOS Sharpe: {best_result['avg_sharpe_oos']:.2f}"
            )
            
            return {
                'best_params': best_result['params'],
                'best_metrics': {
                    'avg_sharpe_oos': best_result['avg_sharpe_oos'],
                    'avg_dd_oos': best_result['avg_dd_oos'],
                    'min_funding_trades_oos': best_result['min_funding_trades_oos'],
                    'funding_pnl_share': best_result['funding_pnl_share'],
                },
                'all_results': results[:10],
            }
        
        self.logger.warning("No funding parameter sets met criteria")
        return {'error': 'No valid funding parameter sets found'}
    
    def _get_current_funding_params(self) -> Dict:
        """Extract current funding strategy parameters."""
        fo = self.config.strategy.funding_opportunity
        params = {
            'min_funding_rate': fo.min_funding_rate,
            'exit_funding_threshold': fo.exit.exit_funding_threshold,
            'base_size_fraction': fo.sizing.base_size_fraction,
            'max_total_funding_exposure': fo.risk.max_total_funding_exposure,
            'max_holding_hours': fo.exit.max_holding_hours,
            'stop_loss_atr_multiplier': fo.exit.stop_loss_atr_multiplier,
            'take_profit_rr': fo.exit.take_profit_rr,
            'require_trend_alignment': fo.entry.require_trend_alignment,
        }
        return params
    
    def _generate_funding_param_candidates(
        self,
        n_trials: int,
        param_ranges: Dict,
        rng=random,
        method: str = "uniform",
    ) -> List[Dict]:
        """Generate random funding parameter sets."""
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
    
    def _create_funding_test_config(self, params: Dict, timeframe: str) -> BotConfig:
        """Create a test config with funding parameters."""
        import copy
        
        test_config = copy.deepcopy(self.config)
        test_config.exchange.timeframe = timeframe
        
        # Update funding parameters
        fo = test_config.strategy.funding_opportunity
        
        if 'min_funding_rate' in params:
            fo.min_funding_rate = params['min_funding_rate']
        if 'exit_funding_threshold' in params:
            fo.exit.exit_funding_threshold = params['exit_funding_threshold']
        if 'base_size_fraction' in params:
            fo.sizing.base_size_fraction = params['base_size_fraction']
        if 'max_total_funding_exposure' in params:
            fo.risk.max_total_funding_exposure = params['max_total_funding_exposure']
        if 'max_holding_hours' in params:
            fo.exit.max_holding_hours = params['max_holding_hours']
        if 'stop_loss_atr_multiplier' in params:
            fo.exit.stop_loss_atr_multiplier = params['stop_loss_atr_multiplier']
        if 'take_profit_rr' in params:
            fo.exit.take_profit_rr = params['take_profit_rr']
        if 'require_trend_alignment' in params:
            fo.entry.require_trend_alignment = params['require_trend_alignment']
        
        # Optionally disable main strategy for pure funding test
        if self.config.optimizer.funding.disable_main_strategy:
            test_config.strategy.trend.ma_short = 999  # Effectively disable
            test_config.strategy.cross_sectional.top_k = 0
        
        return test_config


# Import numpy here to avoid circular import
import numpy as np

