"""Universe parameter optimization framework."""

import json
import random
import sqlite3
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

from ..config import BotConfig, UniverseConfig
from ..universe.selector import UniverseSelector
from ..universe.store import UniverseStore
from ..data.ohlcv_store import OHLCVStore
from ..backtest.backtester import Backtester
from ..utils import parse_timeframe_to_hours
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class UniverseOptimizationResult:
    """Results for a single universe parameter set."""
    # Parameters tested
    params: Dict[str, Any]
    
    # Strategy performance
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    estimated_fees: float = 0.0
    
    # Universe quality
    avg_universe_size: float = 0.0
    median_universe_size: float = 0.0
    min_universe_size: int = 0
    max_universe_size: int = 0
    universe_size_std: float = 0.0
    
    avg_volume_24h: float = 0.0
    median_volume_24h: float = 0.0
    min_volume_24h: float = 0.0
    p25_volume_24h: float = 0.0
    p75_volume_24h: float = 0.0
    
    avg_additions_per_rebalance: float = 0.0
    avg_removals_per_rebalance: float = 0.0
    total_additions: int = 0
    total_removals: int = 0
    universe_turnover_rate: float = 0.0
    
    avg_time_in_universe_days: float = 0.0
    pct_symbols_stayed_entire_period: float = 0.0
    
    pct_pnl_from_top5: float = 0.0
    pct_pnl_from_top10: float = 0.0
    unique_symbols_traded: int = 0
    
    # Robustness
    performance_by_regime: Dict[str, float] = None
    regime_consistency_score: float = 0.0
    sensitivity_score: float = 0.0
    
    # Metadata
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.performance_by_regime is None:
            self.performance_by_regime = {}


class UniverseOptimizer:
    """Optimize universe selection parameters."""
    
    def __init__(
        self,
        base_config: BotConfig,
        ohlcv_store: OHLCVStore,
        db_path: str
    ):
        """
        Initialize universe optimizer.
        
        Args:
            base_config: Base bot configuration
            ohlcv_store: OHLCV data store
            db_path: Path to database for storing results
        """
        self.base_config = base_config
        self.ohlcv_store = ohlcv_store
        self.db_path = db_path
        self.logger = get_logger(__name__)
        self._init_results_db()
    
    def _init_results_db(self):
        """Initialize results database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe_optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                params TEXT NOT NULL,
                strategy_performance TEXT NOT NULL,
                universe_quality TEXT NOT NULL,
                robustness TEXT NOT NULL,
                metadata TEXT NOT NULL,
                composite_score REAL,
                backtest_start_date TEXT,
                backtest_end_date TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_opt_timestamp
            ON universe_optimization_runs(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def generate_parameter_combinations(
        self,
        n_combinations: int = 200,
        method: str = "random"
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations to test.
        
        Args:
            n_combinations: Number of combinations to generate
            method: 'random' or 'grid'
        
        Returns:
            List of parameter dictionaries
        """
        # Parameter ranges (from design doc)
        param_ranges = {
            'min_24h_volume_entry': [5_000_000, 10_000_000, 20_000_000, 30_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000],
            'volume_exit_ratio': [0.5, 0.6, 0.7, 0.8, 0.9],  # Ratio of entry threshold
            'volume_check_days': [3, 5, 7, 10, 14],
            'min_history_days': [14, 21, 30, 45, 60, 90, 120, 180],
            'warmup_days': [0, 7, 14, 21, 30, 45],
            'min_time_in_universe_days': [3, 5, 7, 10, 14, 21],
            'max_turnover_per_rebalance_pct': [10, 15, 20, 25, 30, 40, 50],
            'max_realized_vol_pct': [100, 150, 200, 250, 300, 400, 500],
        }
        
        combinations = []
        
        if method == "random":
            for _ in range(n_combinations):
                params = {}
                for param_name, values in param_ranges.items():
                    params[param_name] = random.choice(values)
                
                # Calculate exit threshold from entry and ratio
                exit_ratio = params.pop('volume_exit_ratio')  # Remove ratio
                params['min_24h_volume_exit'] = params['min_24h_volume_entry'] * exit_ratio
                
                # Add fixed parameters from base config
                base_universe = self.base_config.universe
                params['min_open_interest'] = base_universe.min_open_interest
                params['max_spread_bps'] = base_universe.max_spread_bps
                params['max_data_gap_pct'] = base_universe.max_data_gap_pct
                params['max_days_since_last_update'] = base_universe.max_days_since_last_update
                params['min_price_usdt'] = base_universe.min_price_usdt
                params['limit_move_frequency_pct'] = base_universe.limit_move_frequency_pct
                params['rebalance_frequency_hours'] = base_universe.rebalance_frequency_hours
                params['include_list'] = base_universe.include_list
                params['exclude_list'] = base_universe.exclude_list
                
                combinations.append(params)
        
        elif method == "grid":
            # Grid search (smaller subset due to combinatorial explosion)
            import itertools
            
            # Use coarser grid for grid search
            grid_ranges = {
                'min_24h_volume_entry': [10_000_000, 30_000_000, 50_000_000, 100_000_000],
                'volume_exit_ratio': [0.6, 0.7, 0.8],
                'min_history_days': [30, 60, 90],
                'warmup_days': [7, 14, 30],
            }
            
            param_names = list(grid_ranges.keys())
            param_values = list(grid_ranges.values())
            
            for combination in itertools.product(*param_values):
                params = dict(zip(param_names, combination))
                
                # Calculate exit threshold
                exit_ratio = params.pop('volume_exit_ratio')
                params['min_24h_volume_exit'] = params['min_24h_volume_entry'] * exit_ratio
                
                # Add defaults for other params
                params['volume_check_days'] = 7
                params['min_time_in_universe_days'] = 7
                params['max_turnover_per_rebalance_pct'] = 20
                params['max_realized_vol_pct'] = 200
                
                # Add fixed parameters
                base_universe = self.base_config.universe
                params['min_open_interest'] = base_universe.min_open_interest
                params['max_spread_bps'] = base_universe.max_spread_bps
                params['max_data_gap_pct'] = base_universe.max_data_gap_pct
                params['max_days_since_last_update'] = base_universe.max_days_since_last_update
                params['min_price_usdt'] = base_universe.min_price_usdt
                params['limit_move_frequency_pct'] = base_universe.limit_move_frequency_pct
                params['rebalance_frequency_hours'] = base_universe.rebalance_frequency_hours
                params['include_list'] = base_universe.include_list
                params['exclude_list'] = base_universe.exclude_list
                
                combinations.append(params)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations using {method} method")
        return combinations
    
    def build_historical_universe(
        self,
        universe_config: UniverseConfig,
        timeframe: str,
        test_date: date,
        all_symbols: List[str],
        symbol_data: Dict[str, pd.DataFrame],
        volume_cache: Dict[Tuple[str, date], float]
    ) -> Set[str]:
        """
        Build universe at a specific historical date (time-respecting).
        
        Args:
            universe_config: Universe configuration
            timeframe: Trading timeframe
            test_date: Date to build universe for
            all_symbols: All available symbols
            symbol_data: Dictionary of symbol to OHLCV DataFrame
            volume_cache: Cache of 24h volumes per symbol per date
        
        Returns:
            Set of symbols in universe at test_date
        """
        universe = set()
        
        for symbol in all_symbols:
            if symbol in universe_config.exclude_list:
                continue
            
            if symbol in universe_config.include_list:
                # Always include
                universe.add(symbol)
                continue
            
            # Check if symbol has data up to test_date
            if symbol not in symbol_data:
                continue
            
            df = symbol_data[symbol]
            if df.empty:
                continue
            
            # Check if symbol has data at or before test_date
            df_up_to_date = df[df.index.date <= test_date]
            if df_up_to_date.empty:
                continue
            
            # Check if symbol was delisted (last data significantly before test_date)
            last_data_date = df_up_to_date.index[-1].date()
            days_since_last_data = (test_date - last_data_date).days
            if days_since_last_data > universe_config.max_days_since_last_update:
                # Symbol delisted, skip
                continue
            
            # Check history requirement (timeframe-aware)
            hours_per_bar = parse_timeframe_to_hours(timeframe)
            if hours_per_bar <= 0:
                continue
            candles_per_day = max(24.0 / hours_per_bar, 1.0)
            required_bars = int(math.ceil(universe_config.min_history_days * candles_per_day))
            if len(df_up_to_date) < required_bars:
                # Insufficient history for this timeframe
                continue
            
            # Calculate listing date (first data timestamp)
            listing_date = df_up_to_date.index[0].date()
            days_since_listing = (test_date - listing_date).days
            
            # Check warm-up period
            if days_since_listing < (universe_config.min_history_days + universe_config.warmup_days):
                # Still in warm-up period
                continue
            
            # Calculate 24h volume at test_date (or closest available)
            volume_key = (symbol, test_date)
            if volume_key not in volume_cache:
                # Calculate from OHLCV data
                df_24h = df_up_to_date[
                    (df_up_to_date.index.date >= test_date - timedelta(days=1)) &
                    (df_up_to_date.index.date <= test_date)
                ]
                if not df_24h.empty:
                    # Approximate 24h volume as sum of hourly volumes * close price
                    volume_24h = (df_24h['volume'] * df_24h['close']).sum()
                else:
                    volume_24h = 0.0
                volume_cache[volume_key] = volume_24h
            else:
                volume_24h = volume_cache[volume_key]
            
            # Check volume threshold (simplified: check at test_date, not consecutive days for now)
            # TODO: Implement proper consecutive days check for hysteresis
            if volume_24h < universe_config.min_24h_volume_entry:
                continue
            
            # Check volatility (simplified check)
            if len(df_up_to_date) >= 30 * 24:  # At least 30 days of 1h data
                recent_df = df_up_to_date.tail(30 * 24)
                returns = recent_df['close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility_pct = returns.std() * np.sqrt(365 * 24) * 100
                    if volatility_pct > universe_config.max_realized_vol_pct:
                        continue
            
            # Check price floor
            current_price = df_up_to_date['close'].iloc[-1]
            if current_price < universe_config.min_price_usdt:
                continue
            
            # Passed all checks, add to universe
            universe.add(symbol)
        
        return universe
    
    def calculate_24h_volume_time_series(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date
    ) -> Dict[Tuple[str, date], float]:
        """
        Pre-calculate 24h volume time series for all symbols and dates.
        
        Args:
            symbol_data: Dictionary of symbol to OHLCV DataFrame
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary mapping (symbol, date) to 24h volume
        """
        volume_cache = {}
        
        current_date = start_date
        while current_date <= end_date:
            for symbol, df in symbol_data.items():
                if df.empty:
                    continue
                
                # Get data up to current_date
                df_up_to_date = df[df.index.date <= current_date]
                if df_up_to_date.empty:
                    continue
                
                # Calculate 24h volume ending at current_date
                df_24h = df_up_to_date[
                    (df_up_to_date.index.date >= current_date - timedelta(days=1)) &
                    (df_up_to_date.index.date <= current_date)
                ]
                
                if not df_24h.empty:
                    volume_24h = (df_24h['volume'] * df_24h['close']).sum()
                    volume_cache[(symbol, current_date)] = volume_24h
            
            current_date += timedelta(days=1)
        
        return volume_cache
    
    def run_universe_backtest(
        self,
        universe_config: UniverseConfig,
        symbol_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        timeframe: str
    ) -> Tuple[Dict, Dict]:
        """
        Run backtest with dynamic universe construction.
        
        Args:
            universe_config: Universe configuration to test
            symbol_data: Dictionary of symbol to OHLCV DataFrame
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Trading timeframe
        
        Returns:
            Tuple of (strategy_results, universe_metrics)
        """
        # Pre-calculate volume time series
        self.logger.debug("Pre-calculating volume time series...")
        volume_cache = self.calculate_24h_volume_time_series(symbol_data, start_date, end_date)
        
        # Get all symbols
        all_symbols = list(symbol_data.keys())
        
        # Build universe at each rebalance date
        universe_history: Dict[date, Set[str]] = {}
        current_date = start_date
        rebalance_interval = timedelta(hours=universe_config.rebalance_frequency_hours)
        
        self.logger.debug(f"Building historical universe from {start_date} to {end_date}...")
        
        while current_date <= end_date:
            universe = self.build_historical_universe(
                universe_config,
                timeframe,
                current_date,
                all_symbols,
                symbol_data,
                volume_cache
            )
            universe_history[current_date] = universe
            
            # Move to next rebalance date
            current_date += rebalance_interval
            if current_date > end_date:
                break
        
        # Calculate universe metrics
        universe_metrics = self._calculate_universe_metrics(universe_history, volume_cache, symbol_data)
        
        # Filter symbol_data to only include symbols that appear in the universe at least once
        universe_symbols = set()
        for universe_set in universe_history.values():
            universe_symbols.update(universe_set)
        
        # Create filtered symbol_data for strategy backtest
        filtered_symbol_data = {
            symbol: symbol_data[symbol]
            for symbol in universe_symbols
            if symbol in symbol_data
        }
        
        # Run strategy backtest using universe-aware backtester
        strategy_config = deepcopy(self.base_config)
        strategy_config.universe = universe_config
        
        backtester = Backtester(strategy_config)
        strategy_results = backtester.backtest(
            filtered_symbol_data,
            initial_capital=10000.0,
            taker_fee=0.00055,
            universe_history=universe_history,
        )
        
        # Add universe membership tracking to results (for metrics)
        strategy_results['universe_history'] = universe_history
        
        return strategy_results, universe_metrics
    
    def _calculate_universe_metrics(
        self,
        universe_history: Dict[date, Set[str]],
        volume_cache: Dict[Tuple[str, date], float],
        symbol_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Calculate universe quality metrics."""
        universe_sizes = [len(u) for u in universe_history.values()]
        volumes_by_date: Dict[date, List[float]] = {}
        
        for date_key, universe_set in universe_history.items():
            volumes = []
            for symbol in universe_set:
                volume = volume_cache.get((symbol, date_key), 0.0)
                if volume > 0:
                    volumes.append(volume)
            volumes_by_date[date_key] = volumes
        
        all_volumes = [v for volumes in volumes_by_date.values() for v in volumes]
        
        # Calculate symbol longevity
        symbol_membership_periods: Dict[str, List[Tuple[date, date]]] = {}
        sorted_dates = sorted(universe_history.keys())
        
        universe_symbols = set.union(*universe_history.values()) if universe_history else set()
        
        for symbol in universe_symbols:
            periods = []
            in_universe = False
            start_date = None
            
            for date_key in sorted_dates:
                is_in = symbol in universe_history[date_key]
                if is_in and not in_universe:
                    # Entered
                    start_date = date_key
                    in_universe = True
                elif not is_in and in_universe:
                    # Exited
                    periods.append((start_date, date_key))
                    in_universe = False
                    start_date = None
            
            if in_universe:
                periods.append((start_date, sorted_dates[-1]))
            
            symbol_membership_periods[symbol] = periods
        
        time_in_universe_days = []
        for periods in symbol_membership_periods.values():
            for start, end in periods:
                days = (end - start).days
                time_in_universe_days.append(days)
        
        # Calculate additions/removals
        additions_per_rebalance = []
        removals_per_rebalance = []
        
        for i in range(1, len(sorted_dates)):
            prev_universe = universe_history[sorted_dates[i-1]]
            curr_universe = universe_history[sorted_dates[i]]
            
            additions = len(curr_universe - prev_universe)
            removals = len(prev_universe - curr_universe)
            
            additions_per_rebalance.append(additions)
            removals_per_rebalance.append(removals)
        
        avg_universe_size = np.mean(universe_sizes) if universe_sizes else 0
        median_universe_size = np.median(universe_sizes) if universe_sizes else 0
        avg_universe_size_for_turnover = avg_universe_size if avg_universe_size > 0 else 1
        total_additions = sum(additions_per_rebalance) if additions_per_rebalance else 0
        total_removals = sum(removals_per_rebalance) if removals_per_rebalance else 0
        total_changes = total_additions + total_removals
        num_rebalances = len(sorted_dates) - 1 if len(sorted_dates) > 1 else 1
        
        return {
            'avg_universe_size': avg_universe_size,
            'median_universe_size': median_universe_size,
            'min_universe_size': min(universe_sizes) if universe_sizes else 0,
            'max_universe_size': max(universe_sizes) if universe_sizes else 0,
            'universe_size_std': np.std(universe_sizes) if universe_sizes else 0.0,
            'avg_volume_24h': np.mean(all_volumes) if all_volumes else 0.0,
            'median_volume_24h': np.median(all_volumes) if all_volumes else 0.0,
            'min_volume_24h': min(all_volumes) if all_volumes else 0.0,
            'p25_volume_24h': np.percentile(all_volumes, 25) if all_volumes else 0.0,
            'p75_volume_24h': np.percentile(all_volumes, 75) if all_volumes else 0.0,
            'avg_additions_per_rebalance': np.mean(additions_per_rebalance) if additions_per_rebalance else 0.0,
            'avg_removals_per_rebalance': np.mean(removals_per_rebalance) if removals_per_rebalance else 0.0,
            'total_additions': total_additions,
            'total_removals': total_removals,
            'universe_turnover_rate': (total_changes / avg_universe_size_for_turnover) / num_rebalances * 100 if num_rebalances > 0 else 0.0,
            'avg_time_in_universe_days': np.mean(time_in_universe_days) if time_in_universe_days else 0.0,
            'pct_symbols_stayed_entire_period': 100.0 * len([p for p in symbol_membership_periods.values() if len(p) == 1 and (p[0][1] - p[0][0]).days >= (sorted_dates[-1] - sorted_dates[0]).days - 1]) / len(symbol_membership_periods) if symbol_membership_periods else 0.0,
            'unique_symbols_traded': len(symbol_membership_periods),
        }
    
    def _calculate_strategy_metrics(
        self,
        strategy_results: Dict,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Calculate strategy performance metrics from backtest results."""
        if 'error' in strategy_results:
            return {}
        
        equity_history = strategy_results.get('equity_history', [])
        trades = strategy_results.get('trades', [])
        
        if not equity_history:
            return {}
        
        initial_capital = strategy_results.get('initial_capital', 10000.0)
        final_equity = strategy_results.get('final_equity', initial_capital)
        
        # Calculate returns
        equity_series = pd.Series(equity_history)
        returns = equity_series.pct_change().dropna()
        
        total_return_pct = strategy_results.get('total_return_pct', 0.0)
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        
        # Sharpe ratio (annualized)
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24)  # For hourly data
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = 0.0
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(365 * 24)
        
        # Max drawdown
        max_drawdown_pct = strategy_results.get('max_drawdown_pct', 0.0)
        
        # Calmar ratio
        calmar_ratio = annualized_return_pct / abs(max_drawdown_pct) if max_drawdown_pct < 0 else 0.0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        
        gross_profit = sum([t['pnl'] for t in winning_trades])
        gross_loss = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Estimated fees
        estimated_fees = sum([
            abs(t.get('size', 0)) * t.get('exit_price', 0) * 0.00055  # 0.055% taker fee
            for t in trades
        ])
        
        # PnL concentration
        symbol_pnl = {}
        for trade in trades:
            symbol = trade.get('symbol', '')
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = 0.0
            symbol_pnl[symbol] += trade.get('pnl', 0.0)
        
        sorted_symbol_pnl = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
        total_pnl = sum([pnl for _, pnl in symbol_pnl.items()])
        
        pct_pnl_from_top5 = abs(sum([pnl for _, pnl in sorted_symbol_pnl[:5]]) / total_pnl * 100) if total_pnl != 0 else 0.0
        pct_pnl_from_top10 = abs(sum([pnl for _, pnl in sorted_symbol_pnl[:10]]) / total_pnl * 100) if total_pnl != 0 else 0.0
        
        return {
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'estimated_fees': estimated_fees,
            'pct_pnl_from_top5': pct_pnl_from_top5,
            'pct_pnl_from_top10': pct_pnl_from_top10,
            'unique_symbols_traded': len(symbol_pnl),
        }
    
    def evaluate_parameter_set(
        self,
        params: Dict[str, Any],
        symbol_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        timeframe: str
    ) -> UniverseOptimizationResult:
        """
        Evaluate a single parameter set.
        
        Args:
            params: Parameter dictionary
            symbol_data: OHLCV data for all symbols
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Trading timeframe
        
        Returns:
            UniverseOptimizationResult
        """
        # Create universe config from params
        universe_config = UniverseConfig(**params)

        total_days = (end_date - start_date).days

        # If we have enough history, split into in-sample and out-of-sample
        if total_days >= 60:
            split_days = int(total_days * 0.7)
            split_date = start_date + timedelta(days=split_days)

            # In-sample backtest
            is_strategy_results, is_universe_metrics = self.run_universe_backtest(
                universe_config,
                symbol_data,
                start_date,
                split_date,
                timeframe,
            )
            is_metrics = self._calculate_strategy_metrics(
                is_strategy_results, start_date, split_date
            )

            # Out-of-sample backtest
            oos_strategy_results, oos_universe_metrics = self.run_universe_backtest(
                universe_config,
                symbol_data,
                split_date,
                end_date,
                timeframe,
            )
            oos_metrics = self._calculate_strategy_metrics(
                oos_strategy_results, split_date, end_date
            )

            # Use OOS metrics as primary performance numbers
            strategy_metrics = oos_metrics or is_metrics or {}
            universe_metrics = oos_universe_metrics or is_universe_metrics or {}

            performance_by_regime = {
                "is_sharpe": is_metrics.get("sharpe_ratio", 0.0) if is_metrics else 0.0,
                "oos_sharpe": oos_metrics.get("sharpe_ratio", 0.0) if oos_metrics else 0.0,
                "is_max_drawdown_pct": is_metrics.get("max_drawdown_pct", 0.0)
                if is_metrics
                else 0.0,
                "oos_max_drawdown_pct": oos_metrics.get("max_drawdown_pct", 0.0)
                if oos_metrics
                else 0.0,
                "is_total_trades": is_metrics.get("total_trades", 0)
                if is_metrics
                else 0,
                "oos_total_trades": oos_metrics.get("total_trades", 0)
                if oos_metrics
                else 0,
            }

            # Simple regime consistency score based on Sharpe similarity
            is_sh = performance_by_regime["is_sharpe"]
            oos_sh = performance_by_regime["oos_sharpe"]
            regime_consistency_score = max(0.0, 1.0 - abs(is_sh - oos_sh))
        else:
            # Fallback: single-period backtest if not enough history
            strategy_results, universe_metrics = self.run_universe_backtest(
                universe_config,
                symbol_data,
                start_date,
                end_date,
                timeframe,
            )
            strategy_metrics = self._calculate_strategy_metrics(
                strategy_results, start_date, end_date
            )
            performance_by_regime = {}
            regime_consistency_score = 0.0

        # Combine into result
        result = UniverseOptimizationResult(
            params=params,
            backtest_start_date=start_date.isoformat(),
            backtest_end_date=end_date.isoformat(),
            timestamp=datetime.now().isoformat(),
            **strategy_metrics,
            **universe_metrics,
            pct_pnl_from_top5=strategy_metrics.get("pct_pnl_from_top5", 0.0)
            if strategy_metrics
            else 0.0,
            pct_pnl_from_top10=strategy_metrics.get("pct_pnl_from_top10", 0.0)
            if strategy_metrics
            else 0.0,
            unique_symbols_traded=strategy_metrics.get("unique_symbols_traded", 0)
            if strategy_metrics
            else 0,
            performance_by_regime=performance_by_regime,
            regime_consistency_score=regime_consistency_score,
            sensitivity_score=0.0,
        )

        return result
    
    def calculate_composite_score(self, result: UniverseOptimizationResult) -> float:
        """Calculate composite score for ranking."""
        # Prefer OOS Sharpe/drawdown if available
        oos_sharpe = (
            (result.performance_by_regime or {}).get("oos_sharpe", result.sharpe_ratio)
        )
        oos_dd = (
            (result.performance_by_regime or {}).get(
                "oos_max_drawdown_pct", result.max_drawdown_pct
            )
        )

        # Normalize metrics to [0, 1]
        sharpe_normalized = max(0, min(1, (oos_sharpe + 2) / 4))  # Assume range [-2, 2]
        return_normalized = max(
            0, min(1, (result.annualized_return_pct + 50) / 100)
        )  # Assume range [-50%, 50%]
        dd_normalized = max(
            0, min(1, (oos_dd + 50) / 50)
        )  # Assume range [-50%, 0%], invert

        # Composite: 0.4 * Sharpe + 0.3 * Return - 0.3 * Drawdown (higher is better)
        composite = (
            0.4 * sharpe_normalized + 0.3 * return_normalized + 0.3 * dd_normalized
        )

        return composite
    
    def check_constraints(self, result: UniverseOptimizationResult) -> Tuple[bool, List[str]]:
        """Check if result passes all constraints."""
        failures = []

        cfg = self.base_config.universe_optimizer

        if result.avg_universe_size < cfg.min_avg_universe_size:
            failures.append(
                f"avg_universe_size < {cfg.min_avg_universe_size}"
            )
        if result.avg_universe_size > cfg.max_avg_universe_size:
            failures.append(
                f"avg_universe_size > {cfg.max_avg_universe_size}"
            )
        if result.universe_turnover_rate > cfg.max_universe_turnover_pct:
            failures.append(
                f"universe_turnover_rate > {cfg.max_universe_turnover_pct}%"
            )
        if result.max_drawdown_pct < cfg.max_drawdown_pct:
            failures.append(
                f"max_drawdown_pct < {cfg.max_drawdown_pct}%"
            )
        if result.total_trades < cfg.min_total_trades:
            failures.append(
                f"total_trades < {cfg.min_total_trades}"
            )
        if result.win_rate < cfg.min_win_rate:
            failures.append(
                f"win_rate < {cfg.min_win_rate*100:.0f}%"
            )

        return len(failures) == 0, failures
    
    def optimize(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        timeframe: str,
        n_combinations: int = 200,
        method: str = "random"
    ) -> List[UniverseOptimizationResult]:
        """
        Run universe parameter optimization.
        
        Args:
            symbol_data: OHLCV data for all symbols
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Trading timeframe
            n_combinations: Number of parameter sets to test
            method: 'random' or 'grid'
        
        Returns:
            List of UniverseOptimizationResult, sorted by composite score
        """
        self.logger.info(f"Starting universe optimization: {n_combinations} combinations, {method} method")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(n_combinations, method)
        
        results = []
        
        for i, params in enumerate(param_combinations):
            try:
                self.logger.info(f"Testing parameter set {i+1}/{len(param_combinations)}")
                
                result = self.evaluate_parameter_set(
                    params,
                    symbol_data,
                    start_date,
                    end_date,
                    timeframe
                )
                
                # Calculate composite score
                result.composite_score = self.calculate_composite_score(result)
                
                # Check constraints
                passes, failures = self.check_constraints(result)
                if not passes:
                    self.logger.debug(f"Parameter set {i+1} failed constraints: {failures}")
                    continue
                
                results.append(result)
                
                # Save to database
                self._save_result(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i+1}/{len(param_combinations)} tests, {len(results)} passed constraints")
            
            except Exception as e:
                self.logger.error(f"Error testing parameter set {i+1}: {e}", exc_info=True)
                continue
        
        # Sort by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.logger.info(f"Optimization complete: {len(results)} parameter sets passed constraints")
        
        return results
    
    def _save_result(self, result: UniverseOptimizationResult):
        """Save result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp_int = int(datetime.now().timestamp())
        
        cursor.execute("""
            INSERT INTO universe_optimization_runs
            (timestamp, params, strategy_performance, universe_quality, robustness, metadata, composite_score, backtest_start_date, backtest_end_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp_int,
            json.dumps(result.params),
            json.dumps({
                'total_return_pct': result.total_return_pct,
                'annualized_return_pct': result.annualized_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
            }),
            json.dumps({
                'avg_universe_size': result.avg_universe_size,
                'universe_turnover_rate': result.universe_turnover_rate,
                'avg_volume_24h': result.avg_volume_24h,
            }),
            json.dumps({
                'regime_consistency_score': result.regime_consistency_score,
                'sensitivity_score': result.sensitivity_score,
            }),
            json.dumps({
                'backtest_start_date': result.backtest_start_date,
                'backtest_end_date': result.backtest_end_date,
                'timestamp': result.timestamp,
            }),
            result.composite_score,
            result.backtest_start_date,
            result.backtest_end_date,
        ))
        
        conn.commit()
        conn.close()
    
    def select_best_configs(
        self,
        results: List[UniverseOptimizationResult],
        n_top: int = 5
    ) -> List[UniverseOptimizationResult]:
        """
        Select best parameter configurations.
        
        Args:
            results: List of optimization results (sorted by composite score)
            n_top: Number of top results to return
        
        Returns:
            List of top N results
        """
        # Already sorted by composite score
        top_results = results[:n_top]
        
        # Add labels
        labels = ['Primary Recommended', 'Conservative Alternative', 'Aggressive Alternative', 'Alternative 4', 'Alternative 5']
        for i, result in enumerate(top_results):
            if i < len(labels):
                result.params['_label'] = labels[i]
        
        return top_results
    
    def results_to_config_yaml(self, result: UniverseOptimizationResult) -> str:
        """Convert result to config.yaml format."""
        params = result.params.copy()
        
        # Remove internal labels
        params.pop('_label', None)
        
        yaml_lines = ["universe:"]
        for key, value in params.items():
            if isinstance(value, list):
                yaml_lines.append(f"  {key}: {value}")
            elif isinstance(value, str):
                yaml_lines.append(f"  {key}: \"{value}\"")
            elif value is None:
                yaml_lines.append(f"  {key}: null")
            else:
                yaml_lines.append(f"  {key}: {value}")
        
        return "\n".join(yaml_lines)

