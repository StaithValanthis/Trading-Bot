#!/usr/bin/env python3
"""
One-shot script to: download data, optimize parameters per timeframe, then compare timeframes.

This script orchestrates the complete timeframe analysis workflow:
1. Downloads historical OHLCV data (if needed or forced)
2. Optimizes parameters for each timeframe separately
3. Runs timeframe comparison using optimized parameters for fair comparison

Usage:
    python scripts/optimize_and_compare_timeframes.py --config config.yaml --start 2022-01-01 --end 2024-12-31
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import json
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BotConfig
from src.data.ohlcv_store import OHLCVStore
from src.exchange.bybit_client import BybitClient
from src.data.downloader import DataDownloader
from src.optimizer.optimizer import Optimizer
from src.optimizer.timeframe_analyzer import TimeframeAnalyzer
from src.universe.store import UniverseStore
from src.universe.selector import UniverseSelector
from src.logging_utils import setup_logging, get_logger

# Candidate timeframes to test
CANDIDATE_TIMEFRAMES = ['1h', '2h', '4h', '6h', '12h', '1d']

logger = None


def download_data_for_timeframes(
    config: BotConfig,
    symbols: List[str],
    timeframes: List[str],
    lookback_days: int = 730,
    force_download: bool = False
) -> Dict[str, List[str]]:
    """
    Download historical data for symbols and timeframes.
    
    Args:
        config: Bot configuration
        symbols: List of symbols to download
        timeframes: List of timeframes to download
        lookback_days: Days of history to download
        force_download: If True, re-download even if data exists
    
    Returns:
        Dictionary mapping timeframe to list of symbols with data
    """
    logger.info(f"Downloading historical data for {len(symbols)} symbols, {len(timeframes)} timeframes")
    
    # Initialize exchange client and data store
    exchange = BybitClient(config.exchange)
    store = OHLCVStore(config.data.db_path)
    downloader = DataDownloader(exchange, store, logger=logger)
    
    results = {tf: [] for tf in timeframes}
    
    for timeframe in timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading data for timeframe: {timeframe}")
        logger.info(f"{'='*60}")
        
        for symbol in symbols:
            try:
                logger.info(f"  Downloading {symbol} {timeframe}...")
                
                # Check if data already exists
                if not force_download:
                    existing_data = store.get_ohlcv(symbol, timeframe)
                    if not existing_data.empty:
                        # Check if we have enough recent data
                        latest_timestamp = existing_data.index[-1] if len(existing_data) > 0 else None
                        if latest_timestamp:
                            days_old = (datetime.now() - latest_timestamp).days
                            if days_old < 1:  # Data is less than 1 day old
                                logger.info(f"    ✓ {symbol} {timeframe}: Data already exists ({len(existing_data)} bars)")
                                results[timeframe].append(symbol)
                                continue
                
                # Download data
                downloader.download_and_store(symbol, timeframe, lookback_days=lookback_days)
                
                # Verify data was stored
                data = store.get_ohlcv(symbol, timeframe)
                if not data.empty and len(data) >= 100:
                    logger.info(f"    ✓ {symbol} {timeframe}: Downloaded {len(data)} bars")
                    results[timeframe].append(symbol)
                else:
                    logger.warning(f"    ✗ {symbol} {timeframe}: Insufficient data after download")
                    
            except Exception as e:
                logger.error(f"    ✗ {symbol} {timeframe}: Error - {e}")
                continue
    
    return results


def optimize_timeframe_parameters(
    config: BotConfig,
    symbols: List[str],
    timeframe: str,
    store: OHLCVStore
) -> Optional[Dict]:
    """
    Optimize parameters for a specific timeframe.
    
    Args:
        config: Bot configuration
        symbols: List of symbols to optimize on
        timeframe: Timeframe to optimize (e.g., '4h')
        store: OHLCV data store
    
    Returns:
        Dictionary with best parameters and metrics, or None if optimization failed
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing parameters for timeframe: {timeframe}")
    logger.info(f"{'='*60}")
    
    # Ensure ranking_window is in optimizer param_ranges if not already present
    opt_config = copy.deepcopy(config)
    opt_config.exchange.timeframe = timeframe
    
    # Add ranking_window to param_ranges if not present
    if 'ranking_window' not in opt_config.optimizer.param_ranges:
        logger.info("Adding ranking_window to optimizer parameter ranges")
        # Reasonable range: 12-48 bars (timeframe-dependent)
        # These will scale appropriately for different timeframes
        opt_config.optimizer.param_ranges['ranking_window'] = [12, 18, 24, 36, 48]
    
    # Create optimizer instance with updated config
    optimizer = Optimizer(opt_config, store)
    
    # Run optimization
    try:
        result = optimizer.optimize(symbols=symbols, timeframe=timeframe)
        
        if 'error' in result:
            logger.error(f"  ✗ Optimization failed: {result['error']}")
            return None
        
        best_params = result.get('best_params', {})
        best_metrics = result.get('best_metrics', {})
        
        logger.info(f"  ✓ Optimization complete:")
        logger.info(f"    Best parameters: {best_params}")
        logger.info(f"    Sharpe: {best_metrics.get('avg_sharpe', 0):.2f}")
        logger.info(f"    Return: {best_metrics.get('avg_return_pct', 0):.2f}%")
        logger.info(f"    Drawdown: {best_metrics.get('avg_drawdown_pct', 0):.2f}%")
        
        # Store timeframe in result for reference
        result['timeframe'] = timeframe
        return result
        
    except Exception as e:
        logger.error(f"  ✗ Optimization error: {e}", exc_info=True)
        return None


def apply_optimized_params_to_config(
    config: BotConfig,
    timeframe: str,
    optimized_params: Dict
) -> BotConfig:
    """
    Apply optimized parameters to config for a specific timeframe.
    
    Args:
        config: Base configuration
        timeframe: Timeframe string
        optimized_params: Dictionary of optimized parameters
    
    Returns:
        Modified config with optimized parameters
    """
    # Create deep copy to avoid modifying original
    tf_config = copy.deepcopy(config)
    tf_config.exchange.timeframe = timeframe
    
    # Apply optimized trend parameters
    if 'ma_short' in optimized_params:
        tf_config.strategy.trend.ma_short = optimized_params['ma_short']
    if 'ma_long' in optimized_params:
        tf_config.strategy.trend.ma_long = optimized_params['ma_long']
    if 'momentum_lookback' in optimized_params:
        tf_config.strategy.trend.momentum_lookback = optimized_params['momentum_lookback']
    if 'atr_stop_multiplier' in optimized_params:
        tf_config.strategy.trend.atr_stop_multiplier = optimized_params['atr_stop_multiplier']
    
    # Apply optimized cross-sectional parameters
    if 'top_k' in optimized_params:
        tf_config.strategy.cross_sectional.top_k = optimized_params['top_k']
    if 'ranking_window' in optimized_params:
        tf_config.strategy.cross_sectional.ranking_window = optimized_params['ranking_window']
    
    return tf_config


def compare_timeframes_with_optimized_params(
    config: BotConfig,
    symbols: List[str],
    timeframes: List[str],
    optimized_params_by_tf: Dict[str, Dict],
    start_date: datetime,
    end_date: datetime,
    store: OHLCVStore
) -> List:
    """
    Compare timeframes using optimized parameters for each timeframe.
    
    Args:
        config: Base configuration
        symbols: List of symbols to test
        timeframes: List of timeframes to compare
        optimized_params_by_tf: Dictionary mapping timeframe to optimized parameters
        start_date: Start date for backtest
        end_date: End date for backtest
        store: OHLCV data store
    
    Returns:
        List of TimeframeResult objects
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing timeframes with optimized parameters")
    logger.info(f"{'='*60}")
    
    results = []
    
    for tf in timeframes:
        # Use optimized parameters if available, otherwise use defaults
        if tf not in optimized_params_by_tf:
            logger.warning(f"No optimized parameters for {tf}, using default parameters")
            best_params = {}  # Empty dict will use config defaults
        else:
            optimized_result = optimized_params_by_tf[tf]
            best_params = optimized_result.get('best_params', {})
        
        try:
            if best_params:
                logger.info(f"\nTesting timeframe: {tf} with optimized parameters")
            else:
                logger.info(f"\nTesting timeframe: {tf} with default parameters")
            
            # Create config with optimized parameters for this timeframe
            tf_config = apply_optimized_params_to_config(config, tf, best_params)
            
            # Load data for all symbols at this timeframe
            since_ms = int(start_date.timestamp() * 1000)
            symbol_data = {}
            
            for symbol in symbols:
                try:
                    df = store.get_ohlcv(symbol, tf, since=since_ms)
                    if not df.empty and len(df) > 100:
                        # Filter to date range
                        filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
                        if not filtered_df.empty and len(filtered_df) >= 100:
                            symbol_data[symbol] = filtered_df
                            logger.debug(f"  Loaded {len(filtered_df)} bars for {symbol}")
                except Exception as e:
                    logger.warning(f"  Error loading {tf} data for {symbol}: {e}")
                    continue
            
            if not symbol_data:
                logger.warning(f"  No data available for timeframe {tf}, skipping")
                continue
            
            # Find common timestamp range
            all_timestamps = set()
            for df in symbol_data.values():
                all_timestamps.update(df.index)
            common_timestamps = sorted(all_timestamps)
            
            if len(common_timestamps) < 100:
                logger.warning(f"  Insufficient common timestamps for {tf}, skipping")
                continue
            
            # Run backtest with optimized parameters
            from src.backtest.backtester import Backtester
            backtester = Backtester(tf_config)
            backtest_result = backtester.backtest(
                symbol_data=symbol_data,
                initial_capital=10000.0,
                taker_fee=0.00055,  # Bybit taker fee
                stop_slippage_bps=10.0,
                tp_slippage_bps=5.0
            )
            
            if not backtest_result or backtest_result.get('total_trades', 0) == 0:
                logger.warning(f"  No trades for timeframe {tf}, skipping")
                continue
            
            # Calculate metrics using TimeframeAnalyzer's method
            analyzer = TimeframeAnalyzer(tf_config, store)
            equity_history = backtest_result.get('equity_history', [])
            
            try:
                tf_result = analyzer._calculate_metrics(
                    backtest_result,
                    tf,
                    symbol_data,
                    common_timestamps,
                    is_oos_split=0.7,
                    classify_regimes=True,
                    equity_history=equity_history
                )
            except Exception as e:
                logger.error(f"  Error calculating metrics for {tf}: {e}", exc_info=True)
                continue
            
            # Add optimized parameters to result metadata
            tf_result.optimized_params = best_params
            
            results.append(tf_result)
            logger.info(
                f"  ✓ {tf} completed: Return={tf_result.total_return_pct:.2f}%, "
                f"Sharpe={tf_result.sharpe_ratio:.2f}, Trades={tf_result.total_trades}"
            )
            
        except Exception as e:
            logger.error(f"  ✗ Error testing timeframe {tf}: {e}", exc_info=True)
            continue
    
    # Sort by Sharpe ratio (descending)
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    
    return results


def generate_optimization_report(optimized_params_by_tf: Dict[str, Dict]) -> str:
    """Generate report of optimized parameters per timeframe."""
    lines = []
    lines.append("=" * 80)
    lines.append("OPTIMIZED PARAMETERS BY TIMEFRAME")
    lines.append("=" * 80)
    lines.append("")
    
    for tf in sorted(optimized_params_by_tf.keys()):
        opt_result = optimized_params_by_tf[tf]
        best_params = opt_result.get('best_params', {})
        best_metrics = opt_result.get('best_metrics', {})
        
        lines.append(f"{tf}:")
        lines.append(f"  Parameters: {best_params}")
        lines.append(f"  Sharpe: {best_metrics.get('avg_sharpe', 0):.2f}")
        lines.append(f"  Return: {best_metrics.get('avg_return_pct', 0):.2f}%")
        lines.append(f"  Drawdown: {best_metrics.get('avg_drawdown_pct', 0):.2f}%")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Download data, optimize parameters per timeframe, then compare timeframes'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,  # None means use universe or fetch all
        help='Symbols to test (default: use all symbols from universe if available, otherwise fetch all from exchange)'
    )
    parser.add_argument(
        '--use-universe',
        action='store_true',
        help='Use universe symbols from database (default: try universe first, fallback to exchange)'
    )
    parser.add_argument(
        '--fetch-all',
        action='store_true',
        help='Fetch all symbols from exchange (ignores universe)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=CANDIDATE_TIMEFRAMES,
        help=f'Timeframes to test (default: {CANDIDATE_TIMEFRAMES})'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step (use existing data)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data even if it exists'
    )
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip optimization step (use default parameters)'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=730,
        help='Days of history to download (default: 730 = ~2 years)'
    )
    parser.add_argument(
        '--output',
        help='Output file for results JSON (optional)'
    )
    parser.add_argument(
        '--report',
        help='Output file for text report (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir='logs', level='INFO', service_name='optimize_and_compare_timeframes')
    global logger
    logger = get_logger(__name__)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
    except ValueError as e:
        logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD")
        sys.exit(1)
    
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
    
    # Load config
    config = BotConfig.from_yaml(args.config)
    store = OHLCVStore(config.data.db_path)
    
    # Determine which symbols to use
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Using specified symbols: {symbols}")
    else:
        # No symbols specified, try to get from universe or exchange
        symbols = None
        
        if args.use_universe or not args.fetch_all:
            # Try to get symbols from universe first
            try:
                universe_store = UniverseStore(config.data.db_path)
                universe_symbols = universe_store.get_current_universe()
                
                if universe_symbols:
                    symbols = sorted(list(universe_symbols))
                    logger.info(f"Using {len(symbols)} symbols from universe: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
                else:
                    logger.info("No symbols found in universe store")
            except Exception as e:
                logger.warning(f"Could not load symbols from universe: {e}")
        
        # If still no symbols, fetch from exchange
        if not symbols or args.fetch_all:
            try:
                logger.info("Fetching all symbols from exchange...")
                exchange = BybitClient(config.exchange)
                universe_store = UniverseStore(config.data.db_path)
                universe_selector = UniverseSelector(
                    config.universe,
                    exchange,
                    store,
                    universe_store
                )
                all_symbols = universe_selector.fetch_all_symbols()
                symbols = sorted(all_symbols)
                logger.info(f"Fetched {len(symbols)} symbols from exchange: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
            except Exception as e:
                logger.error(f"Could not fetch symbols from exchange: {e}")
                logger.info("Falling back to default symbols: BTCUSDT ETHUSDT SOLUSDT BNBUSDT")
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
        
        if not symbols:
            logger.error("No symbols available. Cannot continue.")
            sys.exit(1)
    
    logger.info("="*80)
    logger.info("TIMEFRAME OPTIMIZATION AND COMPARISON")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Symbols: {len(symbols)} symbols {('(' + ', '.join(symbols[:5]) + ('...' if len(symbols) > 5 else '') + ')') if len(symbols) > 5 else symbols}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info("")
    
    # Step 1: Download data
    if not args.skip_download:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DOWNLOADING HISTORICAL DATA")
        logger.info("="*80)
        
        download_results = download_data_for_timeframes(
            config=config,
            symbols=symbols,
            timeframes=args.timeframes,
            lookback_days=args.lookback_days,
            force_download=args.force_download
        )
        
        # Filter timeframes to only those with sufficient data
        timeframes_with_data = [
            tf for tf in args.timeframes 
            if tf in download_results and len(download_results[tf]) > 0
        ]
        
        if not timeframes_with_data:
            logger.error("No data available for any timeframes. Cannot continue.")
            sys.exit(1)
        
        logger.info(f"\nData available for timeframes: {timeframes_with_data}")
    else:
        logger.info("Skipping data download (using existing data)")
        timeframes_with_data = args.timeframes
    
    # Step 2: Optimize parameters for each timeframe
    optimized_params_by_tf = {}
    
    if not args.skip_optimization:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: OPTIMIZING PARAMETERS PER TIMEFRAME")
        logger.info("="*80)
        
        for tf in timeframes_with_data:
            opt_result = optimize_timeframe_parameters(
                config=config,
                symbols=symbols,
                timeframe=tf,
                store=store
            )
            
            if opt_result:
                optimized_params_by_tf[tf] = opt_result
            else:
                logger.warning(f"Optimization failed for {tf}, will use default parameters")
        
        if optimized_params_by_tf:
            opt_report = generate_optimization_report(optimized_params_by_tf)
            logger.info("\n" + opt_report)
    else:
        logger.info("Skipping optimization (using default parameters)")
    
    # Step 3: Compare timeframes using optimized parameters
    logger.info("\n" + "="*80)
    logger.info("STEP 3: COMPARING TIMEFRAMES WITH OPTIMIZED PARAMETERS")
    logger.info("="*80)
    
    timeframe_results = compare_timeframes_with_optimized_params(
        config=config,
        symbols=symbols,
        timeframes=timeframes_with_data,
        optimized_params_by_tf=optimized_params_by_tf,
        start_date=start_datetime,
        end_date=end_datetime,
        store=store
    )
    
    if not timeframe_results:
        logger.error("No timeframe comparison results generated")
        sys.exit(1)
    
    # Convert results to dictionary format for reporting
    from scripts.timeframe_comparison import convert_timeframe_results_to_dict
    results_by_timeframe = convert_timeframe_results_to_dict(timeframe_results)
    
    # Generate comparison report
    from scripts.timeframe_comparison import generate_comparison_report
    report = generate_comparison_report(results_by_timeframe)
    
    # Add optimization info to report
    if optimized_params_by_tf:
        opt_report = generate_optimization_report(optimized_params_by_tf)
        full_report = opt_report + "\n\n" + report
    else:
        full_report = report
    
    print("\n" + full_report)
    
    # Save outputs
    if args.output:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else None
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'optimized_parameters': {
                tf: opt_result.get('best_params', {}) 
                for tf, opt_result in optimized_params_by_tf.items()
            },
            'comparison_results': results_by_timeframe,
            'metadata': {
                'start_date': args.start,
                'end_date': args.end,
                'symbols': symbols,
                'timeframes': args.timeframes,
                'symbols_source': 'specified' if args.symbols else ('universe' if not args.fetch_all else 'exchange')
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"\nResults saved to {args.output}")
    
    if args.report:
        report_dir = os.path.dirname(args.report) if os.path.dirname(args.report) else None
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        
        with open(args.report, 'w') as f:
            f.write(full_report)
        logger.info(f"Report saved to {args.report}")
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION AND COMPARISON COMPLETE")
    logger.info("="*80)
    
    # Print summary
    logger.info("\nBest performing timeframes:")
    for i, result in enumerate(timeframe_results[:5], 1):
        logger.info(
            f"  {i}. {result.timeframe}: Sharpe={result.sharpe_ratio:.2f}, "
            f"Return={result.total_return_pct:.2f}%, "
            f"Max DD={result.max_drawdown_pct:.2f}%"
        )


if __name__ == '__main__':
    main()

