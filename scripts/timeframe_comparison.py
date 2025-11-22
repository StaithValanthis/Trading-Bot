#!/usr/bin/env python3
"""
Compare backtest performance across multiple timeframes.

This script runs backtests for different timeframes using time-normalized parameters
to ensure fair comparison. Results are compiled into a comparison report.

Usage:
    python scripts/timeframe_comparison.py --config config.yaml --start 2022-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BotConfig
from src.data.ohlcv_store import OHLCVStore
from src.backtest.backtester import Backtester, parse_timeframe_to_hours
from src.logging_utils import setup_logging, get_logger

# Candidate timeframes to test
CANDIDATE_TIMEFRAMES = ['1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Base parameters (current 4h defaults)
BASE_PARAMS_4H = {
    'ma_short': 5,
    'ma_long': 25,
    'momentum_lookback': 6,
    'atr_period': 4,
    'ranking_window': 18,
}

def scale_parameters_for_timeframe(
    base_params: Dict,
    base_tf_hours: float,
    new_tf_hours: float
) -> Dict:
    """
    Scale parameters to maintain equivalent time windows.
    
    Args:
        base_params: Parameters for base timeframe
        base_tf_hours: Hours per bar for base timeframe
        new_tf_hours: Hours per bar for new timeframe
    
    Returns:
        Scaled parameters
    """
    scale_factor = base_tf_hours / new_tf_hours
    scaled = {}
    
    # Parameters that should be scaled (bar-based)
    bar_params = ['ma_short', 'ma_long', 'momentum_lookback', 'atr_period', 'ranking_window']
    
    for key, value in base_params.items():
        if key in bar_params and isinstance(value, int):
            # Scale and round to nearest integer, minimum 1
            scaled_value = max(1, int(round(value * scale_factor)))
            scaled[key] = scaled_value
        else:
            # Keep as-is (multipliers, thresholds, etc.)
            scaled[key] = value
    
    return scaled

def update_config_for_timeframe(
    config: BotConfig,
    timeframe: str,
    scaled_params: Dict
) -> BotConfig:
    """
    Create a modified config for a different timeframe.
    
    Note: This creates a shallow copy and modifies in place.
    For production, consider creating a deep copy instead.
    """
    # Update timeframe
    config.exchange.timeframe = timeframe
    
    # Update trend parameters
    config.strategy.trend.ma_short = scaled_params['ma_short']
    config.strategy.trend.ma_long = scaled_params['ma_long']
    config.strategy.trend.momentum_lookback = scaled_params['momentum_lookback']
    config.strategy.trend.atr_period = scaled_params['atr_period']
    
    # Update cross-sectional parameters
    config.strategy.cross_sectional.ranking_window = scaled_params['ranking_window']
    
    # Adjust rebalance frequency based on timeframe
    # Keep 8h rebalance for most timeframes, but adjust for very long ones
    new_tf_hours = parse_timeframe_to_hours(timeframe)
    if new_tf_hours >= 24:
        # For 1d, rebalance daily (24h)
        config.strategy.cross_sectional.rebalance_frequency_hours = 24
    elif new_tf_hours >= 12:
        # For 12h, rebalance every 12h or 24h
        config.strategy.cross_sectional.rebalance_frequency_hours = 24
    elif new_tf_hours >= 8:
        # For 8h, rebalance every 8h makes sense
        config.strategy.cross_sectional.rebalance_frequency_hours = 8
    else:
        # For shorter timeframes, keep 8h rebalance
        config.strategy.cross_sectional.rebalance_frequency_hours = 8
    
    return config

def load_symbol_data(
    store: OHLCVStore,
    symbols: List[str],
    timeframe: str,
    start_date: date,
    end_date: date
) -> Dict[str, any]:
    """
    Load OHLCV data for symbols in date range.
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    symbol_data = {}
    
    # Convert dates to timestamps
    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)
    
    for symbol in symbols:
        try:
            df = store.get_ohlcv(symbol, timeframe, since=start_ts)
            if df.empty:
                logger.warning(f"No data for {symbol} at {timeframe}")
                continue
            
            # Filter to date range
            df = df[df.index >= datetime.combine(start_date, datetime.min.time())]
            df = df[df.index <= datetime.combine(end_date, datetime.max.time())]
            
            if len(df) < 100:  # Need minimum bars for meaningful backtest
                logger.warning(
                    f"Insufficient data for {symbol} at {timeframe}: "
                    f"{len(df)} bars (need at least 100)"
                )
                continue
            
            symbol_data[symbol] = df
            logger.debug(f"Loaded {len(df)} bars for {symbol} at {timeframe}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol} at {timeframe}: {e}")
            continue
    
    return symbol_data

def run_timeframe_backtest(
    config_path: str,
    timeframe: str,
    start_date: date,
    end_date: date,
    symbols: List[str],
    use_time_normalized: bool = True
) -> Dict:
    """
    Run backtest for a single timeframe.
    
    Args:
        config_path: Path to config file
        timeframe: Timeframe to test (e.g., '1h', '4h')
        start_date: Start date for backtest
        end_date: End date for backtest
        symbols: List of symbols to test
        use_time_normalized: If True, scale parameters to maintain time windows
    
    Returns:
        Backtest results dictionary
    """
    logger.info(f"Running backtest for {timeframe}...")
    
    # Load base config
    config = BotConfig.from_yaml(config_path)
    
    # Scale parameters if requested
    if use_time_normalized:
        base_tf_hours = 4.0  # Current default is 4h
        new_tf_hours = parse_timeframe_to_hours(timeframe)
        
        scaled_params = scale_parameters_for_timeframe(
            BASE_PARAMS_4H,
            base_tf_hours,
            new_tf_hours
        )
        
        logger.info(
            f"Scaled parameters for {timeframe}: "
            f"ma_short={scaled_params['ma_short']}, "
            f"ma_long={scaled_params['ma_long']}, "
            f"ranking_window={scaled_params['ranking_window']}"
        )
        
        config = update_config_for_timeframe(config, timeframe, scaled_params)
    else:
        # Use native parameters (same bar counts regardless of timeframe)
        config.exchange.timeframe = timeframe
    
    # Load data
    store = OHLCVStore(config.data.db_path)
    symbol_data = load_symbol_data(store, symbols, timeframe, start_date, end_date)
    
    if not symbol_data:
        logger.error(f"No data available for any symbols at {timeframe}")
        return {'error': 'no_data'}
    
    logger.info(f"Loaded data for {len(symbol_data)} symbols at {timeframe}")
    
    # Run backtest
    backtester = Backtester(config)
    
    try:
        results = backtester.backtest(
            symbol_data=symbol_data,
            initial_capital=10000.0,
            taker_fee=0.00055,  # Bybit taker fee
            stop_slippage_bps=10.0,
            tp_slippage_bps=5.0
        )
        
        # Add timeframe and parameters to results
        results['timeframe'] = timeframe
        results['parameters'] = {
            'ma_short': config.strategy.trend.ma_short,
            'ma_long': config.strategy.trend.ma_long,
            'momentum_lookback': config.strategy.trend.momentum_lookback,
            'atr_period': config.strategy.trend.atr_period,
            'ranking_window': config.strategy.cross_sectional.ranking_window,
            'rebalance_frequency_hours': config.strategy.cross_sectional.rebalance_frequency_hours,
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest for {timeframe}: {e}", exc_info=True)
        return {'error': str(e)}

def generate_comparison_report(results_by_timeframe: Dict[str, Dict]) -> str:
    """Generate a formatted comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("TIMEFRAME COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary table
    lines.append("Performance Summary:")
    lines.append("-" * 80)
    lines.append(
        f"{'Timeframe':<12} {'Ann. Return':>12} {'Sharpe':>8} {'Sortino':>8} "
        f"{'Max DD %':>10} {'Profit Factor':>12} {'Trades/Day':>12}"
    )
    lines.append("-" * 80)
    
    for tf in CANDIDATE_TIMEFRAMES:
        if tf not in results_by_timeframe:
            continue
        
        res = results_by_timeframe[tf]
        
        if 'error' in res:
            lines.append(f"{tf:<12} {'ERROR':>12}")
            continue
        
        ann_return = res.get('annualized_return', 0) * 100
        sharpe = res.get('sharpe_ratio', 0)
        sortino = res.get('sortino_ratio', 0)
        max_dd = res.get('max_drawdown_pct', 0)
        profit_factor = res.get('profit_factor', 0)
        trades_per_day = res.get('trades_per_day', 0)
        
        lines.append(
            f"{tf:<12} {ann_return:>11.2f}% {sharpe:>8.2f} {sortino:>8.2f} "
            f"{max_dd:>9.2f}% {profit_factor:>12.2f} {trades_per_day:>12.2f}"
        )
    
    lines.append("")
    lines.append("=" * 80)
    
    # Detailed results per timeframe
    for tf in CANDIDATE_TIMEFRAMES:
        if tf not in results_by_timeframe:
            continue
        
        res = results_by_timeframe[tf]
        
        if 'error' in res:
            lines.append(f"\n{tf}: ERROR - {res.get('error')}")
            continue
        
        lines.append(f"\nDetailed Results for {tf}:")
        lines.append("-" * 80)
        lines.append(f"  Parameters: {res.get('parameters', {})}")
        lines.append(f"  Total Return: {res.get('total_return', 0)*100:.2f}%")
        lines.append(f"  Annualized Return: {res.get('annualized_return', 0)*100:.2f}%")
        lines.append(f"  Sharpe Ratio: {res.get('sharpe_ratio', 0):.2f}")
        lines.append(f"  Sortino Ratio: {res.get('sortino_ratio', 0):.2f}")
        lines.append(f"  Max Drawdown: {res.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"  Profit Factor: {res.get('profit_factor', 0):.2f}")
        lines.append(f"  Win Rate: {res.get('win_rate', 0)*100:.2f}%")
        lines.append(f"  Total Trades: {res.get('total_trades', 0)}")
        lines.append(f"  Trades/Day: {res.get('trades_per_day', 0):.2f}")
        lines.append(f"  Total Fees: ${res.get('total_fees', 0):.2f}")
        lines.append(f"  Average Leverage: {res.get('avg_leverage', 0):.2f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description='Compare backtest performance across multiple timeframes'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'],
        help='Symbols to test (default: BTCUSDT ETHUSDT SOLUSDT BNBUSDT)'
    )
    parser.add_argument(
        '--output',
        help='Output file for results JSON (optional)'
    )
    parser.add_argument(
        '--report',
        help='Output file for text report (optional)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=CANDIDATE_TIMEFRAMES,
        help=f'Timeframes to test (default: {CANDIDATE_TIMEFRAMES})'
    )
    parser.add_argument(
        '--no-time-normalized',
        action='store_true',
        help='Use native bar counts instead of time-normalized parameters'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir='logs', level='INFO', service_name='timeframe_comparison')
    global logger
    logger = get_logger(__name__)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD")
        sys.exit(1)
    
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
    
    logger.info(f"Starting timeframe comparison")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Timeframes: {args.timeframes}")
    logger.info(f"  Time-normalized: {not args.no_time_normalized}")
    
    # Run backtests for each timeframe
    results_by_timeframe = {}
    
    for tf in args.timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing timeframe: {tf}")
        logger.info(f"{'='*60}")
        
        results = run_timeframe_backtest(
            args.config,
            tf,
            start_date,
            end_date,
            args.symbols,
            use_time_normalized=not args.no_time_normalized
        )
        
        results_by_timeframe[tf] = results
        
        if 'error' not in results:
            logger.info(
                f"✓ {tf} completed: "
                f"Sharpe={results.get('sharpe_ratio', 0):.2f}, "
                f"Return={results.get('annualized_return', 0)*100:.1f}%, "
                f"Max DD={results.get('max_drawdown_pct', 0):.1f}%"
            )
        else:
            logger.error(f"✗ {tf} failed: {results.get('error')}")
    
    # Generate report
    report = generate_comparison_report(results_by_timeframe)
    print("\n" + report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results_by_timeframe, f, indent=2, default=str)
        logger.info(f"\nResults saved to {args.output}")
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    
    logger.info("\n" + "="*60)
    logger.info("TIMEFRAME COMPARISON COMPLETE")
    logger.info("="*60)

if __name__ == '__main__':
    main()

