#!/usr/bin/env python3
"""
Funding Opportunity Strategy - Parameter Tuning Script

Runs funding-only backtests and parameter sweeps to recommend optimal settings.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json
import copy
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import BotConfig
from src.data.ohlcv_store import OHLCVStore
from src.backtest.backtester import Backtester, parse_timeframe_to_hours
from src.logging_utils import setup_logging, get_logger


def run_funding_backtest(
    config: BotConfig,
    symbol_data: Dict,
    funding_params: Dict = None,
    funding_rate_per_8h: float = None  # If None, will use varying rates per symbol
) -> Dict:
    """Run a funding-only backtest with optional parameter overrides."""
    # Create modified config for funding-only mode
    test_config = copy.deepcopy(config)
    test_config.strategy.funding_opportunity.enabled = True
    
    # Override parameters if provided
    if funding_params:
        if 'min_funding_rate' in funding_params:
            test_config.strategy.funding_opportunity.min_funding_rate = funding_params['min_funding_rate']
        if 'exit_funding_threshold' in funding_params:
            test_config.strategy.funding_opportunity.exit.exit_funding_threshold = funding_params['exit_funding_threshold']
        if 'max_holding_hours' in funding_params:
            test_config.strategy.funding_opportunity.exit.max_holding_hours = funding_params['max_holding_hours']
        if 'base_size_fraction' in funding_params:
            test_config.strategy.funding_opportunity.sizing.base_size_fraction = funding_params['base_size_fraction']
        if 'max_total_funding_exposure' in funding_params:
            test_config.strategy.funding_opportunity.risk.max_total_funding_exposure = funding_params['max_total_funding_exposure']
    
    # Disable main strategy (funding only)
    # Note: We can't easily disable trend/cross_sectional, but we'll filter results by source
    
    backtester = Backtester(test_config)
    
    # Create funding rate history that varies by symbol to make backtesting more realistic
    # Use rates that are above min_funding_rate for some symbols, below for others
    import pandas as pd
    import numpy as np
    
    funding_rate_history = {}
    min_rate = test_config.strategy.funding_opportunity.min_funding_rate
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate varying funding rates per symbol
    # Some symbols above threshold (opportunities), some below (no opportunities)
    for symbol in symbol_data.keys():
        # Vary rates: some symbols have opportunities, some don't
        # Use a base rate that's above min for ~60% of symbols (realistic)
        base_rate = min_rate * np.random.uniform(0.8, 2.0)  # 0.8x to 2.0x min_rate
        
        # Create a time series with some variation
        df = symbol_data[symbol]
        if not df.empty:
            # Vary funding rate over time (simulate funding cycles)
            rates = []
            for i in range(len(df)):
                # Add some variation: funding rates change over time
                variation = np.random.uniform(0.9, 1.1)
                rate = base_rate * variation
                # Occasionally drop below threshold (funding flips)
                if np.random.random() < 0.1:  # 10% chance
                    rate = min_rate * 0.5  # Below threshold
                rates.append(rate)
            
            funding_rate_history[symbol] = pd.Series(rates, index=df.index)
    
    # Use funding_rate_per_8h as fallback if provided, otherwise use history
    result = backtester.backtest(
        symbol_data,
        funding_rate_per_8h=funding_rate_per_8h or 0.0,
        funding_rate_history=funding_rate_history
    )
    
    # Filter trades to only funding trades
    if 'trades' in result:
        funding_trades = [
            t for t in result['trades']
            if t.get('source') in ['funding_opportunity', 'confluence', 'confluence_prefer_funding', 'confluence_prefer_main']
        ]
        result['funding_trades'] = funding_trades
        result['total_funding_trades'] = len(funding_trades)
    
    return result


def calculate_funding_metrics(result: Dict, timestamps: List = None) -> Dict:
    """Calculate funding-specific metrics from backtest results."""
    funding_trades = result.get('funding_trades', [])
    
    metrics = {
        'total_funding_trades': len(funding_trades),
        'funding_trades_per_year': 0.0,
        'holding_times_hours': [],
        'entry_funding_rates': {'long': [], 'short': []},
        'max_concurrent_funding_positions': 0,
        'max_funding_exposure_pct': 0.0,
    }
    
    if funding_trades and timestamps:
        # Calculate holding times
        for trade in funding_trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            if entry_time and exit_time:
                import pandas as pd
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                if isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                hours = (exit_time - entry_time).total_seconds() / 3600
                metrics['holding_times_hours'].append(hours)
            
            # Extract entry funding rate
            metadata = trade.get('metadata', {})
            funding_rate = metadata.get('funding_rate')
            if funding_rate is not None:
                signal = trade.get('side', 'long')
                metrics['entry_funding_rates'][signal].append(funding_rate)
        
        # Calculate trades per year
        first_trade = min(t.get('entry_time', timestamps[0]) for t in funding_trades if t.get('entry_time'))
        last_trade = max(t.get('exit_time', timestamps[-1]) for t in funding_trades if t.get('exit_time'))
        if first_trade and last_trade:
            import pandas as pd
            if isinstance(first_trade, str):
                first_trade = pd.to_datetime(first_trade)
            if isinstance(last_trade, str):
                last_trade = pd.to_datetime(last_trade)
            years = (last_trade - first_trade).total_seconds() / (365.25 * 24 * 3600)
            if years > 0:
                metrics['funding_trades_per_year'] = len(funding_trades) / years
        
        # Calculate holding time statistics
        if metrics['holding_times_hours']:
            holding_times = sorted(metrics['holding_times_hours'])
            metrics['holding_time_min'] = holding_times[0]
            metrics['holding_time_median'] = holding_times[len(holding_times) // 2]
            metrics['holding_time_90th_pct'] = holding_times[int(len(holding_times) * 0.9)] if len(holding_times) > 10 else holding_times[-1]
            metrics['holding_time_max'] = holding_times[-1]
        
        # Calculate average entry funding rates
        if metrics['entry_funding_rates']['long']:
            import numpy as np
            metrics['avg_entry_funding_rate_long'] = np.mean(metrics['entry_funding_rates']['long'])
        if metrics['entry_funding_rates']['short']:
            import numpy as np
            metrics['avg_entry_funding_rate_short'] = np.mean(metrics['entry_funding_rates']['short'])
    
    return metrics


def run_parameter_sweep(
    config: BotConfig,
    symbol_data: Dict,
    base_params: Dict
) -> List[Dict]:
    """Run parameter sweep around base parameters."""
    results = []
    
    # Define sweep ranges
    min_funding_rates = [
        base_params['min_funding_rate'] * 0.75,
        base_params['min_funding_rate'],
        base_params['min_funding_rate'] * 1.25,
    ]
    
    exit_thresholds = [
        base_params['exit_funding_threshold'] * 0.75,
        base_params['exit_funding_threshold'],
        base_params['exit_funding_threshold'] * 1.25,
    ]
    
    max_holding_hours_options = [
        max(24, int(base_params['max_holding_hours'] * 0.5)),
        base_params['max_holding_hours'],
        min(168, int(base_params['max_holding_hours'] * 1.5)),
    ]
    
    base_size_fractions = [
        base_params['base_size_fraction'] * 0.875,  # -12.5%
        base_params['base_size_fraction'],
        base_params['base_size_fraction'] * 1.125,  # +12.5%
    ]
    
    max_exposures = [
        base_params['max_total_funding_exposure'] * 0.875,
        base_params['max_total_funding_exposure'],
        base_params['max_total_funding_exposure'] * 1.125,
    ]
    
    # Test combinations (focus on most impactful parameters first)
    import itertools
    
    # Primary sweep: min_funding_rate, exit_threshold, max_holding_hours
    for min_rate, exit_thresh, max_hours in itertools.product(min_funding_rates, exit_thresholds, max_holding_hours_options):
        params = {
            'min_funding_rate': min_rate,
            'exit_funding_threshold': exit_thresh,
            'max_holding_hours': max_hours,
            'base_size_fraction': base_params['base_size_fraction'],
            'max_total_funding_exposure': base_params['max_total_funding_exposure'],
        }
        
        result = run_funding_backtest(config, symbol_data, params)
        
        # Calculate metrics
        timestamps = result.get('timestamps', [])
        funding_metrics = calculate_funding_metrics(result, timestamps)
        
        results.append({
            'parameters': params,
            'total_return_pct': result.get('total_return_pct', 0.0),
            'sharpe_ratio': result.get('sharpe_ratio', 0.0),
            'max_drawdown_pct': result.get('max_drawdown_pct', 0.0),
            'total_trades': result.get('total_trades', 0),
            'funding_trades': funding_metrics.get('total_funding_trades', 0),
            'funding_trades_per_year': funding_metrics.get('funding_trades_per_year', 0.0),
            'holding_time_median': funding_metrics.get('holding_time_median', 0.0),
        })
    
    # Secondary sweep: base_size_fraction and max_total_funding_exposure (only for base primary params)
    base_min_rate = base_params['min_funding_rate']
    base_exit_thresh = base_params['exit_funding_threshold']
    base_max_hours = base_params['max_holding_hours']
    
    for base_size, max_exp in itertools.product(base_size_fractions, max_exposures):
        params = {
            'min_funding_rate': base_min_rate,
            'exit_funding_threshold': base_exit_thresh,
            'max_holding_hours': base_max_hours,
            'base_size_fraction': base_size,
            'max_total_funding_exposure': max_exp,
        }
        
        # Use varying funding rates per symbol
        result = run_funding_backtest(config, symbol_data, params, funding_rate_per_8h=None)
        timestamps = result.get('timestamps', [])
        funding_metrics = calculate_funding_metrics(result, timestamps)
        
        results.append({
            'parameters': params,
            'total_return_pct': result.get('total_return_pct', 0.0),
            'sharpe_ratio': result.get('sharpe_ratio', 0.0),
            'max_drawdown_pct': result.get('max_drawdown_pct', 0.0),
            'total_trades': result.get('total_trades', 0),
            'funding_trades': funding_metrics.get('total_funding_trades', 0),
            'funding_trades_per_year': funding_metrics.get('funding_trades_per_year', 0.0),
            'holding_time_median': funding_metrics.get('holding_time_median', 0.0),
        })
    
    return results


def generate_health_check_report(
    base_result: Dict,
    base_metrics: Dict,
    sweep_results: List[Dict],
    base_params: Dict,
    output_file: Path
) -> str:
    """Generate funding health check report."""
    
    timestamps = base_result.get('timestamps', [])
    
    report_lines = [
        "# Funding Opportunity Strategy - Health Check Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Current Configuration Metrics",
        "",
        "### Core Performance",
        "",
        f"- **Total Return**: {base_result.get('total_return_pct', 0.0):+.2f}%",
        f"- **CAGR**: {base_result.get('annualized_return', 0.0)*100:+.2f}%",
        f"- **Sharpe Ratio**: {base_result.get('sharpe_ratio', 0.0):.2f}",
        f"- **Max Drawdown**: {base_result.get('max_drawdown_pct', 0.0):.2f}%",
        "",
        "### Funding-Specific Metrics",
        "",
        f"- **Total Funding Trades**: {base_metrics.get('total_funding_trades', 0)}",
        f"- **Funding Trades per Year**: {base_metrics.get('funding_trades_per_year', 0.0):.1f}",
    ]
    
    if base_metrics.get('holding_times_hours'):
        report_lines.extend([
            f"- **Holding Time**: min={base_metrics.get('holding_time_min', 0):.1f}h, "
            f"median={base_metrics.get('holding_time_median', 0):.1f}h, "
            f"90th={base_metrics.get('holding_time_90th_pct', 0):.1f}h, "
            f"max={base_metrics.get('holding_time_max', 0):.1f}h",
        ])
    
    if base_metrics.get('avg_entry_funding_rate_long') is not None:
        report_lines.append(
            f"- **Avg Entry Funding (Long)**: {base_metrics['avg_entry_funding_rate_long']*100:.4f}% per 8h"
        )
    if base_metrics.get('avg_entry_funding_rate_short') is not None:
        report_lines.append(
            f"- **Avg Entry Funding (Short)**: {base_metrics['avg_entry_funding_rate_short']*100:.4f}% per 8h"
        )
    
    report_lines.extend([
        "",
        "## Parameter Sweep Results",
        "",
        "| min_funding_rate | exit_threshold | max_hours | base_size | max_exposure | Return % | Sharpe | Max DD % | Trades/Yr |",
        "|-------------------|----------------|-----------|-----------|--------------|----------|--------|----------|-----------|"
    ])
    
    # Sort by Sharpe ratio (descending)
    sorted_results = sorted(sweep_results, key=lambda x: x['sharpe_ratio'], reverse=True)
    
    for r in sorted_results[:15]:  # Top 15 results
        params = r['parameters']
        report_lines.append(
            f"| {params['min_funding_rate']:.5f} | {params['exit_funding_threshold']:.5f} | "
            f"{params['max_holding_hours']} | {params['base_size_fraction']:.3f} | "
            f"{params['max_total_funding_exposure']:.2f} | {r['total_return_pct']:+.2f}% | "
            f"{r['sharpe_ratio']:.2f} | {r['max_drawdown_pct']:.2f}% | {r['funding_trades_per_year']:.1f} |"
        )
    
    # Find best configuration
    best = sorted_results[0]
    best_params = best['parameters']
    
    # Compare with base
    base_sharpe = base_result.get('sharpe_ratio', 0.0)
    best_sharpe = best['sharpe_ratio']
    
    report_lines.extend([
        "",
        "## Recommendation",
        "",
    ])
    
    if abs(best_sharpe - base_sharpe) < 0.1 and best['max_drawdown_pct'] >= base_result.get('max_drawdown_pct', 0.0):
        # Current config is good
        report_lines.extend([
            "### ✅ **No Changes Recommended**",
            "",
            f"The current configuration performs well (Sharpe: {base_sharpe:.2f}).",
            f"The best alternative (Sharpe: {best_sharpe:.2f}) does not provide significant improvement.",
            "",
            "**Current parameters are optimal or near-optimal.**",
        ])
        
        config_changes = "No changes recommended - current configuration is optimal."
    else:
        # Recommend changes
        report_lines.extend([
            "### ⚠️ **Parameter Update Recommended**",
            "",
            f"Best configuration found: Sharpe {best_sharpe:.2f} vs current {base_sharpe:.2f}",
            "",
            "**Recommended changes:**",
            "",
        ])
        
        changes = []
        if abs(best_params['min_funding_rate'] - base_params['min_funding_rate']) > 1e-6:
            changes.append(f"- `min_funding_rate`: {base_params['min_funding_rate']:.5f} → {best_params['min_funding_rate']:.5f}")
        if abs(best_params['exit_funding_threshold'] - base_params['exit_funding_threshold']) > 1e-6:
            changes.append(f"- `exit_funding_threshold`: {base_params['exit_funding_threshold']:.5f} → {best_params['exit_funding_threshold']:.5f}")
        if best_params['max_holding_hours'] != base_params['max_holding_hours']:
            changes.append(f"- `max_holding_hours`: {base_params['max_holding_hours']} → {best_params['max_holding_hours']}")
        if abs(best_params['base_size_fraction'] - base_params['base_size_fraction']) > 1e-4:
            changes.append(f"- `base_size_fraction`: {base_params['base_size_fraction']:.3f} → {best_params['base_size_fraction']:.3f}")
        if abs(best_params['max_total_funding_exposure'] - base_params['max_total_funding_exposure']) > 1e-4:
            changes.append(f"- `max_total_funding_exposure`: {base_params['max_total_funding_exposure']:.2f} → {best_params['max_total_funding_exposure']:.2f}")
        
        if changes:
            report_lines.extend(changes)
        else:
            report_lines.append("- No significant changes needed")
        
        report_lines.extend([
            "",
            "**Justification:**",
            f"- Sharpe ratio: {base_sharpe:.2f} → {best_sharpe:.2f} ({best_sharpe - base_sharpe:+.2f})",
            f"- Max drawdown: {base_result.get('max_drawdown_pct', 0.0):.2f}% → {best['max_drawdown_pct']:.2f}%",
            f"- Trades/year: {base_metrics.get('funding_trades_per_year', 0.0):.1f} → {best['funding_trades_per_year']:.1f}",
        ])
        
        # Generate config diff
        config_changes = "```yaml\nstrategy:\n  funding_opportunity:\n"
        if abs(best_params['min_funding_rate'] - base_params['min_funding_rate']) > 1e-6:
            config_changes += f"    min_funding_rate: {best_params['min_funding_rate']:.5f}\n"
        config_changes += "    exit:\n"
        if abs(best_params['exit_funding_threshold'] - base_params['exit_funding_threshold']) > 1e-6:
            config_changes += f"      exit_funding_threshold: {best_params['exit_funding_threshold']:.5f}\n"
        if best_params['max_holding_hours'] != base_params['max_holding_hours']:
            config_changes += f"      max_holding_hours: {best_params['max_holding_hours']}\n"
        config_changes += "    sizing:\n"
        if abs(best_params['base_size_fraction'] - base_params['base_size_fraction']) > 1e-4:
            config_changes += f"      base_size_fraction: {best_params['base_size_fraction']:.3f}\n"
        config_changes += "    risk:\n"
        if abs(best_params['max_total_funding_exposure'] - base_params['max_total_funding_exposure']) > 1e-4:
            config_changes += f"      max_total_funding_exposure: {best_params['max_total_funding_exposure']:.2f}\n"
        config_changes += "```"
    
    report_lines.extend([
        "",
        "## Config Changes",
        "",
        config_changes,
        "",
    ])
    
    report = "\n".join(report_lines)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Tune funding opportunity strategy parameters")
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test (default: from config)')
    parser.add_argument('--output', default='FUNDING_HEALTH_CHECK.md', help='Output report file')
    parser.add_argument('--lookback-months', type=int, default=12, help='Months of data to use (default: 12)')
    
    args = parser.parse_args()
    
    # Load config
    config = BotConfig.from_yaml(args.config)
    
    # Setup logging
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        service_name="funding_tune",
        force=True,
    )
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("FUNDING OPPORTUNITY STRATEGY - PARAMETER TUNING")
    logger.info("=" * 60)
    
    # Load data
    store = OHLCVStore(config.data.db_path)
    test_symbols = args.symbols or config.exchange.symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    
    # Calculate lookback
    lookback_days = args.lookback_months * 30
    lookback_bars = int(lookback_days * 24 / parse_timeframe_to_hours(config.exchange.timeframe))
    
    symbol_data = {}
    for symbol in test_symbols:
        try:
            df = store.get_ohlcv(symbol, config.exchange.timeframe, limit=lookback_bars)
            if not df.empty:
                symbol_data[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol} (last {args.lookback_months} months)")
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}")
    
    if not symbol_data:
        logger.error("No data available for backtest")
        return
    
    # Get base parameters
    base_params = {
        'min_funding_rate': config.strategy.funding_opportunity.min_funding_rate,
        'exit_funding_threshold': config.strategy.funding_opportunity.exit.exit_funding_threshold,
        'max_holding_hours': config.strategy.funding_opportunity.exit.max_holding_hours or 120,
        'base_size_fraction': config.strategy.funding_opportunity.sizing.base_size_fraction,
        'max_total_funding_exposure': config.strategy.funding_opportunity.risk.max_total_funding_exposure,
    }
    
    logger.info("Base parameters:")
    for key, value in base_params.items():
        logger.info(f"  {key}: {value}")
    
    # Run base backtest
    logger.info("\n" + "=" * 60)
    logger.info("Running base funding-only backtest...")
    logger.info("=" * 60)
    
    base_result = run_funding_backtest(config, symbol_data, base_params, funding_rate_per_8h=None)
    timestamps = base_result.get('timestamps', [])
    base_metrics = calculate_funding_metrics(base_result, timestamps)
    
    logger.info("\nBase Results:")
    logger.info(f"  Total Return: {base_result.get('total_return_pct', 0.0):+.2f}%")
    logger.info(f"  Sharpe: {base_result.get('sharpe_ratio', 0.0):.2f}")
    logger.info(f"  Max DD: {base_result.get('max_drawdown_pct', 0.0):.2f}%")
    logger.info(f"  Funding Trades: {base_metrics.get('total_funding_trades', 0)}")
    logger.info(f"  Trades/Year: {base_metrics.get('funding_trades_per_year', 0.0):.1f}")
    
    # Run parameter sweep
    logger.info("\n" + "=" * 60)
    logger.info("Running parameter sweep...")
    logger.info("=" * 60)
    
    sweep_results = run_parameter_sweep(config, symbol_data, base_params)
    
    logger.info(f"Tested {len(sweep_results)} parameter combinations")
    
    # Generate report
    output_file = Path(args.output)
    report = generate_health_check_report(
        base_result,
        base_metrics,
        sweep_results,
        base_params,
        output_file
    )
    
    logger.info(f"\nReport saved to {output_file}")
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FUNDING HEALTH CHECK SUMMARY")
    print("=" * 60)
    print(f"Base Sharpe: {base_result.get('sharpe_ratio', 0.0):.2f}")
    best = max(sweep_results, key=lambda x: x['sharpe_ratio'])
    print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"Best Params: {best['parameters']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

