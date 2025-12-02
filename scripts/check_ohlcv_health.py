#!/usr/bin/env python3
"""
OHLCV Data Health Checker

Diagnostic tool to check OHLCV data quality for symbols/timeframes.
Reports history days, gap percentage, missing ranges, and health status.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import BotConfig
from src.data.ohlcv_store import OHLCVStore
from src.logging_utils import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Check OHLCV data health for symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check specific symbols
  python scripts/check_ohlcv_health.py --config config.yaml --timeframe 4h --symbols BTCUSDT ETHUSDT --days 30

  # Check all symbols in universe
  python scripts/check_ohlcv_health.py --config config.yaml --timeframe 4h --universe --days 30

  # Check with custom gap threshold
  python scripts/check_ohlcv_health.py --config config.yaml --timeframe 4h --symbols BTCUSDT --days 30 --max-gap-pct 10.0
        """
    )
    
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--timeframe', required=True, help='Timeframe (e.g., 4h, 1h)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to check (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--universe', action='store_true', help='Check all symbols in universe')
    parser.add_argument('--days', type=int, required=True, help='Required days of history')
    parser.add_argument('--max-gap-pct', type=float, default=5.0, help='Maximum allowed gap percentage (default: 5.0)')
    parser.add_argument('--auto-backfill', action='store_true', help='Automatically backfill missing data (requires exchange client)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    logger = get_logger(__name__)
    
    # Load config
    try:
        config = BotConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize store
    store = OHLCVStore(config.data.db_path)
    
    # Determine symbols to check
    symbols_to_check = []
    if args.universe:
        from src.exchange.bybit_client import BybitClient
        from src.universe.selector import UniverseSelector
        from src.universe.store import UniverseStore
        
        exchange = BybitClient(config.exchange)
        universe_store = UniverseStore(config.data.db_path)
        selector = UniverseSelector(config.universe, exchange, store, universe_store)
        symbols_to_check = list(selector.get_universe())
        
        if not symbols_to_check:
            print("No symbols in universe. Run 'universe build' first.", file=sys.stderr)
            sys.exit(1)
    elif args.symbols:
        symbols_to_check = args.symbols
    else:
        print("Error: Must specify either --symbols or --universe", file=sys.stderr)
        sys.exit(1)
    
    # Check health for each symbol
    unhealthy_count = 0
    results = []
    
    for symbol in symbols_to_check:
        try:
            health = store.check_health(
                symbol,
                args.timeframe,
                args.days,
                max_gap_pct=args.max_gap_pct
            )
            results.append(health)
            
            if not health['is_healthy']:
                unhealthy_count += 1
            
            # Print results
            print(f"\n{'='*60}")
            print(f"Symbol: {symbol}")
            print(f"  Timeframe: {args.timeframe}")
            print(f"  Required days: {args.days}")
            print(f"  History days: {health['history_days']:.1f}")
            print(f"  Actual bars: {health['actual_bars']}")
            print(f"  Expected bars: {health['expected_bars']}")
            print(f"  Gap %: {health['gap_pct']:.2f}%")
            
            if health['earliest_timestamp']:
                from datetime import datetime
                earliest_dt = datetime.fromtimestamp(health['earliest_timestamp'] / 1000)
                print(f"  Earliest: {earliest_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                print(f"  Earliest: N/A")
            
            if health['latest_timestamp']:
                from datetime import datetime
                latest_dt = datetime.fromtimestamp(health['latest_timestamp'] / 1000)
                print(f"  Latest: {latest_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                print(f"  Latest: N/A")
            
            # Status
            if health['is_healthy']:
                print(f"  Status: ✓ HEALTHY")
            else:
                print(f"  Status: ✗ UNHEALTHY")
                print(f"  Issues: {', '.join(health['issues'])}")
                if health['needs_backfill']:
                    print(f"  Action needed: BACKFILL REQUIRED")
            
            # Missing ranges
            if health['missing_ranges']:
                print(f"  Missing ranges: {len(health['missing_ranges'])} gaps detected")
                if args.verbose:
                    for start_ms, end_ms in health['missing_ranges'][:10]:  # Show first 10
                        start_dt = datetime.fromtimestamp(start_ms / 1000)
                        end_dt = datetime.fromtimestamp(end_ms / 1000)
                        print(f"    - {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
                    if len(health['missing_ranges']) > 10:
                        print(f"    ... and {len(health['missing_ranges']) - 10} more gaps")
            
        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}", exc_info=args.verbose)
            print(f"\n{'='*60}")
            print(f"Symbol: {symbol}")
            print(f"  Status: ✗ ERROR - {e}")
            unhealthy_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Total symbols checked: {len(symbols_to_check)}")
    print(f"  Healthy: {len(symbols_to_check) - unhealthy_count}")
    print(f"  Unhealthy: {unhealthy_count}")
    
    # Auto-backfill if requested
    if args.auto_backfill and unhealthy_count > 0:
        print(f"\nAuto-backfilling unhealthy symbols...")
        from src.exchange.bybit_client import BybitClient
        from src.data.downloader import DataDownloader
        
        exchange = BybitClient(config.exchange)
        downloader = DataDownloader(exchange, store)
        
        for health in results:
            if health['needs_backfill']:
                symbol = health['symbol']
                print(f"  Backfilling {symbol} {args.timeframe}...")
                try:
                    # Calculate lookback days (use required_days + buffer)
                    lookback_days = health['required_days'] + 5  # Add buffer
                    downloader.download_and_store(
                        symbol,
                        args.timeframe,
                        lookback_days=lookback_days
                    )
                    print(f"    ✓ Backfill completed")
                except Exception as e:
                    print(f"    ✗ Backfill failed: {e}")
                    logger.error(f"Backfill failed for {symbol}: {e}", exc_info=args.verbose)
    
    # Exit code
    if unhealthy_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

