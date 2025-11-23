#!/usr/bin/env python3
"""
Download historical OHLCV data for backtesting.

This script downloads historical candle data for multiple symbols and timeframes,
which is required before running the timeframe comparison backtests.

Usage:
    python scripts/download_historical_data.py --config config.yaml --symbols BTCUSDT ETHUSDT --timeframes 1h 4h 1d --days 1095
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BotConfig
from src.exchange.bybit_client import BybitClient
from src.data.ohlcv_store import OHLCVStore
from src.data.downloader import DataDownloader
from src.logging_utils import setup_logging, get_logger

def download_historical_data(
    config_path: str,
    symbols: List[str],
    timeframes: List[str],
    lookback_days: int
):
    """
    Download historical OHLCV data for symbols and timeframes.
    
    Args:
        config_path: Path to config file
        symbols: List of symbols to download
        timeframes: List of timeframes to download (e.g., ['1h', '4h'])
        lookback_days: Number of days of history to fetch
    """
    logger = get_logger(__name__)
    
    # Load config
    logger.info(f"Loading config from {config_path}")
    config = BotConfig.from_yaml(config_path)
    
    # Initialize components
    logger.info("Initializing exchange client and data store...")
    exchange = BybitClient(config.exchange)
    store = OHLCVStore(config.data.db_path)
    downloader = DataDownloader(exchange, store)
    
    total_combinations = len(symbols) * len(timeframes)
    logger.info(
        f"Will download data for {len(symbols)} symbols × {len(timeframes)} timeframes = {total_combinations} combinations"
    )
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Lookback: {lookback_days} days")
    
    completed = 0
    errors = []
    
    # Download for each symbol/timeframe combination
    for symbol in symbols:
        for timeframe in timeframes:
            completed += 1
            logger.info(
                f"\n[{completed}/{total_combinations}] Downloading {symbol} {timeframe} data..."
            )
            
            try:
                # Check what data already exists
                existing_data = store.get_ohlcv(symbol, timeframe)
                if not existing_data.empty:
                    existing_bars = len(existing_data)
                    logger.info(
                        f"  Found {existing_bars} existing bars in database. "
                        f"Will fetch additional data if needed."
                    )
                
                # Download data (DataDownloader will fetch from latest timestamp or lookback_days ago)
                # Note: Some timeframes may not be supported by Bybit (e.g., 8h is not a standard Bybit timeframe)
                try:
                    downloader.download_and_store(symbol, timeframe, lookback_days=lookback_days)
                except Exception as download_error:
                    # Check if this is an unsupported timeframe error
                    error_str = str(download_error).lower()
                    if 'timeframe' in error_str or 'invalid' in error_str or 'not supported' in error_str:
                        logger.warning(
                            f"  ⚠ Timeframe {timeframe} may not be supported by Bybit. "
                            f"Error: {download_error}"
                        )
                        errors.append(f"{symbol} {timeframe}: Unsupported timeframe")
                        continue
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Verify what we have now
                updated_data = store.get_ohlcv(symbol, timeframe)
                if not updated_data.empty:
                    updated_bars = len(updated_data)
                    logger.info(f"  ✓ Successfully downloaded/stored {updated_bars} total bars for {symbol} {timeframe}")
                    
                    # Check date range
                    oldest = updated_data.index[0]
                    newest = updated_data.index[-1]
                    days_covered = (newest - oldest).days
                    logger.info(
                        f"  Date range: {oldest.date()} to {newest.date()} ({days_covered} days)"
                    )
                else:
                    logger.warning(
                        f"  ⚠ No data available for {symbol} {timeframe} after download. "
                        f"This timeframe may not be supported by Bybit. "
                        f"Supported Bybit timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M"
                    )
                    errors.append(f"{symbol} {timeframe}: No data after download (may be unsupported timeframe)")
                
                # Rate limiting: sleep between downloads to avoid hitting API limits
                time.sleep(0.5)  # 500ms between downloads
                
            except Exception as e:
                logger.error(f"  ✗ Error downloading {symbol} {timeframe}: {e}", exc_info=True)
                errors.append(f"{symbol} {timeframe}: {str(e)}")
                continue
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Completed: {completed}/{total_combinations} combinations")
    
    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✓ All downloads completed successfully!")
    
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Download historical OHLCV data for backtesting'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Symbols to download (e.g., BTCUSDT ETHUSDT SOLUSDT)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        required=True,
        help='Timeframes to download (e.g., 1h 4h 1d)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1095,  # 3 years default
        help='Number of days of history to fetch (default: 1095 = 3 years)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir='logs', level='INFO', service_name='download_data')
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("HISTORICAL DATA DOWNLOAD")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Lookback: {args.days} days")
    
    try:
        download_historical_data(
            args.config,
            args.symbols,
            args.timeframes,
            args.days
        )
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

