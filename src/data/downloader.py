"""OHLCV data downloader from exchange."""

import time
from typing import List, Optional
from datetime import datetime, timedelta

from ..exchange.bybit_client import BybitClient
from ..data.ohlcv_store import OHLCVStore
from ..logging_utils import get_logger

logger = get_logger(__name__)


class DataDownloader:
    """Downloads and stores OHLCV data from exchange."""
    
    def __init__(self, exchange_client: BybitClient, ohlcv_store: OHLCVStore):
        """
        Initialize data downloader.
        
        Args:
            exchange_client: Bybit exchange client
            ohlcv_store: OHLCV data store
        """
        self.exchange = exchange_client
        self.store = ohlcv_store
        self.logger = get_logger(__name__)
    
    def download_and_store(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 30
    ):
        """
        Download OHLCV data and store in database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            lookback_days: Number of days of history to fetch
        """
        try:
            # Get latest timestamp in database
            latest_ts = self.store.get_latest_timestamp(symbol, timeframe)
            
            # Calculate since timestamp
            if latest_ts:
                # Fetch from latest timestamp (exclusive)
                since = latest_ts + 1
            else:
                # Fetch from lookback_days ago
                since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            # Calculate limit based on timeframe
            timeframe_to_hours = {
                '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 0.5,
                '1h': 1, '4h': 4, '1d': 24
            }
            hours_per_bar = timeframe_to_hours.get(timeframe, 1)
            limit = int(lookback_days * 24 / hours_per_bar) + 100  # Buffer
            
            self.logger.info(f"Downloading {symbol} {timeframe} data from {datetime.fromtimestamp(since/1000)}")
            
            # Fetch data
            ohlcv_data = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=limit
            )
            
            if ohlcv_data:
                self.store.store_ohlcv(symbol, timeframe, ohlcv_data)
                self.logger.info(f"Downloaded and stored {len(ohlcv_data)} candles for {symbol} {timeframe}")
            else:
                self.logger.warning(f"No new data for {symbol} {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol} {timeframe}: {e}")
            raise
    
    def update_all_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        lookback_days: int = 30
    ):
        """
        Update OHLCV data for all symbols.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe
            lookback_days: Number of days of history to fetch
        """
        self.logger.info(f"Updating OHLCV data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                self.download_and_store(symbol, timeframe, lookback_days)
                # Rate limiting between symbols
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Failed to update {symbol}: {e}")
                # Continue with other symbols
                continue

