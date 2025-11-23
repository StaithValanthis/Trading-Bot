"""OHLCV data downloader from exchange."""

import time
from typing import List, Optional
from datetime import datetime, timedelta
import ccxt

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
        lookback_days: int = 30,
        force_from_date: Optional[datetime] = None
    ):
        """
        Download OHLCV data and store in database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            lookback_days: Number of days of history to fetch
            force_from_date: If provided, download from this date (ignores existing data)
        """
        try:
            # Get latest timestamp in database
            latest_ts = self.store.get_latest_timestamp(symbol, timeframe)
            earliest_ts = self.store.get_earliest_timestamp(symbol, timeframe)
            
            # Calculate since timestamp
            if force_from_date is not None:
                # Force download from specific date
                since = int(force_from_date.timestamp() * 1000)
                # If existing data starts after force_from_date, we'll download historical data
                # and it will merge with existing data (or overwrite duplicates via UNIQUE constraint)
                if earliest_ts and earliest_ts > since:
                    # Need to download historical data
                    self.logger.info(
                        f"Force downloading {symbol} {timeframe} from {force_from_date.date()} "
                        f"(existing data starts at {datetime.fromtimestamp(earliest_ts/1000).date()})"
                    )
            elif latest_ts:
                # Fetch from latest timestamp (exclusive) - incremental download
                since = latest_ts + 1
                latest_dt = datetime.fromtimestamp(latest_ts / 1000)
                self.logger.debug(
                    f"Incremental download for {symbol} {timeframe}: "
                    f"existing data up to {latest_dt}, downloading new candles from {datetime.fromtimestamp(since/1000)}"
                )
            else:
                # Fetch from lookback_days ago - initial download
                since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
                self.logger.debug(
                    f"Initial download for {symbol} {timeframe}: "
                    f"no existing data, downloading {lookback_days} days of history"
                )
            
            # Calculate limit based on timeframe
            timeframe_to_hours = {
                '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 0.5,
                '1h': 1, '2h': 2, '4h': 4, '6h': 6, '12h': 12, '1d': 24
            }
            hours_per_bar = timeframe_to_hours.get(timeframe, 1)
            
            # Exchange API limit (usually 1000 candles per request)
            max_limit_per_request = 1000
            
            # Calculate how many candles we need
            if force_from_date is not None:
                days_to_now = (datetime.now() - force_from_date).days
                lookback_days = max(lookback_days, days_to_now)
            
            desired_total_candles = int(lookback_days * 24 / hours_per_bar)
            
            # For large historical downloads, we need to download in chunks
            all_data = []
            current_since = since
            max_iterations = 50  # Safety limit for large downloads
            iteration = 0
            
            self.logger.info(f"Downloading {symbol} {timeframe} data from {datetime.fromtimestamp(since/1000)}")
            
            # Download in chunks forward from since timestamp
            now_ms = int(datetime.now().timestamp() * 1000)
            
            self.logger.info(f"Starting chunked download for {symbol} {timeframe}: from {datetime.fromtimestamp(since/1000)} to now (~{(now_ms - since) / (1000 * 3600 * 24):.0f} days, max {max_iterations} iterations)")
            
            while iteration < max_iterations:
                # Limit per request (exchange max is usually 1000)
                # When force_from_date is set, calculate limit based on remaining time to now
                if force_from_date is not None:
                    # Calculate remaining time in milliseconds to reach now
                    ms_remaining = now_ms - current_since
                    if ms_remaining <= 0:
                        break  # Already reached now
                    # Calculate approximate bars remaining based on timeframe
                    ms_per_bar = hours_per_bar * 3600 * 1000
                    bars_remaining = int(ms_remaining / ms_per_bar)
                    limit = min(max_limit_per_request, bars_remaining + 100)
                else:
                    limit = min(max_limit_per_request, desired_total_candles - len(all_data) + 100)
                
                if limit <= 0:
                    break
                
                # Fetch data chunk
                self.logger.debug(f"  Iteration {iteration+1}: Fetching from {datetime.fromtimestamp(current_since/1000)}, limit={limit}")
                
                ohlcv_data = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not ohlcv_data or len(ohlcv_data) == 0:
                    if force_from_date is not None:
                        self.logger.warning(f"Exchange returned empty data for {symbol} {timeframe} at {datetime.fromtimestamp(current_since/1000)}. Stopping download.")
                    else:
                        self.logger.debug(f"Exchange returned empty data. Stopping.")
                    break
                
                self.logger.debug(f"  Got {len(ohlcv_data)} candles (requested {limit}), first: {datetime.fromtimestamp(ohlcv_data[0][0]/1000) if ohlcv_data else 'N/A'}, last: {datetime.fromtimestamp(ohlcv_data[-1][0]/1000) if ohlcv_data else 'N/A'}")
                
                # Remove duplicates from what we've already downloaded in this session
                existing_timestamps = {candle[0] for candle in all_data}
                new_candles = [candle for candle in ohlcv_data if candle[0] not in existing_timestamps]
                
                # When force_from_date is set, we want to download all the way to now
                # Don't stop just because we hit existing data in the database - continue downloading
                # The database UNIQUE constraint will handle duplicates
                if not new_candles and force_from_date is None:
                    # No new data, we've reached existing data (only stop if NOT force downloading)
                    break
                
                # Add new candles (empty list is OK if force_from_date is set - we'll continue)
                if new_candles:
                    all_data.extend(new_candles)
                    all_data.sort(key=lambda x: x[0])  # Sort by timestamp
                
                # For historical backfill: continue forward from the last timestamp
                last_timestamp = max(candle[0] for candle in ohlcv_data) if ohlcv_data else current_since
                current_since = last_timestamp + 1
                
                # Check if we've reached "now" (no more data available)
                ms_to_now = now_ms - current_since
                bars_to_now = ms_to_now / (hours_per_bar * 3600 * 1000) if hours_per_bar > 0 else 0
                
                if current_since >= now_ms:
                    # Reached present time
                    self.logger.info(f"  Iteration {iteration+1}: Reached present time. Stopping (total downloaded: {len(all_data)} candles).")
                    break
                else:
                    self.logger.debug(
                        f"  Iteration {iteration+1}: current_since={datetime.fromtimestamp(current_since/1000)}, "
                        f"bars_to_now={bars_to_now:.0f}, total_downloaded={len(all_data)}, "
                        f"will continue to next iteration"
                    )
                
                # If we got fewer candles than requested, check if we should stop
                # When force_from_date is set, only stop if we've actually reached "now"
                if len(ohlcv_data) < limit:
                    # Exchange returned fewer candles than requested
                    if force_from_date is not None:
                        # For force downloads, check if we've reached "now" before stopping
                        if ohlcv_data:
                            last_candle_time = ohlcv_data[-1][0]
                            ms_per_bar = hours_per_bar * 3600 * 1000
                            ms_from_last_to_now = now_ms - last_candle_time
                            bars_from_last_to_now = ms_from_last_to_now / ms_per_bar
                            
                            # If we're within 2 bars of "now", we're done
                            if bars_from_last_to_now <= 2:
                                self.logger.info(f"  Iteration {iteration+1}: Reached present time (last candle: {datetime.fromtimestamp(last_candle_time/1000)}, bars to now: {bars_from_last_to_now:.1f}). Stopping.")
                                break
                            # Otherwise, continue - exchange might have rate-limited or there might be a gap
                            self.logger.info(f"  Iteration {iteration+1}: Got {len(ohlcv_data)} candles (less than limit {limit}), but not at 'now' yet (bars to now: {bars_from_last_to_now:.0f}). Continuing...")
                        else:
                            # Empty response - break
                            self.logger.warning(f"  Iteration {iteration+1}: Got empty response. Stopping.")
                            break
                    else:
                        # Not force downloading - stop if we got fewer than requested
                        self.logger.debug(f"  Iteration {iteration+1}: Got {len(ohlcv_data)} candles (less than limit {limit}). Stopping (not force download).")
                        break
                else:
                    # Got full limit (or more) - definitely continue
                    self.logger.debug(f"  Iteration {iteration+1}: Got {len(ohlcv_data)} candles (>= limit {limit}). Continuing to next iteration.")
                
                # Check if we've downloaded enough candles
                # When force_from_date is set, we want to download all the way to now,
                # so ignore the desired_total_candles check
                if force_from_date is None and len(all_data) >= desired_total_candles:
                    break
                
                # For force_from_date: continue downloading all the way to now, don't stop at existing data
                # The database UNIQUE constraint will handle duplicates
                
                iteration += 1
                
                # Log progress every iteration when force downloading (for debugging)
                if force_from_date is not None or iteration % 5 == 0:
                    self.logger.info(
                        f"  Iteration {iteration+1}/{max_iterations}: downloaded {len(all_data)} candles, "
                        f"last timestamp: {datetime.fromtimestamp(last_timestamp/1000) if ohlcv_data else 'N/A'}, "
                        f"next: {datetime.fromtimestamp(current_since/1000)}, "
                        f"bars_to_now: {bars_to_now:.0f}"
                    )
                
                time.sleep(0.5)  # Rate limiting between chunks
            
            if all_data:
                self.store.store_ohlcv(symbol, timeframe, all_data)
                first_ts = all_data[0][0] if all_data else None
                last_ts = all_data[-1][0] if all_data else None
                
                # Determine download type for logging
                if force_from_date is not None:
                    download_type = "historical backfill"
                elif latest_ts:
                    download_type = "incremental update"
                else:
                    download_type = "initial download"
                
                self.logger.info(
                    f"Downloaded and stored {len(all_data)} candles for {symbol} {timeframe} "
                    f"({download_type}, completed {iteration} iterations, "
                    f"range: {datetime.fromtimestamp(first_ts/1000).date() if first_ts else 'N/A'} to {datetime.fromtimestamp(last_ts/1000).date() if last_ts else 'N/A'})"
                )
            else:
                if latest_ts:
                    self.logger.debug(
                        f"No new data for {symbol} {timeframe} "
                        f"(latest in DB: {datetime.fromtimestamp(latest_ts/1000)}, already up-to-date)"
                    )
                else:
                    self.logger.warning(f"No new data for {symbol} {timeframe} (no existing data in DB)")
            
        except ccxt.BadSymbol as e:
            # Symbol doesn't exist on exchange (delisted, inactive, etc.)
            # This is expected for some symbols - log at debug level, not error
            self.logger.debug(
                f"Symbol {symbol} not available on exchange (may be delisted or inactive): {e}"
            )
            # Don't raise - continue with other symbols
            return
        except Exception as e:
            # Other errors (network, rate limit, etc.) - log as error
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
        
        successful = 0
        skipped = 0
        failed = 0
        
        for symbol in symbols:
            try:
                self.download_and_store(symbol, timeframe, lookback_days)
                successful += 1
                # Rate limiting between symbols
                time.sleep(1)
            except ccxt.BadSymbol:
                # Symbol doesn't exist - already logged in download_and_store
                skipped += 1
                continue
            except Exception as e:
                self.logger.error(f"Failed to update {symbol}: {e}")
                failed += 1
                # Continue with other symbols
                continue
        
        if skipped > 0:
            self.logger.info(
                f"Data update complete: {successful} successful, {skipped} skipped (not available), {failed} failed"
            )

