"""Universe selector: dynamic symbol selection logic."""

import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, date, timedelta, timezone
import math
import pandas as pd
import numpy as np

from ..config import UniverseConfig
from ..exchange.bybit_client import BybitClient
from ..data.ohlcv_store import OHLCVStore
from ..universe.store import UniverseStore
from ..logging_utils import get_logger
from ..utils import parse_timeframe_to_hours

logger = get_logger(__name__)


class UniverseSelector:
    """Selects and maintains tradable symbol universe."""
    
    def __init__(
        self,
        config: UniverseConfig,
        exchange_client: BybitClient,
        ohlcv_store: OHLCVStore,
        universe_store: UniverseStore
    ):
        """
        Initialize universe selector.
        
        Args:
            config: Universe configuration
            exchange_client: Exchange client
            ohlcv_store: OHLCV data store
            universe_store: Universe membership store
        """
        self.config = config
        self.exchange = exchange_client
        self.ohlcv_store = ohlcv_store
        self.universe_store = universe_store
        self.logger = get_logger(__name__)
        self._ticker_cache: Dict[str, Dict] = {}
        self._ticker_cache_time: Dict[str, float] = {}
        self._cache_ttl = 3600  # 1 hour
    
    def fetch_all_symbols(self) -> List[str]:
        """
        Fetch all USDT perpetual symbols from exchange.
        
        Returns:
            List of symbol names
        """
        try:
            # Use CCXT to fetch markets
            markets = self.exchange.fetch_markets()
            
            symbols = []
            for symbol_id, market in markets.items():
                # Filter for USDT-margined perpetuals
                # Only include actively trading symbols (explicitly check active flag)
                is_active = market.get('active', False)  # Default to False if not present
                if (market.get('type') == 'swap' and
                    market.get('settle') == 'USDT' and
                    is_active and  # Only active symbols
                    market.get('quote') == 'USDT'):
                    # Extract base symbol (remove /USDT suffix)
                    base_symbol = market.get('base', '')
                    if not base_symbol:
                        continue  # Skip if no base symbol
                    symbol = f"{base_symbol}USDT"
                    symbols.append(symbol)
                    
                    # Cache metadata
                    self.universe_store.update_symbol_metadata(
                        symbol,
                        status=market.get('info', {}).get('status', 'Trading'),
                        category=market.get('info', {}).get('category', 'linear'),
                        quote_coin='USDT',
                        min_price=market.get('limits', {}).get('price', {}).get('min'),
                        max_price=market.get('limits', {}).get('price', {}).get('max'),
                        metadata=market.get('info', {})
                    )
            
            self.logger.info(f"Fetched {len(symbols)} USDT perpetual symbols from exchange")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols from exchange: {e}")
            raise
    
    def fetch_ticker_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch 24h ticker data for a symbol (cached).
        
        Args:
            symbol: Symbol name (internal format, e.g. 'BTCUSDT')
        
        Returns:
            Ticker dictionary with volume, etc., or None if error
        """
        # Check cache
        now = time.time()
        if (
            symbol in self._ticker_cache
            and symbol in self._ticker_cache_time
            and now - self._ticker_cache_time[symbol] < self._cache_ttl
        ):
            return self._ticker_cache[symbol]
        
        try:
            # Delegate to BybitClient, which handles symbol normalization
            ticker = self.exchange.fetch_ticker(symbol)
        except Exception as e:
            # Log at INFO for visibility, but only once per symbol per cache window
            self.logger.info(f"Ticker data unavailable for {symbol}: {e}")
            return None
        
        data = {
            'volume_24h': ticker.get('quoteVolume', 0) or 0.0,
            'open_interest': None,  # OI not always available in CCXT
            'last_price': ticker.get('last', 0) or ticker.get('close', 0) or 0.0,
            'bid': ticker.get('bid', 0) or 0.0,
            'ask': ticker.get('ask', 0) or 0.0,
            'spread_bps': 0.0,
            'ccxt_symbol': ticker.get('symbol'),
        }
        
        # Calculate spread if bid/ask available
        bid = data['bid']
        ask = data['ask']
        if bid > 0 and ask > 0:
            spread = (ask - bid) / bid
            data['spread_bps'] = spread * 10000  # basis points
        
        # Update cache
        self._ticker_cache[symbol] = data
        self._ticker_cache_time[symbol] = now
        
        return data
    
    def check_liquidity_filters(
        self,
        symbol: str,
        entry_threshold: bool = True
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if symbol passes liquidity filters.
        
        Args:
            symbol: Symbol name
            entry_threshold: If True, use entry threshold; if False, use exit threshold
        
        Returns:
            Tuple of (passes, reason, metadata)
        """
        ticker = self.fetch_ticker_data(symbol)
        
        if not ticker:
            return False, "ticker_data_unavailable", {}
        
        volume_24h = ticker.get('volume_24h', 0)
        threshold = (self.config.min_24h_volume_entry if entry_threshold
                    else self.config.min_24h_volume_exit)
        
        if volume_24h < threshold:
            return False, f"volume_below_threshold", {
                'volume_24h': volume_24h,
                'threshold': threshold
            }
        
        # Optional open interest filter
        if self.config.min_open_interest is not None:
            oi = ticker.get('open_interest')
            if oi is None or oi < self.config.min_open_interest:
                return False, "open_interest_below_threshold", {
                    'open_interest': oi,
                    'threshold': self.config.min_open_interest
                }
        
        # Optional spread filter
        if self.config.max_spread_bps is not None:
            spread = ticker.get('spread_bps', 0)
            if spread > self.config.max_spread_bps:
                return False, "spread_too_wide", {
                    'spread_bps': spread,
                    'threshold': self.config.max_spread_bps
                }
        
        return True, None, {
            'volume_24h': volume_24h,
            'open_interest': ticker.get('open_interest'),
            'spread_bps': ticker.get('spread_bps')
        }
    
    def _get_history_requirements(self, timeframe: str) -> Tuple[int, float]:
        """Return required bar count and candles per day for timeframe."""
        hours_per_bar = parse_timeframe_to_hours(timeframe)
        if hours_per_bar <= 0:
            return 0, 0.0
        candles_per_day = max(24 / hours_per_bar, 1.0)
        required_bars = int(math.ceil(self.config.min_history_days * candles_per_day))
        return required_bars, candles_per_day

    def _get_history_buffer_bars(self, candles_per_day: float) -> int:
        """Convert configured buffer days into bars for the timeframe."""
        buffer_days = getattr(self.config, "history_buffer_days", 5)
        return int(math.ceil(max(buffer_days, 0) * candles_per_day))

    def check_historical_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if symbol has sufficient historical data.
        
        Args:
            symbol: Symbol name
            timeframe: Trading timeframe (e.g., '1h', '4h')
        
        Returns:
            Tuple of (passes, reason, metadata)
        """
        try:
            required_bars, candles_per_day = self._get_history_requirements(timeframe)
            if required_bars == 0:
                return False, "invalid_timeframe", {'timeframe': timeframe}
            buffer_bars = self._get_history_buffer_bars(candles_per_day)
            
            # Load data (with buffer for gap checking)
            df = self.ohlcv_store.get_ohlcv(symbol, timeframe, limit=required_bars + buffer_bars)
            
            # If we have no data at all for this symbol/timeframe, treat it as
            # "insufficient_history" for warmup purposes rather than a permanent
            # "no_data" failure, as long as the symbol exists on the exchange.
            if df.empty:
                symbol_meta = self.universe_store.get_symbol_metadata(symbol)
                if symbol_meta:
                    # Start warmup tracking if not already tracked
                    warmup_status = self.universe_store.get_warmup_status(symbol)
                    if warmup_status is None:
                        first_seen = date.today()
                        warmup_start = first_seen
                        self.universe_store.track_warmup(symbol, first_seen, warmup_start)
                    
                    return False, "insufficient_history", {
                        'actual_bars': 0,
                        'required_bars': required_bars,
                        'actual_days': 0,
                        'threshold_days': self.config.min_history_days,
                        'candles_per_day': candles_per_day
                    }
                
                # If we don't even have symbol metadata, treat as generic no_data
                return False, "no_data", {}
            
            # Check data age (FIX: Use UTC-aware datetime)
            latest_timestamp = df.index[-1]
            now_utc = datetime.now(timezone.utc)
            
            # Ensure pandas timestamp is UTC-aware
            if isinstance(latest_timestamp, pd.Timestamp):
                if latest_timestamp.tz is None:
                    latest_dt = latest_timestamp.to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    latest_dt = latest_timestamp.to_pydatetime()
            else:
                # Fallback: assume UTC if naive
                latest_dt = latest_timestamp if hasattr(latest_timestamp, 'tzinfo') and latest_timestamp.tzinfo else latest_timestamp.replace(tzinfo=timezone.utc)
            
            age_days = (now_utc - latest_dt).days
            
            if age_days > self.config.max_days_since_last_update:
                return False, "data_too_old", {
                    'days_since_update': age_days,
                    'threshold': self.config.max_days_since_last_update,
                    'latest_timestamp': latest_timestamp.isoformat() if hasattr(latest_timestamp, 'isoformat') else str(latest_timestamp)
                }

            # Check total history based on days (primary) and bars (for diagnostics)
            actual_bars = len(df)
            oldest_timestamp = df.index[0]
            if isinstance(oldest_timestamp, pd.Timestamp):
                oldest_dt = oldest_timestamp.to_pydatetime()
                if oldest_dt.tzinfo is None:
                    oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            else:
                oldest_dt = oldest_timestamp if hasattr(oldest_timestamp, 'tzinfo') and oldest_timestamp.tzinfo else oldest_timestamp.replace(tzinfo=timezone.utc)
            
            if isinstance(latest_dt, pd.Timestamp):
                latest_dt = latest_dt.to_pydatetime()
            history_days = (latest_dt - oldest_dt).days

            # Require at least min_history_days worth of data in calendar terms
            if history_days < self.config.min_history_days:
                return False, "insufficient_history", {
                    'actual_bars': actual_bars,
                    'required_bars': required_bars,
                    'history_days': history_days,
                    'threshold_days': self.config.min_history_days,
                    'candles_per_day': candles_per_day,
                    'reason': 'history_days_below_threshold',
                }
            
            # Check data gaps (FIX: Use timeframe-aware calculation)
            expected_candles = max(history_days * candles_per_day, 1)
            gap_pct = (1 - actual_bars / expected_candles) * 100 if expected_candles > 0 else 0
            
            if gap_pct > self.config.max_data_gap_pct:
                return False, "too_many_gaps", {
                    'gap_pct': gap_pct,
                    'threshold': self.config.max_data_gap_pct,
                    'expected_candles': expected_candles,
                    'actual_candles': actual_bars,
                    'candles_per_day': candles_per_day,
                    'timeframe': timeframe
                }
            
            return True, None, {
                'history_days': history_days,
                'data_points': actual_bars,
                'gap_pct': gap_pct,
                'candles_per_day': candles_per_day,
                'required_bars': required_bars
            }
            
        except Exception as e:
            self.logger.debug(f"Error checking historical data for {symbol}: {e}")
            return False, "data_check_error", {'error': str(e)}
    
    def check_volatility_filters(self, symbol: str, timeframe: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if symbol passes volatility and price filters.
        
        Args:
            symbol: Symbol name
            timeframe: Trading timeframe
        
        Returns:
            Tuple of (passes, reason, metadata)
        """
        try:
            # Get recent OHLCV
            required_bars, candles_per_day = self._get_history_requirements(timeframe)
            if required_bars == 0:
                return False, "invalid_timeframe", {'timeframe': timeframe}
            buffer_bars = self._get_history_buffer_bars(candles_per_day)
            limit = max(required_bars + buffer_bars, 24)
            df = self.ohlcv_store.get_ohlcv(symbol, timeframe, limit=limit)
            
            if df.empty or len(df) < 24:
                return False, "insufficient_data_for_volatility", {}
            
            # Check price floor
            current_price = df['close'].iloc[-1]
            if current_price < self.config.min_price_usdt:
                return False, "price_too_low", {
                    'price': current_price,
                    'threshold': self.config.min_price_usdt
                }
            
            # Calculate realized volatility
            returns = df['close'].pct_change().dropna()
            if len(returns) < 24:
                return False, "insufficient_data_for_volatility", {}
            
            # Annualized volatility
            volatility_pct = returns.std() * np.sqrt(365 * 24) * 100  # For hourly data
            
            if volatility_pct > self.config.max_realized_vol_pct:
                return False, "volatility_too_high", {
                    'volatility_pct': volatility_pct,
                    'threshold': self.config.max_realized_vol_pct
                }
            
            # Check limit move frequency (simplified: large price moves)
            # For now, skip this check (would require more data)
            
            return True, None, {
                'current_price': current_price,
                'volatility_pct': volatility_pct
            }
            
        except Exception as e:
            self.logger.debug(f"Error checking volatility for {symbol}: {e}")
            return False, "volatility_check_error", {'error': str(e)}
    
    def check_warmup_period(self, symbol: str) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if symbol has completed warm-up period.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Tuple of (passes, reason, metadata)
        """
        warmup_status = self.universe_store.get_warmup_status(symbol)
        
        if warmup_status is None:
            # Not tracking warm-up, assume eligible (old symbol)
            return True, None, {}
        
        if warmup_status.get('eligible_date'):
            # Already eligible
            return True, None, {'eligible_date': warmup_status['eligible_date']}
        
        # Check if warm-up period has passed
        warmup_start = datetime.fromisoformat(warmup_status['warmup_start_date']).date()
        days_in_warmup = (date.today() - warmup_start).days
        
        if days_in_warmup < self.config.warmup_days:
            return False, "warmup_period_not_complete", {
                'days_in_warmup': days_in_warmup,
                'required_days': self.config.warmup_days,
                'warmup_start_date': warmup_start.isoformat()
            }
        
        # Warm-up period complete, mark as eligible
        self.universe_store.mark_warmup_eligible(symbol, date.today())
        
        return True, None, {'warmup_complete_date': date.today().isoformat()}
    
    def check_min_time_in_universe(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol has been in universe for minimum time (hysteresis for exit).
        
        Args:
            symbol: Symbol name
        
        Returns:
            Tuple of (can_exit, reason)
        """
        history = self.universe_store.get_history(symbol)
        
        if not history:
            return True, None  # Not in universe, can exit
        
        # Find last 'added' action
        last_added = None
        for record in reversed(history):
            if record['action'] == 'added':
                last_added = datetime.fromisoformat(record['date']).date()
                break
        
        if last_added is None:
            return True, None  # Never properly added
        
        days_in_universe = (date.today() - last_added).days
        
        if days_in_universe < self.config.min_time_in_universe_days:
            return False, "min_time_not_reached"
        
        return True, None
    
    def evaluate_symbol(
        self,
        symbol: str,
        timeframe: str,
        for_entry: bool = True,
        check_min_time: bool = False
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Evaluate if symbol should be in universe.
        
        Args:
            symbol: Symbol name
            timeframe: Trading timeframe
            for_entry: If True, check entry conditions; if False, check exit conditions
            check_min_time: If True, also check minimum time in universe
        
        Returns:
            Tuple of (eligible, reason, metadata)
        """
        metadata = {}
        
        # Check overrides first
        if symbol in self.config.exclude_list:
            return False, "in_exclude_list", {}
        
        # Check if symbol is in include_list (for logging/prioritization, but still check filters)
        is_preferred = symbol in self.config.include_list
        
        # Check if symbol is delisted (CRITICAL: Always check, even for include_list)
        symbol_meta = self.universe_store.get_symbol_metadata(symbol)
        if symbol_meta and symbol_meta.get('status') not in ['Trading', None]:
            if is_preferred:
                self.logger.warning(
                    f"Symbol {symbol} is in include_list but is delisted (status: {symbol_meta.get('status')}). "
                    f"Will not be included in universe."
                )
            return False, "delisted", {'status': symbol_meta.get('status')}
        
        # Check historical data (CRITICAL: Required for trading, even for include_list)
        passes_data, reason, data_meta = self.check_historical_data(symbol, timeframe)
        metadata.update(data_meta)
        
        if not passes_data:
            if is_preferred:
                self.logger.warning(
                    f"Symbol {symbol} is in include_list but failed data requirements: {reason}. "
                    f"Will not be included until data is available."
                )
                # If insufficient data and it's a new symbol, start warm-up tracking
                if reason == "insufficient_history":
                    warmup_status = self.universe_store.get_warmup_status(symbol)
                    if warmup_status is None:
                        # New symbol, start tracking
                        first_seen = date.today()
                        warmup_start = first_seen
                        self.universe_store.track_warmup(symbol, first_seen, warmup_start)
            
            return False, reason, metadata
        
        # Check liquidity (with hysteresis)
        # For include_list symbols, we still check but with relaxed thresholds if desired
        passes_liquidity, reason, liq_meta = self.check_liquidity_filters(symbol, entry_threshold=for_entry)
        metadata.update(liq_meta)
        
        if not passes_liquidity:
            if is_preferred:
                self.logger.warning(
                    f"Symbol {symbol} is in include_list but failed liquidity requirements: {reason}. "
                    f"Will not be included until liquidity improves."
                )
            return False, reason, metadata
        
        # Check warm-up period (for entry only)
        if for_entry:
            passes_warmup, reason, warmup_meta = self.check_warmup_period(symbol)
            metadata.update(warmup_meta)
            
            if not passes_warmup:
                if is_preferred:
                    self.logger.warning(
                        f"Symbol {symbol} is in include_list but is still in warm-up period: {reason}. "
                        f"Will not be included until warm-up completes."
                    )
                return False, reason, metadata
        
        # Check volatility filters
        passes_vol, reason, vol_meta = self.check_volatility_filters(symbol, timeframe)
        metadata.update(vol_meta)
        
        if not passes_vol:
            if is_preferred:
                self.logger.warning(
                    f"Symbol {symbol} is in include_list but failed volatility requirements: {reason}. "
                    f"Will not be included due to excessive volatility."
                )
            return False, reason, metadata
        
        # Check minimum time in universe (for exit only, with check_min_time flag)
        if not for_entry and check_min_time:
            can_exit, reason = self.check_min_time_in_universe(symbol)
            if not can_exit:
                return True, reason, metadata  # Keep in universe
        
        # All checks passed
        if is_preferred:
            metadata['is_preferred'] = True
            return True, "in_include_list_and_passed_filters", metadata
        else:
            return True, None, metadata
    
    def build_universe(
        self,
        timeframe: str,
        current_date: Optional[date] = None
    ) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Build current tradable universe.
        
        Args:
            timeframe: Trading timeframe
            current_date: Date for universe (default: today)
        
        Returns:
            Tuple of (universe set, changes dictionary)
        """
        if current_date is None:
            current_date = date.today()
        
        self.logger.info(f"Building universe for {current_date}")
        
        # Fetch all symbols from exchange
        all_symbols = self.fetch_all_symbols()
        
        # Get current universe
        current_universe = self.universe_store.get_current_universe()
        
        # Evaluate each symbol
        eligible_symbols = set()
        changes = {}
        rejection_reasons = {}  # Track rejection reasons for diagnostics
        
        for symbol in all_symbols:
            # Check if currently in universe
            currently_in = symbol in current_universe
            
            # Evaluate for entry (if not in) or exit (if in)
            for_entry = not currently_in
            
            # Check minimum time only for exits
            eligible, reason, metadata = self.evaluate_symbol(
                symbol,
                timeframe,
                for_entry=for_entry,
                check_min_time=not for_entry
            )
            
            if eligible:
                eligible_symbols.add(symbol)
                
                if not currently_in:
                    # New addition
                    ticker = self.fetch_ticker_data(symbol)
                    changes[symbol] = {
                        'action': 'added',
                        'reason': reason or 'passed_all_filters',
                        'volume_24h': ticker.get('volume_24h') if ticker else None,
                        'open_interest': ticker.get('open_interest') if ticker else None,
                        'metadata': metadata
                    }
            else:
                # Track rejection reasons
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = []
                rejection_reasons[reason].append(symbol)
                
                # Log rejection (INFO for first few, DEBUG for rest)
                if len(rejection_reasons[reason]) <= 5:
                    self.logger.info(
                        f"Symbol {symbol} rejected: {reason} "
                        f"(metadata: {metadata})"
                    )
                else:
                    self.logger.debug(
                        f"Symbol {symbol} rejected: {reason} "
                        f"(metadata: {metadata})"
                    )
                
                if currently_in:
                    # Removal
                    ticker = self.fetch_ticker_data(symbol)
                    changes[symbol] = {
                        'action': 'removed',
                        'reason': reason or 'failed_filters',
                        'volume_24h': ticker.get('volume_24h') if ticker else None,
                        'open_interest': ticker.get('open_interest') if ticker else None,
                        'metadata': metadata
                    }
        
        # Apply turnover control
        # NOTE: When there is no existing universe (first build), we allow all eligible additions.
        # Turnover limiting only makes sense once we already have a non-empty universe.
        if self.config.max_turnover_per_rebalance_pct > 0 and current_universe:
            max_changes = max(1, int(len(current_universe) * self.config.max_turnover_per_rebalance_pct / 100))
            
            additions = [s for s, c in changes.items() if c['action'] == 'added']
            removals = [s for s, c in changes.items() if c['action'] == 'removed']
            
            total_changes = len(additions) + len(removals)
            if total_changes > max_changes:
                # Prioritize by volume (highest for additions, lowest for removals)
                addition_volumes = {
                    s: changes[s].get('volume_24h', 0) for s in additions
                }
                removal_volumes = {
                    s: changes[s].get('volume_24h', float('inf')) for s in removals
                }
                
                additions_sorted = sorted(additions, key=lambda s: addition_volumes[s], reverse=True)
                removals_sorted = sorted(removals, key=lambda s: removal_volumes[s])
                
                # Decide how many additions/removals to keep under the turnover cap
                # Strategy:
                #   - If only additions or only removals, allow up to max_changes of that type
                #   - If both exist, split budget but ensure at least 1 of each when possible
                if additions and not removals:
                    allowed_additions = min(len(additions_sorted), max_changes)
                    allowed_removals = 0
                elif removals and not additions:
                    allowed_additions = 0
                    allowed_removals = min(len(removals_sorted), max_changes)
                else:
                    # Both additions and removals present
                    # Start with a 50/50 split, but ensure at least 1 of each when enough budget
                    half = max_changes // 2
                    if max_changes == 1:
                        # With very small universes, allow at least 1 change of whichever side has more pressure
                        if len(additions_sorted) >= len(removals_sorted):
                            allowed_additions, allowed_removals = 1, 0
                        else:
                            allowed_additions, allowed_removals = 0, 1
                    else:
                        allowed_additions = min(len(additions_sorted), max(1, half))
                        remaining_budget = max_changes - allowed_additions
                        allowed_removals = min(len(removals_sorted), max(1, remaining_budget)) if remaining_budget > 0 else 0
                
                keep_additions = set(additions_sorted[:allowed_additions])
                keep_removals = set(removals_sorted[:allowed_removals])
                
                # Remove excess additions
                for symbol in additions:
                    if symbol not in keep_additions:
                        if symbol in eligible_symbols:
                            eligible_symbols.remove(symbol)
                        if symbol in changes:
                            del changes[symbol]
                
                # Remove excess removals
                for symbol in removals:
                    if symbol not in keep_removals:
                        if symbol not in eligible_symbols and symbol in current_universe:
                            eligible_symbols.add(symbol)  # Keep in universe
                        if symbol in changes:
                            del changes[symbol]
        
        # Log snapshot
        self.universe_store.log_universe_snapshot(current_date, eligible_symbols, changes)
        
        # Log summary of rejections (diagnostic)
        if rejection_reasons:
            self.logger.info("Universe filter summary:")
            for reason, symbols in sorted(rejection_reasons.items(), key=lambda x: len(x[1]), reverse=True):
                self.logger.info(f"  {reason}: {len(symbols)} symbols rejected")
                if len(symbols) <= 10:
                    self.logger.info(f"    Examples: {', '.join(symbols[:10])}")
        
        # Warn if universe is empty
        if not eligible_symbols:
            self.logger.warning(
                "⚠️  Universe is empty! All symbols were filtered out. "
                "Check filter thresholds and data availability. "
                f"Consider: (1) Lowering min_24h_volume_entry (current: {self.config.min_24h_volume_entry:,.0f}), "
                f"(2) Reducing min_history_days (current: {self.config.min_history_days}), "
                f"(3) Checking if data exists for symbols (timeframe: {timeframe})"
            )
        
        self.logger.info(
            f"Universe built: {len(eligible_symbols)} symbols "
            f"({len([c for c in changes.values() if c['action'] == 'added'])} added, "
            f"{len([c for c in changes.values() if c['action'] == 'removed'])} removed)"
        )
        
        return eligible_symbols, changes
    
    def get_universe(self, as_of_date: Optional[date] = None) -> Set[str]:
        """
        Get current (or historical) universe.
        
        Args:
            as_of_date: Date to query (default: latest)
        
        Returns:
            Set of symbol names
        """
        return self.universe_store.get_universe(as_of_date)
    
    def get_universe_stats(self) -> Dict:
        """
        Get statistics about current universe.
        
        Returns:
            Dictionary with stats
        """
        universe = self.get_universe()
        
        if not universe:
            return {
                'size': 0,
                'avg_volume_24h': 0,
                'top_5_by_volume': []
            }
        
        # Get volumes for all symbols
        volumes = []
        for symbol in universe:
            ticker = self.fetch_ticker_data(symbol)
            if ticker and ticker.get('volume_24h'):
                volumes.append((symbol, ticker['volume_24h']))
        
        volumes.sort(key=lambda x: x[1], reverse=True)
        
        avg_volume = np.mean([v[1] for v in volumes]) if volumes else 0
        top_5 = volumes[:5]
        
        return {
            'size': len(universe),
            'avg_volume_24h': avg_volume,
            'top_5_by_volume': [{'symbol': s, 'volume_24h': v} for s, v in top_5]
        }

