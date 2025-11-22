"""Bybit exchange client wrapper using CCXT."""

import ccxt
import time
import hmac
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging
from random import random
import json

# Optional import for direct Bybit v5 API calls (fallback when CCXT doesn't support conditional orders)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..config import ExchangeConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class BybitClient:
    """Wrapper for Bybit exchange API via CCXT with basic retry/backoff."""

    def __init__(self, config: ExchangeConfig):
        """
        Initialize Bybit client.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        try:
            self.logger.debug("Starting BybitClient initialization...")
            
            # Configure exchange
            exchange_options = {
                "defaultType": "swap",  # USDT-margined perpetual futures
                "adjustForTimeDifference": True,
            }

            if config.testnet:
                # CCXT uses testnet when sandbox=True
                exchange_options["sandbox"] = True
                endpoint_url = "testnet.bybit.com (TESTNET)"
            else:
                endpoint_url = "api.bybit.com (LIVE PRODUCTION)"

            # Create exchange instance
            self.logger.debug(f"Getting CCXT exchange class: {config.name}")
            try:
                exchange_class = getattr(ccxt, config.name)
                self.logger.debug(f"Successfully got exchange class: {exchange_class}")
            except AttributeError as e:
                self.logger.error(f"CCXT exchange '{config.name}' not found: {e}")
                raise
            
            # Strip and validate credentials (strip again as safety measure)
            api_key = (config.api_key or "").strip()
            api_secret = (config.api_secret or "").strip()
            
            # Log credential status (without exposing actual values)
            api_key_present = bool(api_key)
            api_secret_present = bool(api_secret)
            key_len = len(api_key)
            secret_len = len(api_secret)
            
            # Log first/last few chars for debugging (without exposing full key)
            key_preview = f"{api_key[:3]}...{api_key[-3:]}" if len(api_key) >= 6 else ("***" if api_key else "MISSING")
            secret_preview = f"{api_secret[:3]}...{api_secret[-3:]}" if len(api_secret) >= 6 else ("***" if api_secret else "MISSING")
            
            # Critical warning if there's a mismatch
            if config.mode == "live" and config.testnet:
                self.logger.error(
                    "⚠️  CONFIGURATION MISMATCH DETECTED ⚠️\n"
                    "  Config has mode='live' but testnet=true!\n"
                    "  The bot will connect to TESTNET, but you likely want LIVE.\n"
                    "  Set testnet: false in config.yaml for live trading."
                )
            elif config.mode != "paper" and config.mode != "testnet" and not config.testnet:
                self.logger.warning(
                    "⚠️  Live production mode detected!\n"
                    "  Ensure your API keys are for LIVE Bybit (not testnet).\n"
                    "  Live API keys are created at bybit.com (not testnet.bybit.com)."
                )
            
            self.logger.info(
                f"Initializing CCXT exchange - Endpoint: {endpoint_url}\n"
                f"  API key: {api_key_present} ({key_len} chars, preview: {key_preview})\n"
                f"  Secret: {api_secret_present} ({secret_len} chars, preview: {secret_preview})\n"
                f"  Config testnet: {config.testnet}, mode: {config.mode}, sandbox: {exchange_options.get('sandbox', False)}"
            )
            
            if not api_key_present or not api_secret_present:
                error_msg = (
                    "API credentials are missing or empty in config!\n"
                    f"  API key present: {api_key_present} ({key_len} chars)\n"
                    f"  API secret present: {api_secret_present} ({secret_len} chars)\n"
                    "Ensure BYBIT_API_KEY and BYBIT_API_SECRET are set in .env file or config.yaml"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info("Creating CCXT exchange instance...")
            try:
                # Ensure credentials are not empty strings before passing to CCXT
                if not api_key or not api_secret:
                    error_msg = f"API credentials are empty after stripping (key_len={len(api_key)}, secret_len={len(api_secret)})"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                self.exchange = exchange_class(
                    {
                        "apiKey": api_key,
                        "secret": api_secret,
                        "enableRateLimit": True,
                        "options": exchange_options,
                    }
                )
                self.logger.info("CCXT exchange instance created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create CCXT exchange instance: {e}", exc_info=True)
                raise
            
            # Verify what CCXT received (just for debugging)
            self.logger.debug(
                f"CCXT exchange initialized - apiKey set: {bool(self.exchange.apiKey)}, "
                f"secret set: {bool(self.exchange.secret)}, "
                f"sandbox: {self.exchange.options.get('sandbox', False)}"
            )

            # Retry configuration (could be made configurable via ExchangeConfig)
            self.max_retries: int = 3
            self.base_retry_delay: float = 1.0  # seconds

            # Paper mode flag
            self.paper_mode = config.mode == "paper"

            if self.paper_mode:
                self.logger.warning("Running in PAPER MODE - no real orders will be placed")

            self.logger.info(
                f"Initialized Bybit client (mode: {config.mode}, testnet: {config.testnet})"
            )
        except Exception as e:
            self.logger.error(f"CRITICAL: Failed to initialize BybitClient: {e}", exc_info=True)
            raise

    def _call_with_retries(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Call an exchange function with basic retry/backoff.

        Retries on transient network / timeout errors, fails fast on auth/config errors.
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RequestTimeout) as e:
                # Transient error: retry with exponential backoff
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"Max retries reached for {func.__name__}: {e}. Giving up."
                    )
                    raise
                delay = self.base_retry_delay * (2 ** attempt) * (1.0 + random() * 0.2)
                self.logger.warning(
                    f"Transient error in {func.__name__} (attempt {attempt+1}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            except ccxt.AuthenticationError as e:
                # Fatal: bad API key/secret or permissions
                self.logger.error(f"Authentication error in {func.__name__}: {e}")
                raise
            except ccxt.BadSymbol as e:
                # Symbol doesn't exist (delisted, inactive, etc.) - not an error, just log at debug
                self.logger.debug(f"Symbol not available in {func.__name__}: {e}")
                raise  # Re-raise so caller can handle it appropriately
            except ccxt.ExchangeError as e:
                # Non-retriable exchange error (e.g., invalid order)
                self.logger.error(f"Exchange error in {func.__name__}: {e}")
                raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List]:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            since: Timestamp in milliseconds (optional)
            limit: Number of candles to fetch (optional)
        
        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        try:
            # CCXT uses '/' separator and :USDT suffix for perpetual futures
            # Handle both formats: "BTCUSDT" -> "BTC/USDT:USDT" for perpetual futures
            if "/" not in symbol:
                # Convert "BASEUSDT" to "BASE/USDT:USDT" for perpetual futures
                if symbol.endswith("USDT"):
                    base = symbol[:-4]  # Remove "USDT" suffix
                    ccxt_symbol = f"{base}/USDT:USDT"  # Perpetual futures format
                else:
                    # Fallback: try adding "/USDT:USDT" if no slash
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                # Add :USDT suffix if missing (for perpetual futures)
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol

            # Respect rate limit
            time.sleep(self.exchange.rateLimit / 1000)

            ohlcv = self._call_with_retries(
                self.exchange.fetch_ohlcv,
                ccxt_symbol,
                timeframe,
                since=since,
                limit=limit,
            )

            self.logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
            return ohlcv

        except ccxt.BadSymbol:
            # Symbol doesn't exist - already logged at debug level in _call_with_retries
            # Re-raise so downloader can handle it gracefully
            raise
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection and validate API credentials.
        
        Returns:
            True if credentials are valid, False otherwise
            
        Raises:
            AuthenticationError: If credentials are invalid or missing
        """
        try:
            if self.paper_mode:
                # No credentials needed in paper mode
                return True
            
            # Check if credentials are provided
            if not self.config.api_key or not self.config.api_secret:
                raise ccxt.AuthenticationError(
                    "API credentials are missing. Please set BYBIT_API_KEY and BYBIT_API_SECRET "
                    "in your .env file or config.yaml"
                )
            
            # Debug: Log what we're about to send to CCXT
            sandbox_mode = self.exchange.options.get('sandbox', False)
            endpoint = "testnet.bybit.com" if sandbox_mode else "api.bybit.com (LIVE)"
            
            self.logger.info(
                f"Testing connection to {endpoint}\n"
                f"  API key length: {len(self.config.api_key)}, Secret length: {len(self.config.api_secret)}\n"
                f"  Config testnet: {self.config.testnet}, mode: {self.config.mode}, sandbox: {sandbox_mode}\n"
                f"  CCXT apiKey set: {bool(getattr(self.exchange, 'apiKey', None))}, "
                f"CCXT secret set: {bool(getattr(self.exchange, 'secret', None))}"
            )
            
            # Final warning before API call
            if sandbox_mode and self.config.mode == "live":
                self.logger.error(
                    "⚠️  CRITICAL: Connecting to TESTNET but mode is 'live'!\n"
                    "  Your live API keys will be rejected by the testnet endpoint.\n"
                    "  Set testnet: false in config.yaml and restart."
                )
            
            # Test with a simple API call that requires authentication
            # For Bybit unified account (v5), use params dict correctly
            time.sleep(self.exchange.rateLimit / 1000)
            
            # CCXT fetch_balance for Bybit takes params as a keyword argument
            # The 'type' parameter should be passed via params, not directly
            self.logger.info("Calling fetch_balance with params={'type': 'spot'} for testnet compatibility...")
            
            # Use _call_with_retries but pass params correctly as keyword argument
            try:
                # Try spot first (works on both testnet and live for unified account)
                balance = self._call_with_retries(self.exchange.fetch_balance, params={'type': 'spot'})
            except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                # If spot fails, try swap for derivatives account
                self.logger.debug(f"Spot balance fetch failed: {e}, trying swap...")
                balance = self._call_with_retries(self.exchange.fetch_balance, params={'type': 'swap'})
            
            self.logger.info("Connection test successful - API credentials are valid")
            return True
            
        except ccxt.AuthenticationError as e:
            error_msg = (
                f"Authentication failed: {e}\n"
                "Please verify:\n"
                "  1. BYBIT_API_KEY and BYBIT_API_SECRET are set in .env file\n"
                "  2. API keys are valid and not expired\n"
                "  3. API keys have proper permissions (read account, trade)\n"
                "  4. If using testnet, ensure keys are for testnet environment"
            )
            self.logger.error(error_msg)
            raise ccxt.AuthenticationError(error_msg) from e
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise
    
    def fetch_balance(self) -> Dict:
        """
        Fetch account balance.
        
        Returns:
            Balance dictionary with 'USDT' and other assets
        """
        try:
            if self.paper_mode:
                # Return mock balance in paper mode
                return {
                    "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
                    "info": {},
                }
            
            # Debug logging before API call
            self.logger.debug(
                f"fetch_balance called - Testnet: {self.config.testnet}, "
                f"Sandbox option: {self.exchange.options.get('sandbox', False)}, "
                f"API key present: {bool(getattr(self.exchange, 'apiKey', None))}"
            )
            
            time.sleep(self.exchange.rateLimit / 1000)
            
            # For Bybit unified margin, we want the swap account balance
            # CCXT's fetch_balance for Bybit accepts params dict
            # Use 'spot' for unified account, or try 'swap' for perps
            try:
                # Try unified account first (works for both spot and derivatives)
                balance = self._call_with_retries(self.exchange.fetch_balance, params={'type': 'spot'})
            except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                # If spot fails, try swap for derivatives account
                self.logger.debug(f"Spot balance fetch failed: {e}, trying swap...")
                balance = self._call_with_retries(self.exchange.fetch_balance, params={'type': 'swap'})
            
            return balance
            
        except ccxt.AuthenticationError as e:
            self.logger.error(
                f"Authentication error in fetch_balance: {e}\n"
                "This may indicate your API credentials expired or were revoked."
            )
            raise
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch open positions.
        
        Args:
            symbol: Optional symbol to filter by
        
        Returns:
            List of position dictionaries
        """
        try:
            if self.paper_mode:
                # Return empty positions in paper mode
                return []
            
            time.sleep(self.exchange.rateLimit / 1000)
            positions = self._call_with_retries(
                self.exchange.fetch_positions,
                [symbol] if symbol else None,
            )
            
            # Filter to only open positions (contracts > 0)
            open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]
            
            return open_positions
            
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            raise
    
    def fetch_open_orders(self, symbol: Optional[str] = None, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch open orders (including stop-loss and take-profit orders).
        
        Args:
            symbol: Optional symbol to filter by
            params: Optional parameters to pass to CCXT (e.g., {'category': 'linear', 'orderFilter': 'Stop'})
        
        Returns:
            List of open order dictionaries
        """
        try:
            if self.paper_mode:
                # Return empty orders in paper mode
                return []
            
            # Convert symbol format for CCXT if needed
            ccxt_symbol = None
            if symbol:
                if "/" not in symbol:
                    if symbol.endswith("USDT"):
                        base = symbol[:-4]
                        ccxt_symbol = f"{base}/USDT:USDT"
                    else:
                        ccxt_symbol = f"{symbol}/USDT:USDT"
                elif ":USDT" not in symbol:
                    ccxt_symbol = f"{symbol}:USDT"
                else:
                    ccxt_symbol = symbol
            
            time.sleep(self.exchange.rateLimit / 1000)
            
            # Fetch open orders (CCXT method handles conditional orders)
            # If params are provided, pass them through to CCXT
            if params:
                orders = self._call_with_retries(
                    self.exchange.fetch_open_orders,
                    ccxt_symbol if ccxt_symbol else None,
                    params=params
                )
            else:
                orders = self._call_with_retries(
                    self.exchange.fetch_open_orders,
                    ccxt_symbol if ccxt_symbol else None,
                )
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error fetching open orders for {symbol}: {e}")
            raise
    
    def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch a single order by ID.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            params: Optional parameters to pass to CCXT (e.g., {'category': 'linear', 'orderFilter': 'Stop'})
        
        Returns:
            Order dictionary
        """
        try:
            if self.paper_mode:
                # Return mock order in paper mode
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'status': 'open',
                    'type': 'stop_market',
                    'side': 'sell',
                    'amount': 0.0,
                    'price': None,
                    'triggerPrice': None,
                    'info': {}
                }
            
            # Convert symbol format for CCXT
            ccxt_symbol = None
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    ccxt_symbol = f"{base}/USDT:USDT"
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol
            
            time.sleep(self.exchange.rateLimit / 1000)
            
            # Call CCXT's fetch_order with optional params
            if params:
                order = self._call_with_retries(
                    self.exchange.fetch_order,
                    order_id,
                    ccxt_symbol,
                    params
                )
            else:
                order = self._call_with_retries(
                    self.exchange.fetch_order,
                    order_id,
                    ccxt_symbol
                )
            
            return order
            
        except ccxt.OrderNotFound as e:
            # Order not found - re-raise so caller can handle it
            self.logger.debug(f"Order {order_id} not found for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error fetching order {order_id} for {symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching order {order_id}: {e}")
            raise
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Funding rate dictionary with 'fundingRate' and 'nextFundingTime'
        """
        try:
            # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT:USDT for perpetual futures)
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    ccxt_symbol = f"{base}/USDT:USDT"
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol

            time.sleep(self.exchange.rateLimit / 1000)
            funding = self._call_with_retries(self.exchange.fetch_funding_rate, ccxt_symbol)

            return funding
            
        except Exception as e:
            self.logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            # Return default if error
            return {"fundingRate": 0.0, "nextFundingTime": None}
    
    def create_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Create an order.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            amount: Order size in contracts
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            params: Additional parameters
        
        Returns:
            Order dictionary
        """
        if self.paper_mode:
            self.logger.info(
                f"[PAPER] Would create {order_type} {side} order: "
                f"{symbol} {amount} @ {price if price else 'market'}"
            )
            return {
                "id": f"paper_{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "type": order_type,
                "status": "closed",
                "timestamp": int(time.time() * 1000),
                "info": {},
            }
        
        try:
            # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT:USDT for perpetual futures)
            # For Bybit perpetual futures, we need to use the :USDT suffix
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]  # Remove "USDT" suffix
                    ccxt_symbol = f"{base}/USDT:USDT"  # Perpetual futures format
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                # Add :USDT suffix if missing (for perpetual futures)
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol

            time.sleep(self.exchange.rateLimit / 1000)
            
            # CRITICAL: Explicitly specify market type for Bybit
            # Bybit unified account needs explicit 'type' parameter to avoid defaulting to spot
            order_params = params or {}
            order_params['type'] = 'swap'  # Force swap/perpetual futures, not spot
            
            if order_type == "limit" and price is None:
                raise ValueError("Price required for limit orders")

            self.logger.debug(
                f"Creating {order_type} {side} order for {symbol} (CCXT symbol: {ccxt_symbol}, "
                f"market type: swap/perpetual)"
            )

            order = self._call_with_retries(
                self.exchange.create_order,
                ccxt_symbol,
                order_type,
                side,
                amount,
                price,
                order_params,
            )

            self.logger.info(
                f"Created {order_type} {side} order: {symbol} {amount} @ {price if price else 'market'}"
            )
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order for {symbol}: {e}")
            raise
    
    def create_stop_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell' (opposite of position side)
        amount: float,
        trigger_price: float,
        order_type: str = "market",  # "market" or "limit"
        limit_price: Optional[float] = None,
        reduce_only: bool = True,
        current_price: Optional[float] = None,  # Optional current price for validation
    ) -> Dict:
        """
        Create a stop-loss or stop-entry order (conditional order).
        
        For Bybit v5, this uses conditional orders via params:
        - stopOrder: conditional orders
        - triggerPrice: price that triggers the order
        - orderLinkId: optional custom order ID
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell' (direction of stop order)
            amount: Order size in contracts
            trigger_price: Price that triggers the stop order
            order_type: "market" (stop-market) or "limit" (stop-limit)
            limit_price: Limit price (required if order_type="limit")
            reduce_only: If True, only reduce position (for stop-loss)
        
        Returns:
            Order dictionary with order ID
        """
        if self.paper_mode:
            self.logger.info(
                f"[PAPER] Would create stop {order_type} {side} order: "
                f"{symbol} {amount} @ trigger={trigger_price}, limit={limit_price}"
            )
            return {
                "id": f"paper_stop_{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": limit_price or trigger_price,
                "type": f"stop_{order_type}",
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": {},
            }
        
        try:
            # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT:USDT for perpetual futures)
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    ccxt_symbol = f"{base}/USDT:USDT"
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol
            
            # Fetch current price if not provided (for triggerDirection validation)
            if current_price is None:
                try:
                    ticker = self.fetch_ticker(symbol)
                    current_price = ticker.get('last')
                    if current_price:
                        self.logger.debug(f"Fetched current price for {symbol}: {current_price}")
                except Exception as e:
                    self.logger.warning(f"Could not fetch current price for {symbol}: {e}")
                    current_price = None
            
            time.sleep(self.exchange.rateLimit / 1000)
            
            # Bybit v5 conditional order params
            # CCXT may support this via params or may need direct API call
            # For now, we'll use CCXT's create_order with conditional order params
            # NOTE: CCXT likely won't support this properly, so we'll fall back to direct API
            order_params = {
                "stopPrice": trigger_price,  # Trigger price
                "stopLossPrice": trigger_price,  # Alternative field name
                "reduceOnly": reduce_only,
            }
            
            if order_type == "limit":
                if limit_price is None:
                    raise ValueError("limit_price required for stop-limit orders")
                order_params["price"] = limit_price
                order_params["timeInForce"] = "GTC"  # Good till cancel
            
            # Try using CCXT's create_order with conditional order type
            # If CCXT doesn't support it directly, we may need to use exchange-specific API
            # NOTE: This is EXPECTED to fail - CCXT doesn't support Bybit conditional orders properly
            # We'll catch the error and fall back to direct Bybit v5 API (this is normal, not an error)
            try:
                order = self._call_with_retries(
                    self.exchange.create_order,
                    ccxt_symbol,
                    f"stop_{order_type}",  # e.g., "stop_market", "stop_limit"
                    side,
                    amount,
                    None,  # price (handled in params)
                    order_params,
                )
            except (ccxt.ExchangeError, ccxt.NotSupported, Exception) as e:
                # CCXT may not support conditional orders directly - this is EXPECTED behavior
                # Fall back to using Bybit v5 API directly (this is normal, not an error condition)
                # Use WARNING level since this is expected and handled gracefully
                error_msg = str(e)
                self.logger.debug(
                    f"CCXT doesn't support conditional stop orders directly for {symbol}: {error_msg}. "
                    "This is expected - falling back to direct Bybit v5 API call..."
                )
                # Use Bybit v5 API directly for conditional orders
                # Fall back to direct API call since CCXT doesn't support conditional orders
                order = self._create_stop_order_v5_api(
                    symbol, side, amount, trigger_price, order_type, 
                    limit_price, reduce_only, current_price
                )
                # If successful, log and return
                self.logger.info(
                    f"Created stop {order_type} {side} order via Bybit v5 API: {symbol} {amount} @ "
                    f"trigger={trigger_price}, order_id={order.get('id', 'N/A')}"
                )
                return order
            
            # If CCXT succeeded, log and return
            self.logger.info(
                f"Created stop {order_type} {side} order: {symbol} {amount} @ "
                f"trigger={trigger_price}, order_id={order.get('id', 'N/A')}"
            )
            return order
            
        except Exception as e:
            # Only log ERROR if we truly can't handle it (fallback also failed)
            # Most errors should be caught above and handled via fallback to Bybit v5 API
            self.logger.error(f"Error creating stop order for {symbol} (fallback also failed): {e}")
            raise
    
    def _create_stop_order_v5_api(
        self,
        symbol: str,
        side: str,
        amount: float,
        trigger_price: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        reduce_only: bool = True,
        current_price: Optional[float] = None,  # For triggerDirection validation
    ) -> Dict:
        """
        Create stop order using Bybit v5 API directly (bypassing CCXT).
        
        This is used as a fallback when CCXT doesn't support conditional orders.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: 'buy' or 'sell'
            amount: Order size in contracts
            trigger_price: Trigger price
            order_type: "market" or "limit"
            limit_price: Limit price (for stop-limit orders)
            reduce_only: If True, only reduce position
        
        Returns:
            Order dictionary
        """
        # Check if requests is available
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' library is required for direct Bybit v5 API calls. "
                "Install it with: pip install requests"
            )
        
        # Convert symbol format (BTC/USDT or BTCUSDT -> BTCUSDT for Bybit)
        # Bybit v5 uses symbol without slash
        bybit_symbol = symbol.replace("/", "").replace(":USDT", "USDT")
        if not bybit_symbol.endswith("USDT"):
            bybit_symbol = f"{bybit_symbol}USDT"
        
        # Determine base URL
        if self.config.testnet:
            base_url = "https://api-testnet.bybit.com"
        else:
            base_url = "https://api.bybit.com"
        
        endpoint = "/v5/order/create"
        url = f"{base_url}{endpoint}"
        
        # Build request parameters
        timestamp = str(int(time.time() * 1000))
        
        # Determine triggerDirection based on stop order side:
        # - For SELL stop (closing long position): price falls → triggerDirection = 2
        # - For BUY stop (closing short position): price rises → triggerDirection = 1
        # CRITICAL: triggerDirection is REQUIRED for conditional stop orders in Bybit v5
        if side.lower() == 'sell':
            # Stop-loss for long position: triggers when price falls below trigger
            trigger_direction = 2  # Price falls to trigger price
        else:  # side.lower() == 'buy'
            # Stop-loss for short position: triggers when price rises above trigger
            trigger_direction = 1  # Price rises to trigger price
        
        # Optional validation: Check trigger price relative to current price
        if current_price is not None:
            if side.lower() == 'sell' and trigger_price >= current_price:
                self.logger.warning(
                    f"Stop-loss trigger price ({trigger_price}) should be below current price "
                    f"({current_price}) for SELL stop order. Proceeding anyway..."
                )
            elif side.lower() == 'buy' and trigger_price <= current_price:
                self.logger.warning(
                    f"Stop-loss trigger price ({trigger_price}) should be above current price "
                    f"({current_price}) for BUY stop order. Proceeding anyway..."
                )
        
        # Build parameters matching Bybit v5 API format
        # CRITICAL: All numeric values (qty, triggerPrice, price) must be strings
        # Boolean values (reduceOnly) must be boolean (JSON will serialize as true/false)
        # Integer values (positionIdx, triggerDirection) must be integers (JSON will serialize as numbers)
        params = {
            "category": "linear",  # USDT-margined perpetuals
            "symbol": bybit_symbol,
            "side": side.capitalize(),  # Buy or Sell (capitalized)
            "orderType": "Market" if order_type == "market" else "Limit",  # Capitalized
            "qty": str(amount),  # Must be string
            "positionIdx": 0,  # One-way mode (integer, 0 = One-way, 1 = Buy side hedge, 2 = Sell side hedge)
            "reduceOnly": bool(reduce_only),  # Boolean (will serialize as true/false)
            "stopOrderType": "StopMarket" if order_type == "market" else "StopLimit",  # Capitalized
            "triggerPrice": str(trigger_price),  # Must be string
            "triggerDirection": trigger_direction,  # REQUIRED: 1 = price rises, 2 = price falls
            "triggerBy": "LastPrice",  # Optional but recommended: LastPrice, MarkPrice, or IndexPrice
        }
        
        if order_type == "limit" and limit_price:
            params["price"] = str(limit_price)  # Must be string
        
        # Log parameters for debugging (without exposing full amounts)
        self.logger.debug(
            f"Bybit v5 stop order request: "
            f"symbol={params['symbol']}, side={params['side']}, "
            f"stopOrderType={params['stopOrderType']}, "
            f"triggerPrice={params['triggerPrice']}, "
            f"triggerDirection={params['triggerDirection']} ({'price rises' if params['triggerDirection'] == 1 else 'price falls'}), "
            f"triggerBy={params['triggerBy']}, "
            f"qty={params['qty']}, reduceOnly={params['reduceOnly']}"
        )
        
        # Get credentials (ensure they're stripped, same as in __init__)
        api_key = (self.config.api_key or "").strip()
        api_secret = (self.config.api_secret or "").strip()
        
        if not api_key or not api_secret:
            raise ccxt.AuthenticationError("API credentials missing or empty in config")
        
        recv_window = "5000"
        
        # For Bybit v5 POST requests with JSON body:
        # Signature string = timestamp + api_key + recv_window + JSON_string
        # JSON string must have sorted keys and NO spaces (compact JSON)
        # CRITICAL: The JSON string used for signature MUST match exactly what's sent in the request body
        
        # Create compact JSON string (no spaces, sorted keys)
        # IMPORTANT: For Bybit v5 POST requests, the signature must use the EXACT JSON string
        # that will be sent in the request body. This string must:
        # - Have keys sorted alphabetically (sort_keys=True)
        # - Have NO spaces (separators=(',', ':'))
        # - Match EXACTLY what gets sent in the HTTP body
        # CRITICAL: Use ensure_ascii=True (default) to match Bybit's expectations
        # Use sort_keys=True to ensure consistent alphabetical ordering
        json_string = json.dumps(params, separators=(',', ':'), sort_keys=True, ensure_ascii=True)
        
        # Debug: Log the JSON string for troubleshooting (without exposing full secrets)
        self.logger.debug(f"Bybit v5 API request JSON (length={len(json_string)}): {json_string}")
        
        # Build signature string: timestamp + api_key + recv_window + json_string
        # CRITICAL: Use explicit string concatenation to avoid any f-string formatting issues
        # All components must be strings
        param_str = str(timestamp) + str(api_key) + str(recv_window) + json_string
        
        self.logger.debug(
            f"Signature string components: "
            f"timestamp={timestamp} (len={len(str(timestamp))}), "
            f"api_key_len={len(api_key)}, "
            f"recv_window={recv_window}, "
            f"json_len={len(json_string)}, "
            f"total_param_str_len={len(param_str)}"
        )
        
        # Generate signature (HMAC-SHA256)
        # CRITICAL: Both secret and param_str must be UTF-8 encoded bytes
        param_str_bytes = param_str.encode("utf-8")
        api_secret_bytes = api_secret.encode("utf-8")
        
        signature_bytes = hmac.new(
            api_secret_bytes,
            param_str_bytes,
            hashlib.sha256
        ).digest()
        signature = signature_bytes.hex()
        
        self.logger.debug(f"Generated signature (hex, len={len(signature)}): {signature[:16]}...{signature[-8:]}")
        
        # Prepare headers
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",  # SHA256
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        # Make request
        # CRITICAL: Send the exact same JSON string used in signature calculation
        # Use data= with the JSON string directly, not json= which auto-serializes
        try:
            response = requests.post(url, data=json_string.encode('utf-8'), headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("retCode") != 0:
                error_code = result.get("retCode", "unknown")
                error_msg = result.get("retMsg", "Unknown error")
                self.logger.error(
                    f"Bybit v5 API error (code {error_code}): {error_msg}\n"
                    f"Request params: {json.dumps(params, indent=2)}"
                )
                
                # Enhanced error logging for signature errors
                if "error sign" in error_msg.lower() or error_code == 10004:
                    self.logger.error(
                        f"Signature error details:\n"
                        f"  Error code: {error_code}\n"
                        f"  Error message: {error_msg}\n"
                        f"  Request URL: {url}\n"
                        f"  Request JSON: {json_string}\n"
                        f"  Timestamp: {timestamp}\n"
                        f"  Recv window: {recv_window}\n"
                        f"  API key length: {len(api_key)}\n"
                        f"  Signature string length: {len(param_str)}\n"
                        f"  This indicates the signature calculation doesn't match Bybit's expectations.\n"
                        f"  Check that: timestamp + api_key + recv_window + json_string matches Bybit's format."
                    )
                
                raise ccxt.ExchangeError(f"Bybit API error (code {error_code}): {error_msg}")
            
            order_data = result.get("result", {})
            order_id = order_data.get("orderId", "")
            
            self.logger.info(
                f"Created stop order via Bybit v5 API: {symbol} {amount} @ "
                f"trigger={trigger_price}, order_id={order_id}"
            )
            
            # Return CCXT-compatible order structure
            return {
                "id": str(order_id),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": limit_price or trigger_price,
                "type": f"stop_{order_type}",
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": order_data,
            }
            
        except requests.RequestException as e:
            self.logger.error(f"HTTP error creating stop order: {e}")
            raise ccxt.ExchangeError(f"HTTP error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error in Bybit v5 API call: {e}")
            raise
    
    def create_take_profit_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell' (opposite of position side)
        amount: float,
        trigger_price: float,
        order_type: str = "limit",  # Usually limit for TP
        limit_price: Optional[float] = None,
        reduce_only: bool = True,
        current_price: Optional[float] = None,  # Optional current price for validation
    ) -> Dict:
        """
        Create a take-profit order (conditional limit order).
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell' (opposite of position side)
            amount: Order size in contracts
            trigger_price: Price that triggers the TP order
            order_type: "limit" (recommended) or "market"
            limit_price: Limit price (should be trigger_price for TP)
            reduce_only: If True, only reduce position
        
        Returns:
            Order dictionary with order ID
        """
        if self.paper_mode:
            self.logger.info(
                f"[PAPER] Would create take-profit {order_type} {side} order: "
                f"{symbol} {amount} @ trigger={trigger_price}"
            )
            return {
                "id": f"paper_tp_{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": limit_price or trigger_price,
                "type": f"take_profit_{order_type}",
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": {},
            }
        
        try:
            # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT:USDT for perpetual futures)
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    ccxt_symbol = f"{base}/USDT:USDT"
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol
            
            # Fetch current price if not provided (for triggerDirection validation)
            if current_price is None:
                try:
                    ticker = self.fetch_ticker(symbol)
                    current_price = ticker.get('last')
                    if current_price:
                        self.logger.debug(f"Fetched current price for {symbol} (TP): {current_price}")
                except Exception as e:
                    self.logger.warning(f"Could not fetch current price for {symbol} (TP): {e}")
                    current_price = None
            
            time.sleep(self.exchange.rateLimit / 1000)
            
            # For take-profit, use limit order at trigger price
            order_params = {
                "stopPrice": trigger_price,
                "takeProfitPrice": trigger_price,
                "reduceOnly": reduce_only,
            }
            
            if order_type == "limit":
                order_params["price"] = limit_price or trigger_price
                order_params["timeInForce"] = "GTC"
            
            try:
                order = self._call_with_retries(
                    self.exchange.create_order,
                    ccxt_symbol,
                    "take_profit" if order_type == "limit" else "take_profit_market",
                    side,
                    amount,
                    limit_price or trigger_price,
                    order_params,
                )
            except (ccxt.ExchangeError, ccxt.NotSupported) as e:
                # CCXT may not support take-profit orders directly
                # Fall back to using Bybit v5 API directly
                self.logger.warning(
                    f"CCXT doesn't support take-profit orders directly for {symbol}: {e}. "
                    "Attempting direct Bybit v5 API call..."
                )
                # Use Bybit v5 API directly for take-profit orders
                order = self._create_take_profit_order_v5_api(
                    symbol, side, amount, trigger_price, order_type, 
                    limit_price, reduce_only, current_price
                )
                # If successful, log and return
                self.logger.info(
                    f"Created take-profit {order_type} {side} order via Bybit v5 API: {symbol} {amount} @ "
                    f"trigger={trigger_price}, order_id={order.get('id', 'N/A')}"
                )
                return order
            
            # If CCXT succeeded, log and return
            self.logger.info(
                f"Created take-profit {order_type} {side} order: {symbol} {amount} @ "
                f"trigger={trigger_price}, order_id={order.get('id', 'N/A')}"
            )
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating take-profit order for {symbol}: {e}")
            raise
    
    def _create_take_profit_order_v5_api(
        self,
        symbol: str,
        side: str,
        amount: float,
        trigger_price: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        reduce_only: bool = True,
        current_price: Optional[float] = None,  # For triggerDirection validation
    ) -> Dict:
        """
        Create take-profit order using Bybit v5 API directly (bypassing CCXT).
        
        This is used as a fallback when CCXT doesn't support conditional orders.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: 'buy' or 'sell' (opposite of position side)
            amount: Order size in contracts
            trigger_price: Trigger price
            order_type: "limit" or "market"
            limit_price: Limit price (for take-profit limit orders)
            reduce_only: If True, only reduce position
            current_price: Optional current market price (for triggerDirection calculation)
        
        Returns:
            Order dictionary
        """
        # Check if requests is available
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' library is required for direct Bybit v5 API calls. "
                "Install it with: pip install requests"
            )
        
        # Convert symbol format (BTC/USDT or BTCUSDT -> BTCUSDT for Bybit)
        bybit_symbol = symbol.replace("/", "").replace(":USDT", "USDT")
        if not bybit_symbol.endswith("USDT"):
            bybit_symbol = f"{bybit_symbol}USDT"
        
        # Determine base URL
        if self.config.testnet:
            base_url = "https://api-testnet.bybit.com"
        else:
            base_url = "https://api.bybit.com"
        
        endpoint = "/v5/order/create"
        url = f"{base_url}{endpoint}"
        
        # Build request parameters
        timestamp = str(int(time.time() * 1000))
        
        # Determine triggerDirection based on take-profit order side:
        # - For SELL TP (closing long position): price rises → triggerDirection = 1
        # - For BUY TP (closing short position): price falls → triggerDirection = 2
        # CRITICAL: triggerDirection is REQUIRED for conditional take-profit orders in Bybit v5
        if side.lower() == 'sell':
            # Take-profit for long position: triggers when price rises above trigger
            trigger_direction = 1  # Price rises to trigger price
        else:  # side.lower() == 'buy'
            # Take-profit for short position: triggers when price falls below trigger
            trigger_direction = 2  # Price falls to trigger price
        
        # Optional validation: Check trigger price relative to current price
        if current_price is not None:
            if side.lower() == 'sell' and trigger_price <= current_price:
                self.logger.warning(
                    f"Take-profit trigger price ({trigger_price}) should be above current price "
                    f"({current_price}) for SELL take-profit order. Proceeding anyway..."
                )
            elif side.lower() == 'buy' and trigger_price >= current_price:
                self.logger.warning(
                    f"Take-profit trigger price ({trigger_price}) should be below current price "
                    f"({current_price}) for BUY take-profit order. Proceeding anyway..."
                )
        
        # Build parameters matching Bybit v5 API format
        params = {
            "category": "linear",  # USDT-margined perpetuals
            "symbol": bybit_symbol,
            "side": side.capitalize(),  # Buy or Sell (capitalized)
            "orderType": "Market" if order_type == "market" else "Limit",  # Capitalized
            "qty": str(amount),  # Must be string
            "positionIdx": 0,  # One-way mode
            "reduceOnly": bool(reduce_only),  # Boolean
            "stopOrderType": "TakeProfitMarket" if order_type == "market" else "TakeProfit",  # Capitalized
            "triggerPrice": str(trigger_price),  # Must be string
            "triggerDirection": trigger_direction,  # REQUIRED: 1 = price rises, 2 = price falls
            "triggerBy": "LastPrice",  # Optional but recommended
        }
        
        if order_type == "limit":
            if limit_price is None:
                limit_price = trigger_price
            params["price"] = str(limit_price)  # Must be string
            params["timeInForce"] = "GTC"  # Good till cancel
        
        # Log full params for debugging (redact sensitive data)
        self.logger.debug(
            f"Bybit v5 take-profit order request: "
            f"symbol={params['symbol']}, side={params['side']}, "
            f"stopOrderType={params['stopOrderType']}, "
            f"triggerPrice={params['triggerPrice']}, "
            f"triggerDirection={params['triggerDirection']} ({'price rises' if params['triggerDirection'] == 1 else 'price falls'}), "
            f"triggerBy={params['triggerBy']}, "
            f"qty={params['qty']}, reduceOnly={params['reduceOnly']}"
        )
        
        # Get credentials
        api_key = (self.config.api_key or "").strip()
        api_secret = (self.config.api_secret or "").strip()
        
        if not api_key or not api_secret:
            raise ccxt.AuthenticationError("API credentials missing or empty in config")
        
        recv_window = "5000"
        
        # Create compact JSON string (no spaces, sorted keys)
        json_string = json.dumps(params, separators=(',', ':'), sort_keys=True, ensure_ascii=True)
        
        self.logger.debug(f"Bybit v5 API request JSON (TP, length={len(json_string)}): {json_string}")
        
        # Build signature string: timestamp + api_key + recv_window + json_string
        param_str = str(timestamp) + str(api_key) + str(recv_window) + json_string
        
        # Generate signature (HMAC-SHA256)
        param_str_bytes = param_str.encode("utf-8")
        api_secret_bytes = api_secret.encode("utf-8")
        
        signature_bytes = hmac.new(
            api_secret_bytes,
            param_str_bytes,
            hashlib.sha256
        ).digest()
        signature = signature_bytes.hex()
        
        # Prepare headers
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",  # SHA256
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        # Make request
        try:
            response = requests.post(url, data=json_string.encode('utf-8'), headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("retCode") != 0:
                error_code = result.get("retCode", "unknown")
                error_msg = result.get("retMsg", "Unknown error")
                self.logger.error(
                    f"Bybit v5 API error (code {error_code}): {error_msg}\n"
                    f"Request params: {json.dumps(params, indent=2)}"
                )
                raise ccxt.ExchangeError(f"Bybit API error (code {error_code}): {error_msg}")
            
            order_data = result.get("result", {})
            order_id = order_data.get("orderId", "")
            
            self.logger.info(
                f"Created take-profit order via Bybit v5 API: {symbol} {amount} @ "
                f"trigger={trigger_price}, order_id={order_id}"
            )
            
            # Return CCXT-compatible order structure
            return {
                "id": str(order_id),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": limit_price or trigger_price,
                "type": f"take_profit_{order_type}",
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": order_data,
            }
            
        except requests.RequestException as e:
            self.logger.error(f"HTTP error creating take-profit order: {e}")
            raise ccxt.ExchangeError(f"HTTP error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error in Bybit v5 API call (TP): {e}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """
        Cancel an order (including conditional orders like stop-loss/take-profit).
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., 'BTCUSDT' or 'BTC/USDT:USDT')
            params: Optional parameters for conditional orders (e.g., {'category': 'linear', 'orderFilter': 'Stop'})
        
        Returns:
            Cancelled order dictionary
        """
        if self.paper_mode:
            self.logger.info(f"[PAPER] Would cancel order {order_id} for {symbol}")
            return {"id": order_id, "status": "canceled"}
        
        try:
            # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT:USDT for perpetual futures)
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]  # Remove "USDT" suffix
                    ccxt_symbol = f"{base}/USDT:USDT"  # Perpetual futures format
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol

            time.sleep(self.exchange.rateLimit / 1000)
            
            # Try cancelling via CCXT first (with params if provided for conditional orders)
            try:
                if params:
                    result = self._call_with_retries(
                        self.exchange.cancel_order,
                        order_id,
                        ccxt_symbol,
                        params=params
                    )
                else:
                    result = self._call_with_retries(
                        self.exchange.cancel_order,
                        order_id,
                        ccxt_symbol,
                    )
                
                self.logger.info(f"Cancelled order {order_id} for {symbol}")
                return result
                
            except (ccxt.OrderNotFound, ccxt.ExchangeError) as ccxt_error:
                # CCXT might not support conditional order cancellation - try direct Bybit v5 API
                error_msg = str(ccxt_error)
                if "Order does not exist" in error_msg or "not found" in error_msg.lower():
                    # Order might already be cancelled - try direct API to be sure
                    self.logger.debug(
                        f"CCXT cancellation returned 'not found' for {order_id}. "
                        f"Trying direct Bybit v5 API to verify..."
                    )
                else:
                    # Other error - try direct API as fallback
                    self.logger.debug(
                        f"CCXT cancellation failed for {order_id}: {ccxt_error}. "
                        f"Trying direct Bybit v5 API as fallback..."
                    )
                
                # Fallback to direct Bybit v5 API for conditional orders
                if REQUESTS_AVAILABLE:
                    try:
                        return self._cancel_conditional_order_v5_api(order_id, symbol, params)
                    except Exception as v5_error:
                        # If direct API also fails, re-raise the original CCXT error
                        self.logger.error(
                            f"Both CCXT and direct Bybit v5 API failed to cancel order {order_id}: "
                            f"CCXT={ccxt_error}, v5={v5_error}"
                        )
                        raise ccxt_error
                else:
                    # No direct API fallback available - re-raise CCXT error
                    raise ccxt_error
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
    def _cancel_conditional_order_v5_api(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """
        Cancel a conditional order (stop-loss/take-profit) using direct Bybit v5 API.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            params: Optional parameters (will default to category='linear', orderFilter='Stop')
        
        Returns:
            Cancelled order dictionary
        """
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library not available for direct Bybit v5 API calls")
        
        # Default params for conditional orders
        if params is None:
            params = {}
        
        # Ensure required params for Bybit v5
        api_params = {
            'category': params.get('category', 'linear'),
            'orderFilter': params.get('orderFilter', 'Stop'),  # 'Stop' for conditional orders
            'orderLinkId': order_id  # Try orderLinkId first
        }
        
        # If orderLinkId doesn't work, we'll try orderId
        # For now, try with orderLinkId
        api_params['orderLinkId'] = order_id
        
        # Convert symbol to Bybit format (BTCUSDT or BTC/USDT:USDT -> BTCUSDT)
        bybit_symbol = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
        
        url = f"{self.exchange.urls['api']}/v5/order/cancel"
        if self.config.testnet:
            url = url.replace("api.bybit.com", "api-testnet.bybit.com")
        
        timestamp = str(int(time.time() * 1000))
        query_string = f"category={api_params['category']}&orderFilter={api_params['orderFilter']}&orderLinkId={order_id}&symbol={bybit_symbol}&timestamp={timestamp}"
        
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-BAPI-API-KEY': self.config.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, params=api_params, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('retCode') == 0:
                self.logger.info(f"Cancelled conditional order {order_id} for {symbol} via Bybit v5 API")
                return {"id": order_id, "status": "canceled", "info": result}
            elif result.get('retCode') == 170213:  # Order does not exist
                raise ccxt.OrderNotFound(f"Order {order_id} does not exist (already cancelled?)")
            else:
                error_msg = result.get('retMsg', 'Unknown error')
                # If orderLinkId failed, try with orderId
                if 'orderLinkId' in error_msg.lower() or result.get('retCode') in [170011, 170003]:
                    self.logger.debug(f"orderLinkId failed for {order_id}, trying orderId...")
                    api_params.pop('orderLinkId', None)
                    api_params['orderId'] = order_id
                    
                    query_string = f"category={api_params['category']}&orderFilter={api_params['orderFilter']}&orderId={order_id}&symbol={bybit_symbol}&timestamp={timestamp}"
                    signature = hmac.new(
                        self.config.api_secret.encode('utf-8'),
                        query_string.encode('utf-8'),
                        hashlib.sha256
                    ).hexdigest()
                    headers['X-BAPI-SIGN'] = signature
                    
                    response = requests.post(url, params=api_params, headers=headers, timeout=10)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get('retCode') == 0:
                        self.logger.info(f"Cancelled conditional order {order_id} for {symbol} via Bybit v5 API (using orderId)")
                        return {"id": order_id, "status": "canceled", "info": result}
                    elif result.get('retCode') == 170213:
                        raise ccxt.OrderNotFound(f"Order {order_id} does not exist (already cancelled?)")
                    else:
                        raise ccxt.ExchangeError(f"Bybit v5 API error (code {result.get('retCode')}): {result.get('retMsg')}")
                else:
                    raise ccxt.ExchangeError(f"Bybit v5 API error (code {result.get('retCode')}): {error_msg}")
                    
        except requests.RequestException as e:
            raise ccxt.NetworkError(f"Network error cancelling conditional order via Bybit v5 API: {e}")
        except ccxt.OrderNotFound:
            raise
        except Exception as e:
            raise ccxt.ExchangeError(f"Error cancelling conditional order via Bybit v5 API: {e}")
    
    def fetch_markets(self) -> Dict:
        """
        Fetch market information (precision, limits, etc.).
        
        Returns:
            Markets dictionary
        """
        try:
            time.sleep(self.exchange.rateLimit / 1000)
            markets = self.exchange.load_markets()
            return markets
            
        except Exception as e:
            self.logger.error(f"Error fetching markets: {e}")
            raise
    
    def get_market_info(self, symbol: str) -> Dict:
        """
        Get market information for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Market info dictionary with precision, limits, etc.
        """
        try:
            # Convert internal symbol format to CCXT format for perpetual futures
            # Internal: "ETHUSDT" → CCXT: "ETH/USDT:USDT" (perpetual futures, not spot)
            if "/" not in symbol:
                if symbol.endswith("USDT"):
                    base = symbol[:-4]  # Remove "USDT" suffix
                    ccxt_symbol = f"{base}/USDT:USDT"  # Perpetual futures format
                else:
                    ccxt_symbol = f"{symbol}/USDT:USDT"
            elif ":USDT" not in symbol:
                ccxt_symbol = f"{symbol}:USDT"
            else:
                ccxt_symbol = symbol
            
            markets = self.fetch_markets()
            
            if ccxt_symbol not in markets:
                self.logger.warning(
                    f"Symbol {symbol} (CCXT: {ccxt_symbol}) not found in markets. "
                    f"Available symbols (first 10): {list(markets.keys())[:10]}"
                )
                raise ValueError(f"Symbol {symbol} (CCXT: {ccxt_symbol}) not found in markets")
            
            market = markets[ccxt_symbol]
            
            market_info = {
                'precision': {
                    'price': market.get('precision', {}).get('price', 2),
                    'amount': market.get('precision', {}).get('amount', 8),
                },
                'limits': {
                    'amount': {
                        'min': market.get('limits', {}).get('amount', {}).get('min', 0),
                        'max': market.get('limits', {}).get('amount', {}).get('max', None),
                    },
                    'cost': {
                        'min': market.get('limits', {}).get('cost', {}).get('min', 0),
                    },
                },
                'contractSize': market.get('contractSize') or 1.0,
            }
            
            self.logger.debug(
                f"Market info for {symbol} (CCXT: {ccxt_symbol}): "
                f"min_amount={market_info['limits']['amount']['min']}, "
                f"min_cost={market_info['limits']['cost']['min']}, "
                f"precision_amount={market_info['precision']['amount']}, "
                f"contract_size={market_info['contractSize']}"
            )
            
            return market_info
            
        except Exception as e:
            self.logger.error(f"Error getting market info for {symbol}: {e}")
            # Return safe defaults
            return {
                'precision': {'price': 2, 'amount': 8},
                'limits': {'amount': {'min': 0.001, 'max': None}, 'cost': {'min': 5.0}},
                'contractSize': 1.0,
            }
    
    def _precision_to_decimals(self, precision: Any) -> int:
        """
        Convert precision (int or float/tick size) to number of decimal places.
        
        Args:
            precision: Can be an int (decimal places) or float (tick size like 0.1, 0.01)
        
        Returns:
            Integer number of decimal places
        """
        if isinstance(precision, int):
            return precision
        elif isinstance(precision, float):
            # If precision is a tick size (e.g., 0.1, 0.01, 0.001), convert to decimal places
            if precision >= 1.0:
                return int(precision)
            # Count decimal places by converting to string
            precision_str = str(precision).rstrip('0').rstrip('.')
            if '.' in precision_str:
                return len(precision_str.split('.')[1])
            return 0
        else:
            # Fallback: try to convert to int
            try:
                return int(precision)
            except (ValueError, TypeError):
                return 8  # Default to 8 decimal places for amounts
    
    def round_amount(self, symbol: str, amount: float) -> float:
        """
        Round amount to exchange precision.
        
        Also ensures the rounded amount meets the minimum order size.
        If rounding would result in an amount below minimum, rounds up to minimum.
        """
        if amount <= 0:
            return 0.0
        
        market_info = self.get_market_info(symbol)
        precision = market_info['precision']['amount']
        decimals = self._precision_to_decimals(precision)
        
        # Round to exchange precision
        rounded = round(amount, decimals)
        
        # Ensure rounded amount meets minimum order size
        # If rounding down would result in value below minimum, round up to minimum
        min_amount = market_info['limits']['amount']['min']
        if rounded < min_amount and amount >= min_amount:
            # Round up to minimum (or next valid increment above minimum)
            # Use the precision to find the next valid increment
            if decimals > 0:
                # Round up to minimum using precision
                # For example, if min is 0.01 and precision is 2 decimals, round up to 0.01
                multiplier = 10 ** decimals
                min_rounded = math.ceil(min_amount * multiplier) / multiplier
                # Ensure it's at least the minimum
                rounded = max(min_rounded, min_amount)
            else:
                # Integer precision - round up to next integer above minimum
                rounded = max(math.ceil(min_amount), min_amount)
        
        # Final check: ensure we don't return 0 if original amount was positive
        if rounded <= 0 and amount > 0:
            rounded = min_amount
        
        return rounded
    
    def round_price(self, symbol: str, price: float) -> float:
        """
        Round price to exchange precision.
        
        Ensures the result is never 0.0 if input price is > 0.
        """
        if price <= 0:
            return 0.0
        
        market_info = self.get_market_info(symbol)
        precision = market_info['precision']['price']
        decimals = self._precision_to_decimals(precision)
        
        rounded = round(price, decimals)
        
        # Safety check: if rounding results in 0.0 but input was > 0, use minimum viable price
        if rounded == 0.0 and price > 0:
            # Use the precision as tick size if it's a float, otherwise use 0.01 as minimum
            if isinstance(precision, float) and precision > 0:
                # Round up to the next tick
                tick_size = precision
                rounded = math.ceil(price / tick_size) * tick_size
                # Re-round to ensure proper decimal places
                decimals = self._precision_to_decimals(precision)
                rounded = round(rounded, decimals)
            else:
                # Default to at least 2 decimal places if precision is unclear
                rounded = round(price, max(2, decimals))
            
            # Final safety check: if still 0, use a small positive value based on price magnitude
            if rounded == 0.0:
                # Use 1/10th of the price magnitude, rounded to 2 decimals minimum
                magnitude = 10 ** (math.floor(math.log10(abs(price))) - 1)
                rounded = round(math.ceil(price / magnitude) * magnitude, max(2, decimals))
                self.logger.warning(
                    f"Price rounding for {symbol} resulted in 0.0 (input: {price}, precision: {precision}). "
                    f"Using fallback: {rounded}"
                )
        
        return rounded
    
    def validate_order_size(self, symbol: str, amount: float, price: float) -> Tuple[bool, Optional[str]]:
        """
        Validate order size meets exchange requirements.
        
        Returns:
            (is_valid, error_message)
        """
        market_info = self.get_market_info(symbol)
        limits = market_info['limits']
        contract_size = market_info['contractSize']
        
        # Check minimum amount
        min_amount = limits['amount']['min']
        if min_amount is not None and min_amount > 0:
            if amount < min_amount:
                return False, f"Amount {amount} below minimum {min_amount}"
        
        # Check minimum cost (notional)
        min_cost = limits['cost']['min']
        if min_cost is not None and min_cost > 0:
            cost = amount * price * contract_size
            if cost < min_cost:
                return False, f"Order cost {cost} below minimum {min_cost}"
        
        return True, None

