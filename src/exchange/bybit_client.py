"""Bybit exchange client wrapper using CCXT."""

import ccxt
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging
from random import random

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

        # Configure exchange
        exchange_options = {
            "defaultType": "swap",  # USDT-margined perpetual futures
            "adjustForTimeDifference": True,
        }

        if config.testnet:
            # CCXT uses testnet when sandbox=True
            exchange_options["sandbox"] = True

        # Create exchange instance
        exchange_class = getattr(ccxt, config.name)
        self.exchange = exchange_class(
            {
                "apiKey": config.api_key or "",
                "secret": config.api_secret or "",
                "enableRateLimit": True,
                "options": exchange_options,
            }
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
            # CCXT uses '/' separator
            ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol

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
            
            # Test with a simple API call that requires authentication
            time.sleep(self.exchange.rateLimit / 1000)
            self._call_with_retries(self.exchange.fetch_balance, {"type": "swap"})
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
            
            time.sleep(self.exchange.rateLimit / 1000)
            balance = self._call_with_retries(self.exchange.fetch_balance, {"type": "swap"})
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
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Funding rate dictionary with 'fundingRate' and 'nextFundingTime'
        """
        try:
            ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol

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
            ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol

            time.sleep(self.exchange.rateLimit / 1000)
            
            order_params = params or {}
            if order_type == "limit" and price is None:
                raise ValueError("Price required for limit orders")

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
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
        
        Returns:
            Cancelled order dictionary
        """
        if self.paper_mode:
            self.logger.info(f"[PAPER] Would cancel order {order_id} for {symbol}")
            return {"id": order_id, "status": "canceled"}
        
        try:
            ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol

            time.sleep(self.exchange.rateLimit / 1000)
            result = self._call_with_retries(
                self.exchange.cancel_order,
                order_id,
                ccxt_symbol,
            )

            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
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
            ccxt_symbol = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            markets = self.fetch_markets()
            
            if ccxt_symbol not in markets:
                raise ValueError(f"Symbol {symbol} not found in markets")
            
            market = markets[ccxt_symbol]
            return {
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
                'contractSize': market.get('contractSize', 1.0),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market info for {symbol}: {e}")
            # Return safe defaults
            return {
                'precision': {'price': 2, 'amount': 8},
                'limits': {'amount': {'min': 0.001, 'max': None}, 'cost': {'min': 5.0}},
                'contractSize': 1.0,
            }
    
    def round_amount(self, symbol: str, amount: float) -> float:
        """Round amount to exchange precision."""
        market_info = self.get_market_info(symbol)
        precision = market_info['precision']['amount']
        return round(amount, precision)
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        market_info = self.get_market_info(symbol)
        precision = market_info['precision']['price']
        return round(price, precision)
    
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
        if amount < min_amount:
            return False, f"Amount {amount} below minimum {min_amount}"
        
        # Check minimum cost (notional)
        min_cost = limits['cost']['min']
        cost = amount * price * contract_size
        if cost < min_cost:
            return False, f"Order cost {cost} below minimum {min_cost}"
        
        return True, None

