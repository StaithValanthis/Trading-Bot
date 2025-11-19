"""Position sizing calculations."""

from typing import Dict, Optional, Tuple
import numpy as np

from ..config import RiskConfig
from ..exchange.bybit_client import BybitClient
from ..logging_utils import get_logger

logger = get_logger(__name__)


class PositionSizer:
    """Calculate position sizes based on risk parameters."""
    
    def __init__(self, config: RiskConfig, exchange_client: BybitClient):
        """
        Initialize position sizer.
        
        Args:
            config: Risk configuration
            exchange_client: Exchange client for market info
        """
        self.config = config
        self.exchange = exchange_client
        self.logger = get_logger(__name__)
    
    def calculate_position_size(
        self,
        symbol: str,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        signal: str  # 'long' or 'short'
    ) -> Tuple[float, Optional[str]]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            equity: Current account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            signal: Signal direction ('long' or 'short')
        
        Returns:
            Tuple of (contract_size, error_message)
            contract_size is 0 if error, otherwise positive number
        """
        # Calculate risk per trade
        risk_amount = equity * self.config.per_trade_risk_fraction
        
        # Calculate stop distance
        if signal == 'long':
            stop_distance = entry_price - stop_loss_price
        elif signal == 'short':
            stop_distance = stop_loss_price - entry_price
        else:
            return 0.0, "Invalid signal direction"
        
        if stop_distance <= 0:
            return 0.0, "Stop loss must be beyond entry price"
        
        # Get market info
        market_info = self.exchange.get_market_info(symbol)
        contract_size = market_info.get('contractSize', 1.0)
        
        # Calculate position size in contracts
        # Risk = (entry - stop) * contracts * contract_size
        # contracts = Risk / ((entry - stop) * contract_size)
        position_size = risk_amount / (stop_distance * contract_size)
        
        # Apply fractional Kelly if configured
        if self.config.kelly_fraction < 1.0:
            position_size = position_size * self.config.kelly_fraction
        
        # Round to exchange precision
        position_size = self.exchange.round_amount(symbol, position_size)
        
        # Validate order size
        is_valid, error_msg = self.exchange.validate_order_size(
            symbol,
            position_size,
            entry_price
        )
        
        if not is_valid:
            return 0.0, error_msg
        
        return position_size, None
    
    def adjust_for_volatility(
        self,
        symbol: str,
        base_size: float,
        atr: float,
        price: float
    ) -> float:
        """
        Adjust position size based on volatility.
        
        Higher volatility = smaller position size.
        
        Args:
            symbol: Trading pair symbol
            base_size: Base position size
            atr: Average True Range
            price: Current price
        
        Returns:
            Adjusted position size
        """
        # Normalize ATR as percentage of price
        atr_pct = atr / price if price > 0 else 0
        
        # If ATR is very high (>5% of price), reduce size
        # If ATR is very low (<1% of price), can slightly increase size (capped)
        if atr_pct > 0.05:
            # Reduce by up to 50% for very high volatility
            reduction = min(0.5, (atr_pct - 0.05) / 0.1)
            adjusted = base_size * (1 - reduction)
        elif atr_pct < 0.01:
            # Slight increase for low volatility (capped at 20% increase)
            increase = min(0.2, (0.01 - atr_pct) / 0.01 * 0.1)
            adjusted = base_size * (1 + increase)
        else:
            adjusted = base_size
        
        # Round to exchange precision
        adjusted = self.exchange.round_amount(symbol, adjusted)
        
        return max(0.0, adjusted)

