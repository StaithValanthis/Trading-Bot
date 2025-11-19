"""Funding-rate bias overlay for signal adjustment."""

from typing import Dict, Optional

from ..config import FundingBiasConfig
from ..exchange.bybit_client import BybitClient
from ..logging_utils import get_logger

logger = get_logger(__name__)


class FundingBiasGenerator:
    """Generate funding-rate bias adjustments."""
    
    def __init__(self, config: FundingBiasConfig, exchange_client: BybitClient):
        """
        Initialize funding bias generator.
        
        Args:
            config: Funding bias configuration
            exchange_client: Exchange client for fetching funding rates
        """
        self.config = config
        self.exchange = exchange_client
        self.logger = get_logger(__name__)
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Funding rate (e.g., 0.0001 = 0.01% per 8h), or None if error
        """
        try:
            funding_info = self.exchange.fetch_funding_rate(symbol)
            return funding_info.get('fundingRate', 0.0)
        except Exception as e:
            self.logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def calculate_size_adjustment(
        self,
        symbol: str,
        signal: str,  # 'long' or 'short'
        base_size: float
    ) -> float:
        """
        Calculate position size adjustment based on funding rate.
        
        Args:
            symbol: Trading pair symbol
            signal: Signal direction ('long' or 'short')
            base_size: Base position size before adjustment
        
        Returns:
            Adjusted position size
        """
        funding_rate = self.get_funding_rate(symbol)
        
        if funding_rate is None:
            # Return base size if funding rate unavailable
            return base_size
        
        abs_funding = abs(funding_rate)
        
        # Only apply adjustment if funding rate exceeds threshold
        if abs_funding < self.config.min_funding_threshold:
            return base_size
        
        adjusted_size = base_size
        
        if signal == 'long':
            if funding_rate > 0:
                # Positive funding: we pay funding to hold long
                # Reduce position size
                adjusted_size = base_size * self.config.positive_funding_size_reduction
            elif funding_rate < 0:
                # Negative funding: we receive funding to hold long
                # Increase position size (capped)
                boost = self.config.negative_funding_size_boost
                max_boost = self.config.max_boost_factor
                adjusted_size = base_size * min(boost, max_boost)
        
        elif signal == 'short':
            if funding_rate < 0:
                # Negative funding: we pay funding to hold short (we receive negative, so we pay)
                # Reduce position size
                adjusted_size = base_size * self.config.positive_funding_size_reduction
            elif funding_rate > 0:
                # Positive funding: we receive funding to hold short
                # Increase position size (capped)
                boost = self.config.negative_funding_size_boost
                max_boost = self.config.max_boost_factor
                adjusted_size = base_size * min(boost, max_boost)
        
        return adjusted_size
    
    def should_filter_trade(
        self,
        symbol: str,
        signal: str
    ) -> bool:
        """
        Determine if a trade should be filtered out due to funding rate.
        
        Args:
            symbol: Trading pair symbol
            signal: Signal direction ('long' or 'short')
        
        Returns:
            True if trade should be filtered out
        """
        funding_rate = self.get_funding_rate(symbol)
        
        if funding_rate is None:
            return False
        
        abs_funding = abs(funding_rate)
        
        # Only filter if funding rate is extremely high (configurable)
        # For now, we don't filter, just adjust size
        return False

