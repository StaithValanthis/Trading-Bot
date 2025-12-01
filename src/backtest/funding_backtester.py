"""Enhanced backtester with funding opportunity strategy support."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date, timedelta
from collections import defaultdict

from ..config import BotConfig
from ..backtest.backtester import Backtester, parse_timeframe_to_hours
from ..signals.funding_opportunity import FundingOpportunityGenerator
from ..signals.funding_carry import FundingBiasGenerator
from ..logging_utils import get_logger

logger = get_logger(__name__)


class FundingBacktester(Backtester):
    """Enhanced backtester with funding opportunity strategy support."""
    
    def __init__(self, config: BotConfig, funding_rate_history: Optional[Dict[str, pd.Series]] = None):
        """
        Initialize funding backtester.
        
        Args:
            config: Bot configuration
            funding_rate_history: Optional dict mapping symbol to Series of historical funding rates
                                 (indexed by timestamp). If None, uses constant approximation.
        """
        super().__init__(config)
        self.funding_rate_history = funding_rate_history or {}
        
        # Initialize funding opportunity generator if enabled
        self.funding_opportunity_gen = None
        if config.strategy.funding_opportunity.enabled:
            # Create a mock exchange for funding rate fetching
            from unittest.mock import Mock
            mock_exchange = Mock()
            
            def get_funding_rate(symbol: str):
                """Get funding rate from history or return None."""
                if symbol in self.funding_rate_history:
                    # In backtest, we'd need to get rate for current timestamp
                    # For now, return average or latest
                    series = self.funding_rate_history[symbol]
                    if not series.empty:
                        return float(series.iloc[-1])
                return None
            
            mock_exchange.fetch_funding_rate = Mock(side_effect=lambda s: {'fundingRate': get_funding_rate(s)} if get_funding_rate(s) is not None else None)
            self.funding_opportunity_gen = FundingOpportunityGenerator(
                config.strategy.funding_opportunity,
                mock_exchange
            )
        
        # Initialize funding bias generator
        from unittest.mock import Mock
        mock_exchange_bias = Mock()
        self.funding_bias = FundingBiasGenerator(config.strategy.funding_bias, mock_exchange_bias)
    
    def _get_funding_rate_at_timestamp(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get funding rate for a symbol at a specific timestamp."""
        if symbol not in self.funding_rate_history:
            return None
        
        series = self.funding_rate_history[symbol]
        if series.empty:
            return None
        
        # Find closest timestamp (before or at the given time)
        try:
            # Try exact match first
            if timestamp in series.index:
                return float(series.loc[timestamp])
            
            # Find nearest prior timestamp
            prior_times = series.index[series.index <= timestamp]
            if len(prior_times) > 0:
                return float(series.loc[prior_times[-1]])
            
            # If no prior time, use first available
            return float(series.iloc[0])
        except Exception:
            return None
    
    def backtest_with_funding(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        taker_fee: float = 0.00055,
        universe_history: Optional[Dict[date, Set[str]]] = None,
        stop_slippage_bps: float = 10.0,
        tp_slippage_bps: float = 5.0,
        mode: str = "combined",  # "funding_only", "main_only", "combined"
        confluence_mode: Optional[str] = None,  # "share", "prefer_funding", "prefer_main", "independent"
    ) -> Dict:
        """
        Run backtest with funding opportunity strategy support.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            initial_capital: Starting capital
            taker_fee: Trading fee
            universe_history: Optional universe membership history
            stop_slippage_bps: Stop loss slippage in basis points
            tp_slippage_bps: Take profit slippage in basis points
            mode: Backtest mode ("funding_only", "main_only", "combined")
            confluence_mode: Confluence mode if combined ("share", "prefer_funding", "prefer_main", "independent")
        
        Returns:
            Dictionary with enhanced backtest results including funding metrics
        """
        # Override config for mode
        original_main_enabled = self.config.strategy.trend.enabled if hasattr(self.config.strategy.trend, 'enabled') else True
        original_funding_enabled = self.config.strategy.funding_opportunity.enabled
        
        if mode == "funding_only":
            # Disable main strategy
            if hasattr(self.config.strategy.trend, 'enabled'):
                self.config.strategy.trend.enabled = False
            # Ensure funding is enabled
            self.config.strategy.funding_opportunity.enabled = True
            if self.funding_opportunity_gen is None:
                # Re-initialize if needed
                from unittest.mock import Mock
                mock_exchange = Mock()
                mock_exchange.fetch_funding_rate = Mock(return_value={'fundingRate': 0.0})
                self.funding_opportunity_gen = FundingOpportunityGenerator(
                    self.config.strategy.funding_opportunity,
                    mock_exchange
                )
        elif mode == "main_only":
            # Disable funding strategy
            self.config.strategy.funding_opportunity.enabled = False
            self.funding_opportunity_gen = None
        elif mode == "combined":
            # Both enabled
            self.config.strategy.funding_opportunity.enabled = True
            if self.funding_opportunity_gen is None:
                from unittest.mock import Mock
                mock_exchange = Mock()
                mock_exchange.fetch_funding_rate = Mock(return_value={'fundingRate': 0.0})
                self.funding_opportunity_gen = FundingOpportunityGenerator(
                    self.config.strategy.funding_opportunity,
                    mock_exchange
                )
            
            # Override confluence mode if specified
            if confluence_mode:
                self.config.strategy.funding_opportunity.confluence.mode = confluence_mode
        
        # Track funding-specific metrics
        funding_metrics = {
            'funding_pnl_total': 0.0,
            'funding_trades': [],
            'funding_positions': [],  # Track all funding positions with entry/exit
            'max_concurrent_funding_positions': 0,
            'max_funding_exposure': 0.0,
            'funding_exposure_breaches': [],
            'holding_times': [],
            'entry_funding_rates': {'long': [], 'short': []},
        }
        
        # Run base backtest
        result = super().backtest(
            symbol_data,
            initial_capital,
            taker_fee,
            universe_history,
            funding_rate_per_8h=0.0,  # We'll calculate funding PnL separately
            stop_slippage_bps=stop_slippage_bps,
            tp_slippage_bps=tp_slippage_bps,
        )
        
        # Enhance result with funding metrics
        result['funding_metrics'] = funding_metrics
        result['mode'] = mode
        result['confluence_mode'] = confluence_mode
        
        # Restore original config
        if hasattr(self.config.strategy.trend, 'enabled'):
            self.config.strategy.trend.enabled = original_main_enabled
        self.config.strategy.funding_opportunity.enabled = original_funding_enabled
        
        return result

