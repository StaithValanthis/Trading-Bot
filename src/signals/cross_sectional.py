"""Cross-sectional momentum signal generation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from ..config import CrossSectionalStrategyConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class CrossSectionalSignalGenerator:
    """Generate cross-sectional momentum signals."""
    
    def __init__(self, config: CrossSectionalStrategyConfig):
        """
        Initialize cross-sectional signal generator.
        
        Args:
            config: Cross-sectional strategy configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def rank_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, float]]:
        """
        Rank symbols by recent performance.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
        
        Returns:
            List of (symbol, return) tuples, sorted by return descending
        """
        rankings = []
        
        for symbol, df in symbol_data.items():
            if len(df) < self.config.ranking_window:
                continue
            
            close = df['close']
            # Calculate return over ranking window
            if len(close) >= self.config.ranking_window:
                start_price = close.iloc[-self.config.ranking_window]
                end_price = close.iloc[-1]
                return_pct = (end_price - start_price) / start_price
                rankings.append((symbol, return_pct))
        
        # Sort by return descending (best performers first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def select_top_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        trend_signals: Dict[str, Dict] = None,
        require_trend_alignment: bool = True
    ) -> List[str]:
        """
        Select top K symbols based on cross-sectional ranking.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            trend_signals: Optional dictionary mapping symbol to trend signal
            require_trend_alignment: If True, only select symbols with positive trend signals
        
        Returns:
            List of top K symbol names
        """
        rankings = self.rank_symbols(symbol_data)
        
        if not rankings:
            return []
        
        selected = []
        
        for symbol, return_pct in rankings:
            # Check trend alignment if required
            if require_trend_alignment and trend_signals:
                trend_signal = trend_signals.get(symbol, {})
                if trend_signal.get('signal') != 'long':
                    # Skip symbols without long trend signal
                    continue
            
            selected.append(symbol)
            
            if len(selected) >= self.config.top_k:
                break
        
        return selected
    
    def generate_portfolio_weights(
        self,
        selected_symbols: List[str],
        rankings: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Generate portfolio weights for selected symbols.
        
        Args:
            selected_symbols: List of selected symbol names
            rankings: List of (symbol, return) tuples
        
        Returns:
            Dictionary mapping symbol to weight (0-1, sum to 1.0)
        """
        if not selected_symbols:
            return {}
        
        # Create ranking map
        ranking_map = {symbol: return_pct for symbol, return_pct in rankings}
        
        # Weight by normalized returns (higher return = higher weight)
        weights = {}
        total_weight = 0.0
        
        for symbol in selected_symbols:
            return_pct = ranking_map.get(symbol, 0.0)
            # Normalize to positive weights (shift by min return)
            weights[symbol] = max(0.0, return_pct)
            total_weight += weights[symbol]
        
        # Normalize to sum to 1.0
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if all returns are negative/zero
            equal_weight = 1.0 / len(selected_symbols)
            weights = {symbol: equal_weight for symbol in selected_symbols}
        
        return weights

