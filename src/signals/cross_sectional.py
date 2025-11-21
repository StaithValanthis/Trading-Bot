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
                # Use previous completed bar for ranking (avoid look-ahead bias)
                # Entry execution will still use current price (correct)
                end_price = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
                
                # Validate prices (handle NaN, infinity, zero/negative prices)
                if pd.isna(start_price) or pd.isna(end_price):
                    continue  # Skip symbols with NaN data
                
                if start_price <= 0 or end_price <= 0:
                    continue  # Skip delisted symbols (zero/negative prices)
                
                return_pct = (end_price - start_price) / start_price
                
                # Check for infinity or NaN result
                if not np.isfinite(return_pct):
                    continue  # Skip invalid returns
                
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
        
        Supports both long and short signals:
        - For positive momentum (best performers): selects symbols with 'long' trend signals
        - For negative momentum (worst performers): selects symbols with 'short' trend signals
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            trend_signals: Optional dictionary mapping symbol to trend signal
            require_trend_alignment: If True, only select symbols where trend signal matches momentum
                - For positive momentum (best performers): require 'long' trend signal
                - For negative momentum (worst performers): require 'short' trend signal
                - If False: select based purely on momentum regardless of trend signal
        
        Returns:
            List of top K symbol names (mix of longs and shorts based on momentum and trend alignment)
        """
        rankings = self.rank_symbols(symbol_data)
        
        if not rankings:
            return []
        
        selected = []
        
        # Collect candidates for longs (positive momentum) and shorts (negative momentum)
        long_candidates = []  # (symbol, return_pct) for symbols with positive momentum
        short_candidates = []  # (symbol, return_pct) for symbols with negative momentum
        
        for symbol, return_pct in rankings:
            # Check trend alignment if required
            if require_trend_alignment and trend_signals:
                trend_signal = trend_signals.get(symbol, {})
                signal_direction = trend_signal.get('signal', 'flat')
                
                # Skip symbols without valid trend signal
                if signal_direction == 'flat':
                    continue
                
                # When require_trend_alignment=True:
                # - For positive momentum (best performers): require 'long' trend signal
                # - For negative momentum (worst performers): require 'short' trend signal
                if return_pct > 0:
                    # Positive momentum: require long signal
                    if signal_direction == 'long':
                        long_candidates.append((symbol, return_pct))
                else:
                    # Negative momentum: require short signal
                    if signal_direction == 'short':
                        short_candidates.append((symbol, return_pct))
            elif not require_trend_alignment:
                # No trend alignment required: accept all signals
                trend_signal = trend_signals.get(symbol, {}) if trend_signals else {}
                signal_direction = trend_signal.get('signal', 'flat')
                if signal_direction != 'flat':
                    if return_pct > 0:
                        long_candidates.append((symbol, return_pct))
                    else:
                        short_candidates.append((symbol, return_pct))
            else:
                # require_trend_alignment=True but no trend_signals provided: skip
                continue
        
        # Select top candidates from both groups
        # Prioritize best performers (longs) and worst performers (shorts)
        # Distribute top_k between longs and shorts based on availability
        
        # Start with best longs (positive momentum, descending)
        for symbol, _ in long_candidates:
            if len(selected) >= self.config.top_k:
                break
            selected.append(symbol)
        
        # Then add worst performers with short signals (negative momentum, ascending by absolute return)
        short_candidates_sorted = sorted(short_candidates, key=lambda x: x[1])  # Most negative first
        for symbol, _ in short_candidates_sorted:
            if len(selected) >= self.config.top_k:
                break
            selected.append(symbol)
        
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

