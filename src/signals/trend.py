"""Time-series trend-following signal generation."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from ..config import TrendStrategyConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
    
    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


class TrendSignalGenerator:
    """Generate trend-following signals."""
    
    def __init__(self, config: TrendStrategyConfig):
        """
        Initialize trend signal generator.
        
        Args:
            config: Trend strategy configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def generate_signal(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Generate trend-following signal.
        
        Args:
            df: DataFrame with OHLCV data, indexed by datetime
        
        Returns:
            Dictionary with:
                - signal: 'long', 'short', or 'flat'
                - entry_price: Suggested entry price (current close)
                - stop_loss: Stop loss price
                - confidence: Signal strength (0-1)
                - metadata: Additional info (ATR, MAs, etc.)
        """
        if len(df) < self.config.ma_long:
            return {
                'signal': 'flat',
                'entry_price': df['close'].iloc[-1] if not df.empty else None,
                'stop_loss': None,
                'confidence': 0.0,
                'metadata': {'reason': 'insufficient_data'}
            }
        
        # Calculate indicators
        close = df['close']
        ma_short = close.rolling(window=self.config.ma_short).mean()
        ma_long = close.rolling(window=self.config.ma_long).mean()
        atr = calculate_atr(df, self.config.atr_period)
        
        # Momentum (recent return)
        momentum = close.pct_change(periods=self.config.momentum_lookback)
        
        # Current values
        current_price = close.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        current_momentum = momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0
        
        # ATR filter: skip if volatility too low
        atr_threshold = current_price * self.config.min_atr_threshold
        if current_atr < atr_threshold:
            return {
                'signal': 'flat',
                'entry_price': current_price,
                'stop_loss': None,
                'confidence': 0.0,
                'metadata': {
                    'reason': 'low_volatility',
                    'atr': current_atr,
                    'atr_threshold': atr_threshold
                }
            }
        
        # Trend signals
        # Long: price above long MA, short MA above long MA, positive momentum
        long_condition = (
            current_price > current_ma_long and
            current_ma_short > current_ma_long and
            current_momentum > 0
        )
        
        # Short: price below long MA, short MA below long MA, negative momentum
        short_condition = (
            current_price < current_ma_long and
            current_ma_short < current_ma_long and
            current_momentum < 0
        )
        
        # Calculate stop loss
        stop_loss = None
        signal = 'flat'
        confidence = 0.0
        
        if long_condition:
            signal = 'long'
            # Stop loss below entry by ATR multiplier
            stop_loss = current_price - (current_atr * self.config.atr_stop_multiplier)
            # Confidence based on how far price is above MA and momentum strength
            price_above_ma = (current_price - current_ma_long) / current_ma_long
            confidence = min(1.0, 0.5 + abs(current_momentum) * 10 + price_above_ma * 10)
            
        elif short_condition:
            signal = 'short'
            # Stop loss above entry by ATR multiplier
            stop_loss = current_price + (current_atr * self.config.atr_stop_multiplier)
            # Confidence based on how far price is below MA and momentum strength
            price_below_ma = (current_ma_long - current_price) / current_ma_long
            confidence = min(1.0, 0.5 + abs(current_momentum) * 10 + price_below_ma * 10)
        
        return {
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'metadata': {
                'ma_short': current_ma_short,
                'ma_long': current_ma_long,
                'atr': current_atr,
                'momentum': current_momentum,
                'price_above_ma_pct': (current_price - current_ma_long) / current_ma_long * 100
            }
        }

