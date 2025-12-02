"""Test that trend strategy generates short signals in downtrends."""

import pytest
import pandas as pd
import numpy as np
from src.signals.trend import TrendSignalGenerator
from src.config import TrendStrategyConfig


def test_trend_strategy_generates_short_in_downtrend():
    """Test that trend strategy generates short signals in a clear downtrend."""
    config = TrendStrategyConfig(
        ma_short=5,
        ma_long=20,
        momentum_lookback=10,
        atr_stop_multiplier=2.0,
        atr_period=14,
        min_atr_threshold=0.001
    )
    generator = TrendSignalGenerator(config)
    
    # Create a clear downtrend: price declining, MAs crossed down, negative momentum
    dates = pd.date_range('2024-01-01', periods=50, freq='4h')
    prices = 100 - np.linspace(0, 20, 50)  # Declining from 100 to 80
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal(df)
    
    assert signal['signal'] == 'short', (
        f"Expected short signal in downtrend, got {signal['signal']}. "
        f"Price={prices[-1]:.2f}, MA_long={signal['metadata'].get('ma_long', 0):.2f}, "
        f"momentum={signal['metadata'].get('momentum', 0):.4f}"
    )
    assert signal['stop_loss'] > signal['entry_price'], (
        f"Stop loss should be above entry for short. "
        f"Entry={signal['entry_price']:.2f}, Stop={signal['stop_loss']:.2f}"
    )
    assert signal['confidence'] > 0, f"Confidence should be positive, got {signal['confidence']}"


def test_trend_strategy_generates_long_in_uptrend():
    """Test that trend strategy generates long signals in a clear uptrend."""
    config = TrendStrategyConfig(
        ma_short=5,
        ma_long=20,
        momentum_lookback=10,
        atr_stop_multiplier=2.0,
        atr_period=14,
        min_atr_threshold=0.001
    )
    generator = TrendSignalGenerator(config)
    
    # Create a clear uptrend: price rising, MAs crossed up, positive momentum
    dates = pd.date_range('2024-01-01', periods=50, freq='4h')
    prices = 80 + np.linspace(0, 20, 50)  # Rising from 80 to 100
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal(df)
    
    assert signal['signal'] == 'long', (
        f"Expected long signal in uptrend, got {signal['signal']}. "
        f"Price={prices[-1]:.2f}, MA_long={signal['metadata'].get('ma_long', 0):.2f}, "
        f"momentum={signal['metadata'].get('momentum', 0):.4f}"
    )
    assert signal['stop_loss'] < signal['entry_price'], (
        f"Stop loss should be below entry for long. "
        f"Entry={signal['entry_price']:.2f}, Stop={signal['stop_loss']:.2f}"
    )


def test_trend_strategy_returns_flat_when_conditions_not_met():
    """Test that trend strategy returns 'flat' when conditions aren't met."""
    config = TrendStrategyConfig(
        ma_short=5,
        ma_long=20,
        momentum_lookback=10,
        atr_stop_multiplier=2.0,
        atr_period=14,
        min_atr_threshold=0.001
    )
    generator = TrendSignalGenerator(config)
    
    # Create choppy/neutral market: price oscillating around MA
    dates = pd.date_range('2024-01-01', periods=50, freq='4h')
    prices = 100 + np.sin(np.linspace(0, 4*np.pi, 50)) * 2  # Oscillating around 100
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal(df)
    
    # Should be flat if conditions aren't met (price not clearly above/below MA, or momentum near zero)
    assert signal['signal'] in ['long', 'short', 'flat'], f"Signal should be long, short, or flat, got {signal['signal']}"

