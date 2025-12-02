"""Test funding strategy with softened trend alignment for shorts."""

import pytest
import pandas as pd
import numpy as np
from src.signals.funding_opportunity import FundingOpportunityGenerator
from src.config import (
    FundingOpportunityConfig,
    FundingOpportunitySizingConfig,
    FundingOpportunityEntryConfig,
    FundingOpportunityExitConfig,
    FundingOpportunityRiskConfig
)
from unittest.mock import Mock, MagicMock


def test_funding_short_with_mildly_positive_momentum():
    """Test that funding strategy allows shorts with mildly positive momentum (softened penalty)."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=True),  # Trend alignment ON
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': 0.0005})  # Positive funding
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create price data with mildly positive momentum (0-10%)
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    base_price = 50000
    # Mildly positive momentum: +5% over 20 bars
    prices = base_price * (1 + np.linspace(0, 0.05, 30))
    
    price_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal('BTCUSDT', funding_rate=0.0005, price_data=price_data)
    
    # Should still generate short signal (positive funding)
    assert signal['signal'] == 'short', f"Expected short signal, got {signal['signal']}"
    
    # Confidence should be reduced but not zero (softened penalty)
    assert signal['confidence'] > 0, "Confidence should be positive even with mildly positive momentum"
    assert signal['confidence'] < 1.0, "Confidence should be reduced due to positive momentum"
    assert signal['confidence'] >= 0.5, (
        f"Confidence should be at least 0.5 with mildly positive momentum (softened), "
        f"got {signal['confidence']:.3f}"
    )


def test_funding_short_with_strongly_positive_momentum():
    """Test that funding strategy heavily penalizes shorts with strongly positive momentum (>10%)."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=True),
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': 0.0005})
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create price data with strongly positive momentum (>10%)
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    base_price = 50000
    # Strongly positive momentum: +15% over 20 bars
    prices = base_price * (1 + np.linspace(0, 0.15, 30))
    
    price_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal('BTCUSDT', funding_rate=0.0005, price_data=price_data)
    
    # Should still generate short signal (positive funding)
    assert signal['signal'] == 'short', f"Expected short signal, got {signal['signal']}"
    
    # Confidence should be heavily reduced (but not zero)
    assert signal['confidence'] > 0, "Confidence should still be positive"
    assert signal['confidence'] < 0.7, (
        f"Confidence should be heavily reduced with strongly positive momentum, "
        f"got {signal['confidence']:.3f}"
    )


def test_funding_short_with_negative_momentum_full_confidence():
    """Test that funding strategy gives full confidence for shorts with negative momentum."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=True),
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': 0.0005})
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create price data with negative momentum (perfect for shorts)
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    base_price = 50000
    # Negative momentum: -10% over 20 bars
    prices = base_price * (1 - np.linspace(0, 0.10, 30))
    
    price_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    signal = generator.generate_signal('BTCUSDT', funding_rate=0.0005, price_data=price_data)
    
    assert signal['signal'] == 'short', f"Expected short signal, got {signal['signal']}"
    
    # Confidence should be high (trend alignment perfect)
    assert signal['confidence'] >= 0.8, (
        f"Confidence should be high with negative momentum (perfect alignment), "
        f"got {signal['confidence']:.3f}"
    )


def test_funding_short_without_trend_alignment():
    """Test that funding strategy works without trend alignment requirement."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=False),  # No trend alignment
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': 0.0005})
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create price data (momentum doesn't matter if require_trend_alignment=False)
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    price_data = pd.DataFrame({
        'open': [50000] * 30,
        'high': [51000] * 30,
        'low': [49000] * 30,
        'close': [50000] * 30,
        'volume': [1000] * 30
    }, index=dates)
    
    signal = generator.generate_signal('BTCUSDT', funding_rate=0.0005, price_data=price_data)
    
    assert signal['signal'] == 'short', f"Expected short signal, got {signal['signal']}"
    assert signal['confidence'] > 0, "Confidence should be positive"
    # Without trend alignment, confidence should be based mainly on funding rate
    assert signal['confidence'] >= 0.5, "Confidence should be reasonable without trend alignment"

