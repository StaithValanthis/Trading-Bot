"""Test that funding strategy generates short signals on positive funding."""

import pytest
import pandas as pd
from src.signals.funding_opportunity import FundingOpportunityGenerator
from src.config import (
    FundingOpportunityConfig,
    FundingOpportunitySizingConfig,
    FundingOpportunityEntryConfig,
    FundingOpportunityExitConfig,
    FundingOpportunityRiskConfig
)
from unittest.mock import Mock, MagicMock


def test_funding_strategy_opens_short_on_positive_funding():
    """Test that funding strategy generates short signals when funding is positive."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=False),
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': 0.0005})  # Positive funding
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create minimal price data
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': [50000] * 30,
            'high': [51000] * 30,
            'low': [49000] * 30,
            'close': [50000] * 30,
            'volume': [1000] * 30
        }, index=dates)
    }
    
    opportunities = generator.scan_opportunities(['BTCUSDT'], symbol_data=symbol_data)
    
    assert len(opportunities) > 0, "Should find at least one opportunity"
    assert opportunities[0].signal == 'short', (
        f"Expected short signal for positive funding, got {opportunities[0].signal}. "
        f"Funding rate: {opportunities[0].funding_rate}"
    )
    assert opportunities[0].funding_rate > 0, "Funding rate should be positive for short"


def test_funding_strategy_opens_long_on_negative_funding():
    """Test that funding strategy generates long signals when funding is negative."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=False),
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    exchange.fetch_funding_rate = MagicMock(return_value={'fundingRate': -0.0005})  # Negative funding
    
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create minimal price data
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': [50000] * 30,
            'high': [51000] * 30,
            'low': [49000] * 30,
            'close': [50000] * 30,
            'volume': [1000] * 30
        }, index=dates)
    }
    
    opportunities = generator.scan_opportunities(['BTCUSDT'], symbol_data=symbol_data)
    
    assert len(opportunities) > 0, "Should find at least one opportunity"
    assert opportunities[0].signal == 'long', (
        f"Expected long signal for negative funding, got {opportunities[0].signal}. "
        f"Funding rate: {opportunities[0].funding_rate}"
    )
    assert opportunities[0].funding_rate < 0, "Funding rate should be negative for long"


def test_funding_strategy_generate_signal_short():
    """Test generate_signal method for short signals."""
    config = FundingOpportunityConfig(
        min_funding_rate=0.0003,
        sizing=FundingOpportunitySizingConfig(),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=False),
        exit=FundingOpportunityExitConfig(),
        risk=FundingOpportunityRiskConfig(max_funding_rate=0.005)
    )
    
    exchange = Mock()
    generator = FundingOpportunityGenerator(config, exchange)
    
    # Create price data
    dates = pd.date_range('2024-01-01', periods=30, freq='4h')
    price_data = pd.DataFrame({
        'open': [50000] * 30,
        'high': [51000] * 30,
        'low': [49000] * 30,
        'close': [50000] * 30,
        'volume': [1000] * 30
    }, index=dates)
    
    # Positive funding should generate short signal
    signal = generator.generate_signal('BTCUSDT', funding_rate=0.0005, price_data=price_data)
    
    assert signal['signal'] == 'short', f"Expected short signal, got {signal['signal']}"
    assert signal['entry_price'] is not None, "Entry price should be set"
    assert signal['stop_loss'] is not None, "Stop loss should be set"
    assert signal['stop_loss'] > signal['entry_price'], "Stop loss should be above entry for short"

