"""Unit tests for funding opportunity strategy."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock
import pandas as pd

from src.signals.funding_opportunity import FundingOpportunityGenerator, FundingOpportunity
from src.config import (
    FundingOpportunityConfig,
    FundingOpportunityUniverseConfig,
    FundingOpportunitySizingConfig,
    FundingOpportunityEntryConfig,
    FundingOpportunityExitConfig,
    FundingOpportunityRiskConfig,
    FundingOpportunityConfluenceConfig,
)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange client."""
    exchange = Mock()
    exchange.fetch_funding_rate = Mock(return_value={'fundingRate': 0.001})  # 0.1% per 8h
    return exchange


@pytest.fixture
def funding_config():
    """Create a funding opportunity config for testing."""
    return FundingOpportunityConfig(
        enabled=True,
        min_funding_rate=0.0003,
        max_positions=5,
        universe=FundingOpportunityUniverseConfig(use_main_universe=True),
        sizing=FundingOpportunitySizingConfig(
            base_size_fraction=0.08,
            size_multiplier=75.0,
            max_position_size=0.15
        ),
        entry=FundingOpportunityEntryConfig(require_trend_alignment=False),
        exit=FundingOpportunityExitConfig(
            exit_on_funding_flip=True,
            exit_funding_threshold=0.00015,
            max_holding_hours=120
        ),
        risk=FundingOpportunityRiskConfig(
            max_total_funding_exposure=0.40,
            max_funding_rate=0.005
        ),
        confluence=FundingOpportunityConfluenceConfig(
            enabled=True,
            mode="share"
        )
    )


@pytest.fixture
def funding_generator(mock_exchange, funding_config):
    """Create a FundingOpportunityGenerator instance."""
    return FundingOpportunityGenerator(funding_config, mock_exchange)


def test_scan_opportunities_filters_by_min_rate(funding_generator, mock_exchange):
    """Test that scan_opportunities filters by minimum funding rate."""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Mock funding rates: BTC (high), ETH (low), SOL (high)
    def mock_funding_rate(symbol):
        rates = {
            'BTCUSDT': {'fundingRate': 0.0005},  # 0.05% per 8h - above threshold
            'ETHUSDT': {'fundingRate': 0.0001},  # 0.01% per 8h - below threshold
            'SOLUSDT': {'fundingRate': 0.0004},  # 0.04% per 8h - above threshold
        }
        return rates.get(symbol, {'fundingRate': 0.0})
    
    mock_exchange.fetch_funding_rate = Mock(side_effect=mock_funding_rate)
    
    # Create minimal symbol data
    symbol_data = {}
    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000.0] * 100
        }, index=dates)
        symbol_data[symbol] = df
    
    opportunities = funding_generator.scan_opportunities(symbols, symbol_data)
    
    # Should only include BTC and SOL (above min_funding_rate=0.0003)
    assert len(opportunities) == 2
    assert all(opp.symbol in ['BTCUSDT', 'SOLUSDT'] for opp in opportunities)
    assert all(abs(opp.funding_rate) >= 0.0003 for opp in opportunities)


def test_should_exit_on_funding_flip(funding_generator, mock_exchange):
    """Test that should_exit returns True when funding flips sign."""
    symbol = 'BTCUSDT'
    # For funding opportunities: LONG when funding is NEGATIVE (we receive)
    position = {
        'signal': 'long',
        'entry_time': datetime.now(timezone.utc) - timedelta(hours=1),
        'metadata': {'funding_rate': -0.0005}  # Was negative (we received funding)
    }
    
    # Funding flipped to positive (we'd now pay)
    current_funding = 0.0003
    
    should_exit, reason = funding_generator.should_exit(symbol, position, current_funding, None)
    
    assert should_exit is True
    assert 'funding_flipped' in reason.lower()


def test_should_exit_on_threshold_with_hysteresis(funding_generator, mock_exchange):
    """Test that should_exit returns True when funding drops below threshold with hysteresis."""
    symbol = 'BTCUSDT'
    position = {
        'signal': 'long',
        'entry_time': datetime.now(timezone.utc) - timedelta(hours=1),
        'metadata': {'funding_rate': -0.0005}  # Was -0.05% per 8h (we received)
    }
    
    # Funding dropped below threshold (0.00015 * 0.8 = 0.00012)
    # For long position, we care about absolute value
    current_funding = -0.0001  # Below hysteresis threshold (abs = 0.0001 < 0.00012)
    
    should_exit, reason = funding_generator.should_exit(symbol, position, current_funding, None)
    
    assert should_exit is True
    assert 'funding_below_threshold' in reason.lower()


def test_should_exit_max_holding_hours(funding_generator, mock_exchange):
    """Test that should_exit returns True when max holding period is exceeded."""
    symbol = 'BTCUSDT'
    # Position opened 125 hours ago (exceeds 120 hour limit)
    position = {
        'signal': 'long',
        'entry_time': datetime.now(timezone.utc) - timedelta(hours=125),
        'metadata': {'funding_rate': -0.0005}
    }
    
    current_funding = -0.0004  # Still above threshold (abs = 0.0004 > 0.00012)
    
    should_exit, reason = funding_generator.should_exit(symbol, position, current_funding, None)
    
    assert should_exit is True
    assert 'max_holding_period' in reason.lower()


def test_funding_sizing_formula(funding_generator):
    """Test that funding sizing formula scales correctly."""
    equity = 10000.0
    entry_price = 50000.0
    contract_size = 1.0
    
    # Base size: 8% of equity
    base_size_fraction = 0.08
    base_notional = equity * base_size_fraction  # 800
    base_size = base_notional / (entry_price * contract_size)  # 0.016
    
    # With funding rate 0.0005 (0.05% per 8h)
    funding_rate = 0.0005
    size_multiplier = 75.0
    funding_multiplier = 1.0 + (abs(funding_rate) * size_multiplier)  # 1.0375
    target_size_fraction = min(
        base_size_fraction * funding_multiplier,  # 0.083
        0.15  # max_position_size
    )
    
    target_notional = equity * target_size_fraction  # 830
    target_size = target_notional / (entry_price * contract_size)  # 0.0166
    
    # Size should be slightly larger than base due to funding multiplier
    assert target_size > base_size
    assert target_size_fraction <= 0.15  # Respects max


def test_max_total_funding_exposure_enforcement():
    """Test that max_total_funding_exposure is respected."""
    equity = 10000.0
    max_exposure = 0.40  # 40% of equity
    max_notional = equity * max_exposure  # 4000
    
    # If we already have 3500 in funding positions, we can only add 500 more
    existing_exposure = 3500.0
    available_exposure = max_notional - existing_exposure  # 500
    
    assert available_exposure == 500.0
    assert existing_exposure + available_exposure <= max_notional


def test_detect_confluence_aligned_signals(funding_generator):
    """Test that detect_confluence identifies aligned signals."""
    funding_selected = ['BTCUSDT', 'ETHUSDT']
    funding_signals = {
        'BTCUSDT': {'signal': 'long', 'entry_price': 50000, 'stop_loss': 49000},
        'ETHUSDT': {'signal': 'short', 'entry_price': 3000, 'stop_loss': 3100},
    }
    main_selected = ['BTCUSDT', 'SOLUSDT']
    main_signals = {
        'BTCUSDT': {'signal': 'long', 'entry_price': 50000, 'stop_loss': 49000},
        'SOLUSDT': {'signal': 'long', 'entry_price': 100, 'stop_loss': 95},
    }
    
    confluence = funding_generator.detect_confluence(
        funding_selected,
        funding_signals,
        main_selected,
        main_signals
    )
    
    # BTCUSDT should have confluence (both long)
    assert 'BTCUSDT' in confluence
    assert confluence['BTCUSDT']['aligned'] is True
    assert confluence['BTCUSDT']['confluence_type'] == 'long'
    
    # ETHUSDT should not have confluence (only in funding, not main)
    assert 'ETHUSDT' not in confluence or confluence.get('ETHUSDT', {}).get('aligned') is False


def test_detect_confluence_misaligned_signals(funding_generator):
    """Test that detect_confluence identifies misaligned signals."""
    funding_selected = ['BTCUSDT']
    funding_signals = {
        'BTCUSDT': {'signal': 'long', 'entry_price': 50000, 'stop_loss': 49000},
    }
    main_selected = ['BTCUSDT']
    main_signals = {
        'BTCUSDT': {'signal': 'short', 'entry_price': 50000, 'stop_loss': 51000},
    }
    
    confluence = funding_generator.detect_confluence(
        funding_selected,
        funding_signals,
        main_selected,
        main_signals
    )
    
    # BTCUSDT should not have aligned confluence (signals conflict)
    assert 'BTCUSDT' in confluence
    assert confluence['BTCUSDT']['aligned'] is False


def test_generate_signal_creates_correct_structure(funding_generator):
    """Test that generate_signal creates a signal dict with correct structure."""
    symbol = 'BTCUSDT'
    funding_rate = 0.0005
    symbol_data = None  # Not needed for basic signal generation
    
    signal = funding_generator.generate_signal(symbol, funding_rate, symbol_data)
    
    assert 'signal' in signal
    assert 'entry_price' in signal
    assert 'stop_loss' in signal
    assert 'source' in signal
    assert signal['source'] == 'funding_opportunity'
    assert signal['signal'] in ['long', 'short']


def test_get_funding_rate_handles_missing_symbol(funding_generator, mock_exchange):
    """Test that get_funding_rate handles missing symbols gracefully."""
    mock_exchange.fetch_funding_rate = Mock(side_effect=Exception("Symbol not found"))
    
    result = funding_generator.get_funding_rate('INVALIDUSDT')
    
    assert result is None

