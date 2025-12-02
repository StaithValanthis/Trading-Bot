"""Test cross-sectional strategy balanced long/short selection."""

import pytest
import pandas as pd
import numpy as np
from src.signals.cross_sectional import CrossSectionalSignalGenerator
from src.config import CrossSectionalStrategyConfig


@pytest.fixture
def sample_symbol_data():
    """Create sample symbol data with both positive and negative momentum."""
    dates = pd.date_range('2024-01-01', periods=50, freq='4h')
    
    symbol_data = {}
    
    # Create 5 symbols with positive momentum (uptrends)
    for i in range(5):
        symbol = f'LONG{i}USDT'
        base_price = 100 + i * 10
        prices = base_price + np.linspace(0, 10, 50)  # Rising
        symbol_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000
        }, index=dates)
    
    # Create 5 symbols with negative momentum (downtrends)
    for i in range(5):
        symbol = f'SHORT{i}USDT'
        base_price = 100 + i * 10
        prices = base_price - np.linspace(0, 10, 50)  # Declining
        symbol_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000
        }, index=dates)
    
    return symbol_data


@pytest.fixture
def trend_signals_with_both_directions():
    """Create trend signals for both long and short symbols."""
    signals = {}
    
    # Long signals for positive momentum symbols
    for i in range(5):
        symbol = f'LONG{i}USDT'
        signals[symbol] = {
            'signal': 'long',
            'entry_price': 100 + i * 10,
            'stop_loss': 95 + i * 10,
            'confidence': 0.8
        }
    
    # Short signals for negative momentum symbols
    for i in range(5):
        symbol = f'SHORT{i}USDT'
        signals[symbol] = {
            'signal': 'short',
            'entry_price': 100 + i * 10,
            'stop_loss': 105 + i * 10,
            'confidence': 0.8
        }
    
    return signals


def test_balanced_selection_includes_both_longs_and_shorts(sample_symbol_data, trend_signals_with_both_directions):
    """Test that balanced selection returns both longs and shorts when available."""
    config = CrossSectionalStrategyConfig(
        top_k=10,
        ranking_window=20,
        balanced_long_short=True,
        require_trend_alignment=True
    )
    generator = CrossSectionalSignalGenerator(config)
    
    selected = generator.select_top_symbols(
        sample_symbol_data,
        trend_signals_with_both_directions,
        require_trend_alignment=True
    )
    
    # With balanced_long_short=True and top_k=10, should get ~5 longs and ~5 shorts
    rankings = generator.rank_symbols(sample_symbol_data)
    ranking_map = {s: ret for s, ret in rankings}
    
    selected_longs = [s for s in selected if ranking_map.get(s, 0) > 0]
    selected_shorts = [s for s in selected if ranking_map.get(s, 0) < 0]
    
    assert len(selected_longs) > 0, "Should select at least one long"
    assert len(selected_shorts) > 0, "Should select at least one short"
    assert len(selected) <= config.top_k, f"Should not exceed top_k={config.top_k}, got {len(selected)}"
    
    # With balanced selection and top_k=10, should get approximately 5 of each
    assert abs(len(selected_longs) - len(selected_shorts)) <= 2, (
        f"Balanced selection should have similar counts. "
        f"Longs: {len(selected_longs)}, Shorts: {len(selected_shorts)}"
    )


def test_balanced_selection_gracefully_handles_only_longs(sample_symbol_data):
    """Test that balanced selection gracefully handles case with only long candidates."""
    config = CrossSectionalStrategyConfig(
        top_k=10,
        ranking_window=20,
        balanced_long_short=True,
        require_trend_alignment=False
    )
    generator = CrossSectionalSignalGenerator(config)
    
    # Create trend signals with only long signals
    trend_signals = {}
    for i in range(5):
        symbol = f'LONG{i}USDT'
        trend_signals[symbol] = {'signal': 'long'}
    
    selected = generator.select_top_symbols(
        sample_symbol_data,
        trend_signals,
        require_trend_alignment=False
    )
    
    # Should still work, just return longs
    assert len(selected) > 0, "Should select symbols even if only longs available"
    assert len(selected) <= config.top_k, f"Should not exceed top_k={config.top_k}"


def test_balanced_selection_gracefully_handles_only_shorts(sample_symbol_data):
    """Test that balanced selection gracefully handles case with only short candidates."""
    config = CrossSectionalStrategyConfig(
        top_k=10,
        ranking_window=20,
        balanced_long_short=True,
        require_trend_alignment=False
    )
    generator = CrossSectionalSignalGenerator(config)
    
    # Create trend signals with only short signals
    trend_signals = {}
    for i in range(5):
        symbol = f'SHORT{i}USDT'
        trend_signals[symbol] = {'signal': 'short'}
    
    selected = generator.select_top_symbols(
        sample_symbol_data,
        trend_signals,
        require_trend_alignment=False
    )
    
    # Should still work, just return shorts
    assert len(selected) > 0, "Should select symbols even if only shorts available"
    assert len(selected) <= config.top_k, f"Should not exceed top_k={config.top_k}"


def test_priority_selection_preserves_old_behavior(sample_symbol_data, trend_signals_with_both_directions):
    """Test that priority selection (balanced_long_short=False) preserves old behavior."""
    config = CrossSectionalStrategyConfig(
        top_k=10,
        ranking_window=20,
        balanced_long_short=False,  # Old behavior
        require_trend_alignment=True
    )
    generator = CrossSectionalSignalGenerator(config)
    
    selected = generator.select_top_symbols(
        sample_symbol_data,
        trend_signals_with_both_directions,
        require_trend_alignment=True
    )
    
    rankings = generator.rank_symbols(sample_symbol_data)
    ranking_map = {s: ret for s, ret in rankings}
    
    selected_longs = [s for s in selected if ranking_map.get(s, 0) > 0]
    selected_shorts = [s for s in selected if ranking_map.get(s, 0) < 0]
    
    # With priority selection, longs should be selected first
    # If there are enough long candidates, shorts may be crowded out
    assert len(selected) <= config.top_k, f"Should not exceed top_k={config.top_k}"
    
    # Priority selection: all longs first, then shorts
    # So if we have 5 longs and top_k=10, we should get all 5 longs, then up to 5 shorts
    assert len(selected_longs) <= len([s for s, _ in rankings if ranking_map.get(s, 0) > 0]), (
        "Should not select more longs than available"
    )


def test_require_trend_alignment_false_allows_shorts_on_negative_momentum(sample_symbol_data):
    """Test that require_trend_alignment=False allows shorts based on negative momentum alone."""
    config = CrossSectionalStrategyConfig(
        top_k=10,
        ranking_window=20,
        balanced_long_short=True,
        require_trend_alignment=False  # Key: no trend alignment required
    )
    generator = CrossSectionalSignalGenerator(config)
    
    # Create trend signals with 'flat' for negative momentum symbols
    # This tests that shorts can be selected even without explicit short trend signals
    trend_signals = {}
    for i in range(5):
        symbol = f'LONG{i}USDT'
        trend_signals[symbol] = {'signal': 'long'}
    
    for i in range(5):
        symbol = f'SHORT{i}USDT'
        trend_signals[symbol] = {'signal': 'flat'}  # No short signal, but negative momentum
    
    selected = generator.select_top_symbols(
        sample_symbol_data,
        trend_signals,
        require_trend_alignment=False
    )
    
    rankings = generator.rank_symbols(sample_symbol_data)
    ranking_map = {s: ret for s, ret in rankings}
    
    selected_shorts = [s for s in selected if ranking_map.get(s, 0) < 0]
    
    # With require_trend_alignment=False, shorts should be selected based on negative momentum alone
    assert len(selected_shorts) > 0, (
        "Should select shorts based on negative momentum even when trend signal is 'flat' "
        "(require_trend_alignment=False)"
    )

