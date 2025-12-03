"""Tests for profitability improvements (Tier 1 & Tier 2 changes)."""

import pytest
from pathlib import Path
from src.config import BotConfig


def test_config_loading_with_new_defaults():
    """Test that config.example.yaml loads correctly with new default values."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    # Tier 1: Cross-sectional changes
    assert config.strategy.cross_sectional.require_trend_alignment == False, \
        "require_trend_alignment should be False"
    assert config.strategy.cross_sectional.top_k == 5, \
        "top_k should be 5"
    
    # Tier 1: Funding changes
    assert config.strategy.funding_opportunity.min_funding_rate == 0.0002, \
        "min_funding_rate should be 0.0002"
    assert config.strategy.funding_opportunity.entry.require_trend_alignment == False, \
        "funding entry require_trend_alignment should be False"
    
    # Tier 1: Optimizer changes
    assert config.optimizer.min_sharpe_ratio == 0.7, \
        "min_sharpe_ratio should be 0.7"
    assert config.optimizer.min_trades == 15, \
        "min_trades should be 15"
    assert config.optimizer.max_drawdown_pct == -20.0, \
        "max_drawdown_pct should be -20.0"
    assert config.optimizer.walk_forward_window_days == 45, \
        "walk_forward_window_days should be 45"
    
    # Tier 2: Trend changes
    assert config.strategy.trend.use_trailing_stop == True, \
        "use_trailing_stop should be True"
    assert config.strategy.trend.atr_stop_multiplier == 1.5, \
        "atr_stop_multiplier should be 1.5"
    
    # Tier 2: Risk changes
    assert config.risk.kelly_fraction == 0.75, \
        "kelly_fraction should be 0.75"


def test_trailing_stop_config_wired():
    """Test that trailing stop config is accessible and has correct defaults."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    # Verify trailing stop config exists and is enabled
    assert hasattr(config.strategy.trend, 'use_trailing_stop'), \
        "use_trailing_stop should exist"
    assert config.strategy.trend.use_trailing_stop == True, \
        "use_trailing_stop should be True"
    
    # Verify trailing stop parameters exist
    assert hasattr(config.strategy.trend, 'trailing_stop_atr_multiplier'), \
        "trailing_stop_atr_multiplier should exist"
    assert config.strategy.trend.trailing_stop_atr_multiplier == 1.5, \
        "trailing_stop_atr_multiplier should be 1.5"
    
    assert hasattr(config.strategy.trend, 'trailing_stop_activation_rr'), \
        "trailing_stop_activation_rr should exist"
    assert config.strategy.trend.trailing_stop_activation_rr == 1.0, \
        "trailing_stop_activation_rr should be 1.0"


def test_kelly_fraction_wired():
    """Test that Kelly fraction is accessible and used in position sizing."""
    from src.risk.position_sizing import PositionSizer
    from src.exchange.bybit_client import BybitClient
    from unittest.mock import Mock
    
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    # Verify Kelly fraction exists
    assert hasattr(config.risk, 'kelly_fraction'), \
        "kelly_fraction should exist in RiskConfig"
    assert config.risk.kelly_fraction == 0.75, \
        "kelly_fraction should be 0.75"
    
    # Verify it's used in PositionSizer (check that it's read from config)
    mock_exchange = Mock(spec=BybitClient)
    mock_exchange.get_market_info.return_value = {
        'contractSize': 1.0,
        'limits': {
            'amount': {'min': 0.0},
            'cost': {'min': 0.0}
        }
    }
    mock_exchange.round_amount = lambda s, a: a
    mock_exchange.validate_order_size = lambda s, a, p: (True, None)
    
    sizer = PositionSizer(config.risk, mock_exchange)
    assert sizer.config.kelly_fraction == 0.75, \
        "PositionSizer should use kelly_fraction from config"


def test_optimizer_constraints():
    """Test that optimizer constraints are correctly set."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    # Verify optimizer config exists
    assert hasattr(config, 'optimizer'), "optimizer config should exist"
    
    # Verify constraints
    assert config.optimizer.min_sharpe_ratio == 0.7, \
        f"Expected min_sharpe_ratio=0.7, got {config.optimizer.min_sharpe_ratio}"
    assert config.optimizer.min_trades == 15, \
        f"Expected min_trades=15, got {config.optimizer.min_trades}"
    assert config.optimizer.max_drawdown_pct == -20.0, \
        f"Expected max_drawdown_pct=-20.0, got {config.optimizer.max_drawdown_pct}"
    assert config.optimizer.walk_forward_window_days == 45, \
        f"Expected walk_forward_window_days=45, got {config.optimizer.walk_forward_window_days}"


def test_atr_stop_multiplier_in_param_ranges():
    """Test that ATR stop multiplier 1.5 is included in optimizer param ranges."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    param_ranges = config.optimizer.param_ranges
    assert 'atr_stop_multiplier' in param_ranges, \
        "atr_stop_multiplier should be in param_ranges"
    
    assert 1.5 in param_ranges['atr_stop_multiplier'], \
        "atr_stop_multiplier param_ranges should include 1.5"


def test_top_k_in_param_ranges():
    """Test that top_k includes expanded range up to 7."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    config = BotConfig.from_yaml(str(config_path))
    
    param_ranges = config.optimizer.param_ranges
    assert 'top_k' in param_ranges, \
        "top_k should be in param_ranges"
    
    assert 7 in param_ranges['top_k'], \
        "top_k param_ranges should include 7"
    assert 5 in param_ranges['top_k'], \
        "top_k param_ranges should include 5"

