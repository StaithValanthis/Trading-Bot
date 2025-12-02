"""Test funding strategy optimizer."""

import pytest
from unittest.mock import Mock, MagicMock
from src.optimizer.optimizer import Optimizer
from src.config import BotConfig, FundingOptimizerConfig
from src.data.ohlcv_store import OHLCVStore
import tempfile
import os


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_config():
    """Create a mock config with funding optimizer enabled."""
    config = Mock(spec=BotConfig)
    config.optimizer.funding = FundingOptimizerConfig(
        enabled=True,
        trials=5,  # Small number for quick test
        lookback_months=1,
        walk_forward_window_days=7,
        min_trades=5,
        min_sharpe=0.3,
        max_dd=-0.50,
        min_funding_pnl_share=0.1,
        disable_main_strategy=False,
        param_ranges={
            "min_funding_rate": [0.0001, 0.0003],
            "exit_funding_threshold": [0.0001, 0.00015],
            "base_size_fraction": [0.05, 0.08],
            "max_total_funding_exposure": [0.30, 0.40],
            "max_holding_hours": [72, 120],
            "stop_loss_atr_multiplier": [2.0, 2.5],
            "take_profit_rr": [None, 2.0],
            "require_trend_alignment": [True, False],
        }
    )
    config.strategy.funding_opportunity.enabled = True
    config.exchange.timeframe = "1h"
    return config


def test_funding_optimizer_initialization(mock_config, temp_db):
    """Test that funding optimizer can be initialized."""
    store = OHLCVStore(temp_db)
    optimizer = Optimizer(mock_config, store)
    
    # Should have funding optimization methods
    assert hasattr(optimizer, 'optimize_funding_strategy')
    assert hasattr(optimizer, '_get_current_funding_params')
    assert hasattr(optimizer, '_generate_funding_param_candidates')
    assert hasattr(optimizer, '_create_funding_test_config')


def test_get_current_funding_params(mock_config, temp_db):
    """Test extraction of current funding parameters."""
    store = OHLCVStore(temp_db)
    optimizer = Optimizer(mock_config, store)
    
    params = optimizer._get_current_funding_params()
    
    # Should extract funding parameters
    assert 'min_funding_rate' in params
    assert 'exit_funding_threshold' in params
    assert 'base_size_fraction' in params
    assert 'max_total_funding_exposure' in params


def test_generate_funding_param_candidates(mock_config, temp_db):
    """Test generation of funding parameter candidates."""
    store = OHLCVStore(temp_db)
    optimizer = Optimizer(mock_config, store)
    
    param_sets = optimizer._generate_funding_param_candidates(
        n_trials=5,
        param_ranges=mock_config.optimizer.funding.param_ranges,
        method="uniform"
    )
    
    assert len(param_sets) == 5
    assert all(isinstance(p, dict) for p in param_sets)
    assert all('min_funding_rate' in p for p in param_sets)


def test_funding_optimizer_skips_when_disabled(mock_config, temp_db):
    """Test that funding optimizer skips when disabled."""
    mock_config.optimizer.funding.enabled = False
    store = OHLCVStore(temp_db)
    optimizer = Optimizer(mock_config, store)
    
    result = optimizer.optimize_funding_strategy(
        symbols=['BTCUSDT'],
        timeframe='1h'
    )
    
    assert result.get('skipped') is True
    assert result.get('reason') == 'disabled'


def test_funding_optimizer_skips_when_strategy_disabled(mock_config, temp_db):
    """Test that funding optimizer skips when funding strategy is disabled."""
    mock_config.strategy.funding_opportunity.enabled = False
    store = OHLCVStore(temp_db)
    optimizer = Optimizer(mock_config, store)
    
    result = optimizer.optimize_funding_strategy(
        symbols=['BTCUSDT'],
        timeframe='1h'
    )
    
    assert 'error' in result
    assert 'not enabled' in result['error'].lower()

