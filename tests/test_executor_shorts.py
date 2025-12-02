"""Test that executor correctly opens short positions."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.execution.executor import OrderExecutor
from src.config import RiskConfig, TrendStrategyConfig


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = Mock()
    exchange.create_order = MagicMock(return_value={'id': 'test_order_123', 'status': 'closed'})
    exchange.fetch_positions = MagicMock(return_value=[])
    exchange.get_market_info = MagicMock(return_value={
        'contractSize': 1.0,
        'precision': {'amount': 8, 'price': 2},
        'limits': {'amount': {'min': 0.001, 'max': None}, 'cost': {'min': 5.0}}
    })
    exchange.round_amount = MagicMock(side_effect=lambda s, a: round(a, 8))
    exchange.round_price = MagicMock(side_effect=lambda s, p: round(p, 2))
    exchange.validate_order_size = MagicMock(return_value=(True, None))
    return exchange


def test_executor_opens_short_position(mock_exchange):
    """Test that executor correctly opens short positions with negative size."""
    executor = OrderExecutor(mock_exchange)
    
    # Execute short position (negative size)
    result = executor.execute_position_change(
        symbol='BTCUSDT',
        target_size=-0.1,  # Negative = short
        entry_price=50000,
        signal='short'
    )
    
    assert result['status'] in ['opened', 'partial'], (
        f"Expected opened or partial, got {result['status']}. "
        f"Result: {result}"
    )
    
    # Verify order was placed with correct side
    if mock_exchange.create_order.called:
        call_args = mock_exchange.create_order.call_args
        assert call_args is not None, "create_order should have been called"
        
        # Check side (should be 'sell' for short)
        order_params = call_args[1] if len(call_args) > 1 else call_args[0]
        assert order_params.get('side') == 'sell', (
            f"Short position should use 'sell' side, got {order_params.get('side')}"
        )
        assert order_params.get('amount', 0) > 0, (
            f"Amount should be positive (size is already negative), got {order_params.get('amount')}"
        )


def test_executor_opens_long_position(mock_exchange):
    """Test that executor correctly opens long positions with positive size."""
    executor = OrderExecutor(mock_exchange)
    
    # Execute long position (positive size)
    result = executor.execute_position_change(
        symbol='BTCUSDT',
        target_size=0.1,  # Positive = long
        entry_price=50000,
        signal='long'
    )
    
    assert result['status'] in ['opened', 'partial'], (
        f"Expected opened or partial, got {result['status']}. "
        f"Result: {result}"
    )
    
    # Verify order was placed with correct side
    if mock_exchange.create_order.called:
        call_args = mock_exchange.create_order.call_args
        order_params = call_args[1] if len(call_args) > 1 else call_args[0]
        assert order_params.get('side') == 'buy', (
            f"Long position should use 'buy' side, got {order_params.get('side')}"
        )


def test_executor_places_short_stop_loss(mock_exchange):
    """Test that executor places stop loss orders correctly for short positions."""
    from src.state.portfolio import PortfolioState
    
    risk_config = RiskConfig(use_server_side_stops=True, stop_order_type='stop_market')
    executor = OrderExecutor(mock_exchange, risk_config=risk_config)
    
    portfolio = PortfolioState(mock_exchange)
    
    # Place stop loss for short position
    stop_id = executor._place_stop_loss_order(
        symbol='BTCUSDT',
        position_side='short',
        size=0.1,
        stop_loss_price=51000,  # Above entry (for short)
        portfolio_state=portfolio,
        entry_price=50000
    )
    
    assert stop_id is not None, "Stop loss order should be placed"
    
    # Verify stop order side (should be 'buy' to close short)
    if mock_exchange.create_order.called:
        call_args = mock_exchange.create_order.call_args
        order_params = call_args[1] if len(call_args) > 1 else call_args[0]
        assert order_params.get('side') == 'buy', (
            f"Short stop loss should use 'buy' side, got {order_params.get('side')}"
        )

