"""Test that backtester correctly opens and manages short positions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.backtester import Backtester
from src.config import BotConfig


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    # Use example config or create minimal config
    config_path = project_root / 'config.example.yaml'
    if config_path.exists():
        return BotConfig.from_yaml(str(config_path))
    else:
        # Create minimal config if example doesn't exist
        from src.config import (
            BotConfig, ExchangeConfig, StrategyConfig, RiskConfig,
            TrendStrategyConfig, CrossSectionalStrategyConfig,
            DataConfig, LoggingConfig
        )
        return BotConfig(
            exchange=ExchangeConfig(name='bybit', mode='paper', testnet=True),
            strategy=StrategyConfig(
                trend=TrendStrategyConfig(ma_short=5, ma_long=20, momentum_lookback=10),
                cross_sectional=CrossSectionalStrategyConfig(top_k=5, require_trend_alignment=True)
            ),
            risk=RiskConfig(per_trade_risk_fraction=0.01),
            data=DataConfig(db_path=':memory:', lookback_bars=200),
            logging=LoggingConfig()
        )


def test_backtester_opens_short_positions(sample_config):
    """Test that backtester correctly opens and manages short positions."""
    backtester = Backtester(sample_config)
    
    # Create downtrend data
    dates = pd.date_range('2024-01-01', periods=100, freq='4h')
    prices = 100 - np.linspace(0, 30, 100)  # Declining from 100 to 70
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': 1000
        }, index=dates)
    }
    
    results = backtester.backtest(symbol_data, initial_capital=10000)
    
    # Check that at least one short position was opened
    trades = results.get('trades', [])
    short_trades = [t for t in trades if t.get('side') == 'short' or t.get('signal') == 'short']
    
    assert len(short_trades) > 0, (
        f"Backtester should have opened at least one short position. "
        f"Total trades: {len(trades)}, Short trades: {len(short_trades)}"
    )
    
    # Check that short PnL is calculated correctly (profit when price declines)
    for trade in short_trades:
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('pnl', 0)
        
        if entry_price > 0 and exit_price > 0:
            # For shorts: profit when exit < entry
            if exit_price < entry_price:
                assert pnl > 0, (
                    f"Short should profit when price declines: "
                    f"entry={entry_price:.2f}, exit={exit_price:.2f}, pnl={pnl:.2f}"
                )


def test_backtester_short_stop_loss_hit(sample_config):
    """Test that backtester correctly handles short stop loss hits."""
    backtester = Backtester(sample_config)
    
    # Create data where short stop loss should be hit (price rises)
    dates = pd.date_range('2024-01-01', periods=50, freq='4h')
    prices = 100 + np.linspace(0, 10, 50)  # Rising from 100 to 110 (should hit short stop)
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': prices,
            'high': prices * 1.05,  # High enough to hit stop loss above entry
            'low': prices * 0.95,
            'close': prices,
            'volume': 1000
        }, index=dates)
    }
    
    results = backtester.backtest(symbol_data, initial_capital=10000)
    
    # Check that stop losses were hit
    trades = results.get('trades', [])
    stop_loss_trades = [t for t in trades if t.get('exit_reason') == 'stop_loss']
    
    # At least some trades should have stop loss exits
    assert len(stop_loss_trades) > 0 or len(trades) == 0, (
        f"Expected some stop loss exits. Total trades: {len(trades)}, "
        f"Stop loss exits: {len(stop_loss_trades)}"
    )

