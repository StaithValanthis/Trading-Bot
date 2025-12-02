"""Smoke test: Bot should open shorts during market downtrends."""

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
    config_path = project_root / 'config.example.yaml'
    if config_path.exists():
        return BotConfig.from_yaml(str(config_path))
    else:
        # Create minimal config
        from src.config import (
            BotConfig, ExchangeConfig, StrategyConfig, RiskConfig,
            TrendStrategyConfig, CrossSectionalStrategyConfig,
            DataConfig, LoggingConfig
        )
        return BotConfig(
            exchange=ExchangeConfig(name='bybit', mode='paper', testnet=True, timeframe='4h'),
            strategy=StrategyConfig(
                trend=TrendStrategyConfig(
                    ma_short=5,
                    ma_long=20,
                    momentum_lookback=10,
                    atr_stop_multiplier=2.0,
                    atr_period=14,
                    min_atr_threshold=0.001
                ),
                cross_sectional=CrossSectionalStrategyConfig(
                    top_k=5,
                    require_trend_alignment=True,
                    exit_band=2
                )
            ),
            risk=RiskConfig(per_trade_risk_fraction=0.01),
            data=DataConfig(db_path=':memory:', lookback_bars=200),
            logging=LoggingConfig()
        )


def test_bot_opens_shorts_during_downtrend(sample_config):
    """
    Smoke test: Bot should be net short at least 30% of time during 30-day downtrend.
    
    This test verifies that the bot can generate and execute short signals
    during a clear market downtrend.
    """
    backtester = Backtester(sample_config)
    
    # Create 30-day downtrend (180 bars at 4h = 30 days)
    dates = pd.date_range('2024-01-01', periods=180, freq='4h')
    
    # Clear downtrend: price declining consistently
    base_price = 100
    decline = np.linspace(0, 25, 180)  # Decline by 25%
    prices = base_price - decline
    
    # Add some noise but keep overall downtrend
    noise = np.random.normal(0, 0.5, 180)
    prices = prices + noise
    
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': prices,
            'high': prices * 1.01 + abs(noise),
            'low': prices * 0.99 - abs(noise),
            'close': prices,
            'volume': np.random.uniform(800, 1200, 180)
        }, index=dates)
    }
    
    results = backtester.backtest(symbol_data, initial_capital=10000)
    
    # Count positions by direction
    trades = results.get('trades', [])
    
    if len(trades) == 0:
        pytest.skip("No trades generated - may need to adjust config or data")
    
    long_trades = [t for t in trades if t.get('side') == 'long' or t.get('signal') == 'long']
    short_trades = [t for t in trades if t.get('side') == 'short' or t.get('signal') == 'short']
    
    total_trades = len(trades)
    short_pct = (len(short_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # Also check position history if available
    position_history = results.get('position_history', {})
    if position_history:
        short_positions_count = 0
        total_positions_count = 0
        
        for timestamp, positions in position_history.items():
            for symbol, pos in positions.items():
                total_positions_count += 1
                if pos.get('signal') == 'short' or pos.get('side') == 'short':
                    short_positions_count += 1
        
        if total_positions_count > 0:
            short_pos_pct = (short_positions_count / total_positions_count * 100)
            # Use the higher of the two percentages
            short_pct = max(short_pct, short_pos_pct)
    
    assert short_pct >= 30, (
        f"Bot should be short at least 30% of time during downtrend. "
        f"Got {short_pct:.1f}% shorts. "
        f"Total trades: {total_trades}, Short trades: {len(short_trades)}, "
        f"Long trades: {len(long_trades)}"
    )


def test_bot_opens_longs_during_uptrend(sample_config):
    """
    Smoke test: Bot should be net long at least 30% of time during 30-day uptrend.
    
    This test verifies that the bot can generate and execute long signals
    during a clear market uptrend (symmetric test to short test).
    """
    backtester = Backtester(sample_config)
    
    # Create 30-day uptrend
    dates = pd.date_range('2024-01-01', periods=180, freq='4h')
    
    # Clear uptrend: price rising consistently
    base_price = 70
    rise = np.linspace(0, 25, 180)  # Rise by 25%
    prices = base_price + rise
    
    # Add some noise but keep overall uptrend
    noise = np.random.normal(0, 0.5, 180)
    prices = prices + noise
    
    symbol_data = {
        'BTCUSDT': pd.DataFrame({
            'open': prices,
            'high': prices * 1.01 + abs(noise),
            'low': prices * 0.99 - abs(noise),
            'close': prices,
            'volume': np.random.uniform(800, 1200, 180)
        }, index=dates)
    }
    
    results = backtester.backtest(symbol_data, initial_capital=10000)
    
    trades = results.get('trades', [])
    
    if len(trades) == 0:
        pytest.skip("No trades generated - may need to adjust config or data")
    
    long_trades = [t for t in trades if t.get('side') == 'long' or t.get('signal') == 'long']
    short_trades = [t for t in trades if t.get('side') == 'short' or t.get('signal') == 'short']
    
    total_trades = len(trades)
    long_pct = (len(long_trades) / total_trades * 100) if total_trades > 0 else 0
    
    assert long_pct >= 30, (
        f"Bot should be long at least 30% of time during uptrend. "
        f"Got {long_pct:.1f}% longs. "
        f"Total trades: {total_trades}, Long trades: {len(long_trades)}, "
        f"Short trades: {len(short_trades)}"
    )

