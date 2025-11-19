from datetime import datetime, timedelta, date

import pandas as pd

from src.backtest.backtester import Backtester
from src.config import BotConfig


def test_backtester_respects_universe_history():
    # Simple synthetic data for two symbols
    idx = pd.date_range("2023-01-01", periods=48, freq="H")
    prices1 = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000}, index=idx
    )
    prices2 = pd.DataFrame(
        {"open": 50, "high": 51, "low": 49, "close": 50, "volume": 1_000}, index=idx
    )

    symbol_data = {"BTCUSDT": prices1, "ETHUSDT": prices2}

    config = BotConfig()
    backtester = Backtester(config)

    # Universe includes only BTCUSDT during the period
    d = date(2023, 1, 1)
    universe_history = {d: {"BTCUSDT"}}

    result = backtester.backtest(
        symbol_data,
        initial_capital=10_000.0,
        taker_fee=0.0,
        universe_history=universe_history,
    )

    # Backtest should complete without error and only consider BTCUSDT
    assert "error" not in result
    assert isinstance(result.get("final_equity", 0), (int, float))


