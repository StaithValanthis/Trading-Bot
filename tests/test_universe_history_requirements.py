from datetime import datetime, timedelta

import pandas as pd

from src.config import UniverseConfig
from src.universe.selector import UniverseSelector


class DummyOHLCVStore:
    def __init__(self, df_by_symbol_timeframe):
        self._data = df_by_symbol_timeframe

    def get_ohlcv(self, symbol, timeframe, limit=None):
        df = self._data.get((symbol, timeframe), pd.DataFrame())
        if limit is not None and not df.empty:
            return df.tail(limit)
        return df


class DummyUniverseStore:
    def __init__(self):
        self._meta = {}

    def get_symbol_metadata(self, symbol):
        # Pretend symbol exists on exchange so we treat empty data as insufficient_history
        return {"symbol": symbol}

    def get_warmup_status(self, symbol):
        return None

    def track_warmup(self, symbol, first_seen, warmup_start):
        # No-op for tests
        pass


class DummyExchange:
    def fetch_markets(self):
        return {}


def _make_hourly_df(start: datetime, days: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=days * 24, freq="H")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1_000}, index=idx
    )


def test_insufficient_history_when_below_min_days():
    """Symbols with < min_history_days worth of data should be rejected."""
    cfg = UniverseConfig(min_history_days=30, history_buffer_days=5)
    symbol = "TESTUSDT"
    timeframe = "1h"

    df = _make_hourly_df(datetime(2023, 1, 1), days=20)  # 20 < 30 days
    store = DummyOHLCVStore({(symbol, timeframe): df})
    selector = UniverseSelector(cfg, DummyExchange(), store, DummyUniverseStore())

    ok, reason, meta = selector.check_historical_data(symbol, timeframe)

    assert not ok
    assert reason == "insufficient_history"
    assert meta["history_days"] < cfg.min_history_days


def test_sufficient_history_when_above_min_days():
    """Symbols with >= min_history_days worth of data and small gaps should pass."""
    cfg = UniverseConfig(min_history_days=30, history_buffer_days=5, max_data_gap_pct=10.0)
    symbol = "TESTUSDT"
    timeframe = "1h"

    # 34 days of hourly data, with a few gaps
    df = _make_hourly_df(datetime(2023, 1, 1), days=34)
    # Drop a handful of rows to simulate minor gaps (< max_data_gap_pct)
    df = df.drop(df.index[0:5])

    store = DummyOHLCVStore({(symbol, timeframe): df})
    selector = UniverseSelector(cfg, DummyExchange(), store, DummyUniverseStore())

    ok, reason, meta = selector.check_historical_data(symbol, timeframe)

    assert ok
    assert reason is None
    assert meta["history_days"] >= cfg.min_history_days


