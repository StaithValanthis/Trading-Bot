import types

import ccxt

from src.exchange.bybit_client import BybitClient
from src.config import ExchangeConfig


class DummyBybit:
    def __init__(self, *args, **kwargs):
        self.rateLimit = 0

    # Minimal methods used by BybitClient in tests
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [[0, 1, 1, 1, 1, 1]]

    def fetch_balance(self, params=None):
        return {"USDT": {"free": 1000, "used": 0, "total": 1000}}

    def fetch_positions(self, symbols=None):
        return []

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0001, "nextFundingTime": None}

    def create_order(self, *args, **kwargs):
        return {"id": "dummy", "status": "closed"}

    def cancel_order(self, order_id, symbol):
        return {"id": order_id, "status": "canceled"}

    def load_markets(self):
        return {
            "BTC/USDT": {
                "precision": {"price": 2, "amount": 3},
                "limits": {
                    "amount": {"min": 0.001, "max": None},
                    "cost": {"min": 5.0},
                },
                "contractSize": 1.0,
            }
        }


def test_bybit_client_basic_calls(monkeypatch):
    # Monkeypatch ccxt.bybit to our dummy
    monkeypatch.setattr(ccxt, "bybit", DummyBybit)

    cfg = ExchangeConfig(name="bybit", mode="paper", testnet=True, api_key="", api_secret="")
    client = BybitClient(cfg)

    # These calls should not raise and should return basic structures
    ohlcv = client.fetch_ohlcv("BTCUSDT", "1h", limit=1)
    assert len(ohlcv) == 1

    balance = client.fetch_balance()
    assert "USDT" in balance

    positions = client.fetch_positions()
    assert positions == []

    funding = client.fetch_funding_rate("BTCUSDT")
    assert "fundingRate" in funding

    market_info = client.get_market_info("BTCUSDT")
    assert "precision" in market_info
    assert "limits" in market_info


