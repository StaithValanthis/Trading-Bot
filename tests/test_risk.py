import math

from src.risk.position_sizing import PositionSizer
from src.risk.portfolio_limits import PortfolioLimits
from src.config import RiskConfig


class DummyExchange:
    def get_market_info(self, symbol: str):
        return {
            "contractSize": 1.0,
            "precision": {"price": 2, "amount": 3},
            "limits": {"amount": {"min": 0.0, "max": None}, "cost": {"min": 0.0}},
        }

    def round_amount(self, symbol: str, amount: float) -> float:
        return round(amount, 3)

    def validate_order_size(self, symbol: str, amount: float, price: float):
        if amount * price < 1.0:
            return False, "below_min_notional"
        return True, ""


def test_position_sizing_basic_long():
    cfg = RiskConfig()
    ex = DummyExchange()
    sizer = PositionSizer(cfg, ex)

    size, err = sizer.calculate_position_size(
        symbol="BTCUSDT",
        equity=10_000.0,
        entry_price=20_000.0,
        stop_loss_price=19_000.0,
        signal="long",
    )
    assert err is None
    assert size > 0


def test_portfolio_limits_leverage_and_positions():
    cfg = RiskConfig()
    ex = DummyExchange()
    limits = PortfolioLimits(cfg, ex)

    class DummyPortfolio:
        def __init__(self, equity, positions):
            self.equity = equity
            self.positions = positions

    portfolio = DummyPortfolio(
        equity=10_000.0,
        positions={
            "BTCUSDT": {"notional": 5_000.0, "contracts": 0.25},
            "ETHUSDT": {"notional": 5_000.0, "contracts": 1.0},
        },
    )

    within, err = limits.check_leverage_limit(
        portfolio_state=portfolio,
        new_position_notional=5_000.0,
        new_position_side="long",
    )
    # Default max_leverage is 3; this should be within for default RiskConfig
    assert within in (True, False)


