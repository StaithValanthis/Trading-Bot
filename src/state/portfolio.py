"""Portfolio state tracking."""

from typing import Dict, Optional
from datetime import datetime

from ..exchange.bybit_client import BybitClient
from ..logging_utils import get_logger

logger = get_logger(__name__)


class PortfolioState:
    """Track current portfolio and account state."""
    
    def __init__(self, exchange_client: BybitClient):
        """
        Initialize portfolio state tracker.
        
        Args:
            exchange_client: Exchange client
        """
        self.exchange = exchange_client
        self.logger = get_logger(__name__)
        self.equity: float = 0.0
        self.positions: Dict[str, Dict] = {}
        self.last_update: Optional[datetime] = None
    
    def update(self):
        """Update portfolio state from exchange."""
        try:
            # Fetch balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {})
            self.equity = float(usdt_balance.get('total', 0))
            
            # Fetch positions
            positions = self.exchange.fetch_positions()
            
            self.positions = {}
            for pos in positions:
                symbol = pos.get('symbol', '').replace('/USDT', 'USDT')
                contracts = float(pos.get('contracts', 0))
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                notional = float(pos.get('notional', 0))
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                side = pos.get('side', 'long')
                
                if abs(contracts) > 0.001:  # Only include open positions
                    self.positions[symbol] = {
                        'contracts': contracts,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'notional': notional,
                        'unrealized_pnl': unrealized_pnl,
                        'side': side
                    }
            
            self.last_update = datetime.now()
            self.logger.debug(f"Portfolio updated: equity={self.equity:.2f}, positions={len(self.positions)}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in a symbol."""
        pos = self.positions.get(symbol)
        return pos is not None and abs(pos.get('contracts', 0)) > 0.001
    
    def get_total_notional(self) -> float:
        """Get total absolute notional value of all positions."""
        total = 0.0
        for pos in self.positions.values():
            total += abs(pos.get('notional', 0))
        return total
    
    def get_leverage(self) -> float:
        """Calculate current leverage."""
        if self.equity <= 0:
            return 0.0
        return self.get_total_notional() / self.equity

