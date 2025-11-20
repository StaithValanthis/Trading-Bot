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
                    # Extract SL/TP from position info if available
                    # Bybit may provide stopLoss/takeProfit in position info
                    position_info = pos.get('info', {})
                    stop_loss_price = position_info.get('stopLoss') or position_info.get('stopLossPrice')
                    take_profit_price = position_info.get('takeProfit') or position_info.get('takeProfitPrice')
                    
                    self.positions[symbol] = {
                        'contracts': contracts,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'notional': notional,
                        'unrealized_pnl': unrealized_pnl,
                        'side': side,
                        # SL/TP tracking (managed by bot, may not match exchange if using server-side orders)
                        'stop_loss_price': float(stop_loss_price) if stop_loss_price else None,
                        'take_profit_price': float(take_profit_price) if take_profit_price else None,
                        'stop_order_id': None,  # Set by OrderExecutor when placing stop order
                        'tp_order_id': None,  # Set by OrderExecutor when placing TP order
                        'entry_time': None,  # Set when position is opened (for time-based exits)
                        # For trailing stop
                        'highest_price': mark_price if side == 'long' else None,  # Track highest price for long trailing
                        'lowest_price': mark_price if side == 'short' else None,  # Track lowest price for short trailing
                    }
            
            self.last_update = datetime.now()
            
            if self.positions:
                self.logger.info(
                    f"Portfolio updated: equity=${self.equity:,.2f}, "
                    f"{len(self.positions)} open position(s): {', '.join(self.positions.keys())}"
                )
            else:
                self.logger.debug(f"Portfolio updated: equity=${self.equity:,.2f}, no open positions")
            
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
    
    def set_position_metadata(
        self,
        symbol: str,
        stop_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
        entry_time: Optional[datetime] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ):
        """
        Set metadata for a position (SL/TP order IDs, entry time, etc.).
        
        Args:
            symbol: Symbol name
            stop_order_id: Stop-loss order ID
            tp_order_id: Take-profit order ID
            entry_time: Entry timestamp
            stop_loss_price: Stop-loss price
            take_profit_price: Take-profit price
        """
        if symbol in self.positions:
            if stop_order_id is not None:
                self.positions[symbol]['stop_order_id'] = stop_order_id
            if tp_order_id is not None:
                self.positions[symbol]['tp_order_id'] = tp_order_id
            if entry_time is not None:
                self.positions[symbol]['entry_time'] = entry_time
            if stop_loss_price is not None:
                self.positions[symbol]['stop_loss_price'] = stop_loss_price
            if take_profit_price is not None:
                self.positions[symbol]['take_profit_price'] = take_profit_price
    
    def update_trailing_stop(
        self,
        symbol: str,
        new_stop_price: float,
        stop_order_id: Optional[str] = None
    ):
        """
        Update trailing stop price for a position.
        
        Args:
            symbol: Symbol name
            new_stop_price: New trailing stop price
            stop_order_id: Updated stop order ID (if order was modified)
        """
        if symbol in self.positions:
            self.positions[symbol]['stop_loss_price'] = new_stop_price
            if stop_order_id is not None:
                self.positions[symbol]['stop_order_id'] = stop_order_id

