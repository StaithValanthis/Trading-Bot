"""Order execution logic."""

import time
from typing import Dict, Optional, List
from datetime import datetime, timezone

from ..exchange.bybit_client import BybitClient
from ..state.portfolio import PortfolioState
from ..data.trades_store import TradesStore, TradeRecord
from ..data.orders_store import OrdersStore, OrderRecord
from ..config import RiskConfig, TrendStrategyConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class OrderExecutor:
    """Execute trading orders."""

    def __init__(
        self,
        exchange_client: BybitClient,
        trades_store: Optional[TradesStore] = None,
        orders_store: Optional[OrdersStore] = None,
        risk_config: Optional[RiskConfig] = None,
        strategy_config: Optional[TrendStrategyConfig] = None,
    ):
        """
        Initialize order executor.

        Args:
            exchange_client: Exchange client
            trades_store: Optional trades store for logging closed trades
            orders_store: Optional orders store for logging order execution
            risk_config: Optional risk config for SL/TP behavior
            strategy_config: Optional strategy config for TP/trailing settings
        """
        self.exchange = exchange_client
        self.trades_store = trades_store
        self.orders_store = orders_store
        self.risk_config = risk_config
        self.strategy_config = strategy_config
        self.logger = get_logger(__name__)
        self.recent_orders: Dict[str, float] = {}  # symbol -> last order timestamp
    
    def _place_stop_loss_order(
        self,
        symbol: str,
        position_side: str,  # 'long' or 'short'
        size: float,
        stop_loss_price: float,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> Optional[str]:
        """
        Place a stop-loss order for a position.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side ('long' or 'short')
            size: Position size in contracts
            stop_loss_price: Stop-loss trigger price
            portfolio_state: Optional portfolio state to update with order ID
        
        Returns:
            Stop order ID if successful, None otherwise
        """
        if not self.risk_config or not self.risk_config.use_server_side_stops:
            self.logger.debug(f"Skipping server-side stop for {symbol} (use_server_side_stops=False)")
            return None
        
        if stop_loss_price is None:
            self.logger.warning(f"No stop-loss price provided for {symbol}, skipping stop order")
            return None
        
        try:
            # Determine stop order side (opposite of position side)
            stop_side = "sell" if position_side == "long" else "buy"
            
            # Round stop price to exchange precision
            stop_price_rounded = self.exchange.round_price(symbol, stop_loss_price)
            
            # Determine stop order type
            stop_order_type = self.risk_config.stop_order_type if self.risk_config else "stop_market"
            
            self.logger.info(
                f"Placing stop-loss order for {symbol}: {stop_side} {size} @ trigger={stop_price_rounded} "
                f"(type={stop_order_type})"
            )
            
            # Place stop order
            limit_price = stop_price_rounded if stop_order_type == "stop_limit" else None
            stop_order = self.exchange.create_stop_order(
                symbol=symbol,
                side=stop_side,
                amount=size,
                trigger_price=stop_price_rounded,
                order_type="market" if stop_order_type == "stop_market" else "limit",
                limit_price=limit_price,
                reduce_only=True,
            )
            
            stop_order_id = stop_order.get("id")
            if stop_order_id and portfolio_state:
                portfolio_state.set_position_metadata(
                    symbol,
                    stop_order_id=stop_order_id,
                    stop_loss_price=stop_price_rounded,
                )
            
            self.logger.info(f"Stop-loss order placed for {symbol}: order_id={stop_order_id}")
            return stop_order_id
            
        except NotImplementedError as e:
            self.logger.warning(
                f"Server-side stop orders not supported: {e}. "
                "Falling back to strategy-based exit only."
            )
            return None
        except Exception as e:
            self.logger.error(f"Error placing stop-loss order for {symbol}: {e}")
            return None
    
    def _place_take_profit_order(
        self,
        symbol: str,
        position_side: str,  # 'long' or 'short'
        size: float,
        take_profit_price: float,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> Optional[str]:
        """
        Place a take-profit order for a position.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side ('long' or 'short')
            size: Position size in contracts
            take_profit_price: Take-profit trigger price
            portfolio_state: Optional portfolio state to update with order ID
        
        Returns:
            TP order ID if successful, None otherwise
        """
        if take_profit_price is None:
            return None
        
        try:
            # Determine TP order side (opposite of position side)
            tp_side = "sell" if position_side == "long" else "buy"
            
            # Round TP price to exchange precision
            tp_price_rounded = self.exchange.round_price(symbol, take_profit_price)
            
            self.logger.info(
                f"Placing take-profit order for {symbol}: {tp_side} {size} @ trigger={tp_price_rounded}"
            )
            
            # Place TP order (usually limit order)
            tp_order = self.exchange.create_take_profit_order(
                symbol=symbol,
                side=tp_side,
                amount=size,
                trigger_price=tp_price_rounded,
                order_type="limit",
                limit_price=tp_price_rounded,
                reduce_only=True,
            )
            
            tp_order_id = tp_order.get("id")
            if tp_order_id and portfolio_state:
                portfolio_state.set_position_metadata(
                    symbol,
                    tp_order_id=tp_order_id,
                    take_profit_price=tp_price_rounded,
                )
            
            self.logger.info(f"Take-profit order placed for {symbol}: order_id={tp_order_id}")
            return tp_order_id
            
        except NotImplementedError as e:
            self.logger.warning(f"Server-side take-profit orders not supported: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error placing take-profit order for {symbol}: {e}")
            return None
    
    def _cancel_stop_orders(self, symbol: str, portfolio_state: Optional[PortfolioState] = None):
        """
        Cancel existing stop-loss and take-profit orders for a symbol.
        
        Args:
            symbol: Trading pair symbol
            portfolio_state: Optional portfolio state to get order IDs from
        """
        if not portfolio_state or symbol not in portfolio_state.positions:
            return
        
        pos = portfolio_state.positions[symbol]
        stop_order_id = pos.get('stop_order_id')
        tp_order_id = pos.get('tp_order_id')
        
        if stop_order_id:
            try:
                self.exchange.cancel_order(stop_order_id, symbol)
                self.logger.info(f"Cancelled stop-loss order {stop_order_id} for {symbol}")
                if portfolio_state:
                    portfolio_state.set_position_metadata(symbol, stop_order_id=None)
            except Exception as e:
                self.logger.warning(f"Error cancelling stop-loss order for {symbol}: {e}")
        
        if tp_order_id:
            try:
                self.exchange.cancel_order(tp_order_id, symbol)
                self.logger.info(f"Cancelled take-profit order {tp_order_id} for {symbol}")
                if portfolio_state:
                    portfolio_state.set_position_metadata(symbol, tp_order_id=None)
            except Exception as e:
                self.logger.warning(f"Error cancelling take-profit order for {symbol}: {e}")
    
    def execute_position_change(
        self,
        symbol: str,
        target_size: float,  # Target position size in contracts (can be negative for short)
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        signal: str = 'long',  # 'long' or 'short'
        portfolio_state: Optional[PortfolioState] = None,
    ) -> Dict:
        """
        Execute a position change to reach target size.
        
        Args:
            symbol: Trading pair symbol
            target_size: Target position size (positive for long, negative for short)
            entry_price: Entry price
            stop_loss_price: Optional stop loss price
            signal: Signal direction
        
        Returns:
            Dictionary with execution results
        """
        # Rate limiting: don't place orders too frequently for same symbol
        current_time = time.time()
        last_order_time = self.recent_orders.get(symbol, 0)
        min_interval = 5.0  # 5 seconds between orders for same symbol
        
        if current_time - last_order_time < min_interval:
            self.logger.warning(
                f"Rate limit: skipping order for {symbol} "
                f"(last order {current_time - last_order_time:.1f}s ago)"
            )
            return {'status': 'skipped', 'reason': 'rate_limit'}
        
        # Determine side and absolute size
        if target_size > 0:
            side = 'buy'
            size = abs(target_size)
        elif target_size < 0:
            side = 'sell'
            size = abs(target_size)
        else:
            # target_size == 0, close position
            return self.close_position(symbol)
        
        # Validate order size
        is_valid, error_msg = self.exchange.validate_order_size(
            symbol,
            size,
            entry_price
        )
        
        if not is_valid:
            self.logger.warning(f"Invalid order size for {symbol}: {error_msg}")
            return {'status': 'rejected', 'reason': error_msg}
        
        try:
            # Place market order (can be changed to limit if needed)
            order = self.exchange.create_order(
                symbol,
                side,
                size,
                order_type='market',
                price=None
            )
            
            self.recent_orders[symbol] = current_time
            
            self.logger.info(
                f"Executed {side} order for {symbol}: {size} contracts @ {entry_price}"
            )

            # Log order execution if store is available
            if self.orders_store is not None:
                self.orders_store.log_order(
                    OrderRecord(
                        symbol=symbol,
                        side=side,
                        size=size,
                        price=entry_price,
                        order_type="market",
                        reason="reconcile",
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            
            # Place stop-loss order if configured and stop price provided
            stop_order_id = None
            tp_order_id = None
            
            # Determine position side from signal
            position_side = signal  # 'long' or 'short'
            
            # Place stop-loss order
            if stop_loss_price:
                stop_order_id = self._place_stop_loss_order(
                    symbol,
                    position_side,
                    size,
                    stop_loss_price,
                    portfolio_state,
                )
            
            # Place take-profit order if configured
            if take_profit_price and self.strategy_config:
                tp_order_id = self._place_take_profit_order(
                    symbol,
                    position_side,
                    size,
                    take_profit_price,
                    portfolio_state,
                )
            
            # Set entry time in portfolio state if provided
            if portfolio_state:
                portfolio_state.set_position_metadata(
                    symbol,
                    entry_time=datetime.now(timezone.utc),
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
            
            return {
                'status': 'filled',
                'order': order,
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': entry_price,
                'stop_order_id': stop_order_id,
                'tp_order_id': tp_order_id,
            }
            
        except Exception as e:
            self.logger.error(f"Error executing order for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close an existing position.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dictionary with execution results
        """
        # Fetch current position
        positions = self.exchange.fetch_positions(symbol=symbol)

        for pos in positions:
            if pos.get("symbol") == symbol or pos.get("symbol") == symbol.replace(
                "USDT", "/USDT"
            ):
                contracts = float(pos.get("contracts", 0))

                if abs(contracts) < 0.001:  # Already closed
                    return {"status": "no_position"}

                # Determine side to close (relative to current position)
                side = "sell" if contracts > 0 else "buy"
                size = abs(contracts)
                entry_price = float(pos.get("entryPrice", 0.0))
                mark_price = float(pos.get("markPrice", entry_price))
                position_side = pos.get("side", "long" if contracts > 0 else "short")
                entry_time_str = pos.get("timestamp") or pos.get("datetime")
                try:
                    entry_time = (
                        datetime.fromisoformat(entry_time_str)
                        if isinstance(entry_time_str, str)
                        else datetime.now(timezone.utc)
                    )
                except Exception:
                    entry_time = datetime.now(timezone.utc)

                try:
                    order = self.exchange.create_order(
                        symbol,
                        side,
                        size,
                        order_type="market",
                    )

                    exit_time = datetime.now(timezone.utc)
                    exit_price = mark_price

                    # Approximate PnL for the round-trip
                    if position_side == "long":
                        pnl = (exit_price - entry_price) * size
                    else:
                        pnl = (entry_price - exit_price) * size

                    self.logger.info(f"Closed position for {symbol}: {size} contracts")

                    # Log order for close if store available
                    if self.orders_store is not None:
                        self.orders_store.log_order(
                            OrderRecord(
                                symbol=symbol,
                                side=side,
                                size=size,
                                price=exit_price,
                                order_type="market",
                                reason="close_position",
                                timestamp=exit_time,
                            )
                        )

                    # Log trade if store is available
                    if self.trades_store is not None:
                        trade = TradeRecord(
                            symbol=symbol,
                            side=position_side,
                            size=size,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            entry_time=entry_time,
                            exit_time=exit_time,
                            pnl=pnl,
                            reason="close_position",
                        )
                        self.trades_store.log_trade(trade)

                    return {
                        "status": "closed",
                        "order": order,
                        "symbol": symbol,
                        "size": size,
                    }

                except Exception as e:
                    self.logger.error(f"Error closing position for {symbol}: {e}")
                    return {"status": "error", "error": str(e)}

        return {"status": "no_position"}
    
    def reconcile_positions(
        self,
        portfolio_state: PortfolioState,
        target_positions: Dict[str, Dict]  # symbol -> {size, signal, entry_price, stop_loss}
    ) -> List[Dict]:
        """
        Reconcile current positions with target positions and execute changes.
        
        This method ensures that existing positions opened by previous bot runs
        (or manually) are properly managed according to the current strategy.
        
        Args:
            portfolio_state: Current portfolio state (includes existing positions)
            target_positions: Dictionary of target positions from strategy
        
        Returns:
            List of execution results
        """
        results = []
        
        # Get current positions from portfolio state
        current_positions = {}
        for symbol, pos in portfolio_state.positions.items():
            contracts = pos.get('contracts', 0)
            side = pos.get('side', 'long')
            if abs(contracts) > 0.001:
                # Store signed contracts (positive for long, negative for short)
                signed_contracts = abs(contracts) if side == 'long' else -abs(contracts)
                current_positions[symbol] = signed_contracts
                
                if symbol in target_positions:
                    target = target_positions[symbol]
                    target_size = target.get('size', 0)
                    self.logger.info(
                        f"Reconciling {symbol}: current={signed_contracts:.4f}, "
                        f"target={target_size:.4f}"
                    )
        
        # Log positions that will be closed (not in target)
        positions_to_close = [s for s in current_positions if s not in target_positions]
        if positions_to_close:
            self.logger.info(
                f"Closing {len(positions_to_close)} position(s) not in target portfolio: {positions_to_close}"
            )
        
        # Close positions not in target
        for symbol in positions_to_close:
            pos = portfolio_state.positions.get(symbol, {})
            side = pos.get('side', 'long')
            contracts = abs(pos.get('contracts', 0))
            self.logger.info(
                f"Closing {symbol}: {side.upper()} {contracts:.4f} contracts "
                f"(not in target portfolio)"
            )
            
            # Cancel stop/TP orders before closing
            self._cancel_stop_orders(symbol, portfolio_state)
            
            result = self.close_position(symbol)
            results.append(result)
        
        # Open or adjust positions
        for symbol, target in target_positions.items():
            target_size = target.get('size', 0)
            current_size = current_positions.get(symbol, 0)
            
            # Only execute if size difference is significant (>1% of target or >0.001 contracts)
            size_diff = abs(target_size - current_size)
            if size_diff < 0.001 and abs(target_size) < 0.001:
                # Both are effectively zero, skip
                continue
            
            if size_diff < max(0.001, abs(target_size) * 0.01):
                # Size difference is small, skip to avoid over-trading
                continue
            
            # If position already exists, cancel old stop/TP orders before adjusting
            if symbol in portfolio_state.positions:
                self._cancel_stop_orders(symbol, portfolio_state)
            
            # Calculate take-profit price if configured
            take_profit_price = None
            if self.strategy_config and self.strategy_config.take_profit_rr:
                stop_loss = target.get('stop_loss')
                entry_price = target.get('entry_price', 0)
                if stop_loss and entry_price:
                    stop_distance = abs(entry_price - stop_loss)
                    signal = target.get('signal', 'long')
                    if signal == 'long':
                        take_profit_price = entry_price + (stop_distance * self.strategy_config.take_profit_rr)
                    else:  # short
                        take_profit_price = entry_price - (stop_distance * self.strategy_config.take_profit_rr)
            
            # Execute position change
            result = self.execute_position_change(
                symbol,
                target_size,
                target.get('entry_price', 0),
                target.get('stop_loss'),
                take_profit_price,
                target.get('signal', 'long'),
                portfolio_state,
            )
            results.append(result)
        
        return results

