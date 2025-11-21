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
        entry_price: Optional[float] = None,  # Entry price for validation
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
            
            # Use entry price or current mark price as reference for triggerDirection validation
            current_price_ref = entry_price
            if not current_price_ref and portfolio_state:
                pos = portfolio_state.get_position(symbol)
                if pos:
                    current_price_ref = pos.get('mark_price') or pos.get('entry_price')
            
            self.logger.info(
                f"Placing stop-loss order for {symbol}: {stop_side} {size} @ trigger={stop_price_rounded} "
                f"(type={stop_order_type}, entry_ref={current_price_ref})"
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
                current_price=current_price_ref,  # Pass current price reference for validation
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
            
            # Use entry price or current mark price as reference for triggerDirection validation
            current_price_ref = None
            if portfolio_state:
                pos = portfolio_state.get_position(symbol)
                if pos:
                    current_price_ref = pos.get('mark_price') or pos.get('entry_price')
            
            self.logger.info(
                f"Placing take-profit order for {symbol}: {tp_side} {size} @ trigger={tp_price_rounded} "
                f"(entry_ref={current_price_ref})"
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
                current_price=current_price_ref,  # Pass current price reference for validation
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
        
        # CRITICAL: If a position already exists for this symbol, skip placing a new order entirely
        # This prevents stacking positions regardless of size differences.
        # Size differences (due to profit, equity changes, etc.) should NOT trigger new orders.
        # Only close/reverse if direction changes or signal goes flat (handled in reconcile_positions).
        if portfolio_state:
            existing_pos = portfolio_state.get_position(symbol)
            if existing_pos:
                existing_contracts = existing_pos.get('contracts', 0)
                existing_side = existing_pos.get('side', 'long')
                # Convert to signed size for comparison
                existing_signed_size = abs(existing_contracts) if existing_side == 'long' else -abs(existing_contracts)
                
                # Check if direction matches (same sign = same direction)
                same_direction = (
                    (target_size > 0 and existing_signed_size > 0) or
                    (target_size < 0 and existing_signed_size < 0)
                )
                
                if same_direction:
                    # Position exists with same direction - skip entirely regardless of size difference
                    # Size differences due to profit/equity changes should NOT trigger new orders
                    self.logger.info(
                        f"Skipping {symbol}: position already exists (same direction). "
                        f"Current={existing_signed_size:.4f}, Target={target_size:.4f}. "
                        f"Size difference ignored - no new order to prevent stacking."
                    )
                    return {
                        'status': 'skipped',
                        'symbol': symbol,
                        'reason': 'position_already_exists_same_direction',
                        'current_size': existing_signed_size,
                        'target_size': target_size
                    }
                else:
                    # Direction changed - position should be closed first in reconcile_positions
                    # But if we reach here, it means reconciliation didn't close it - skip to prevent stacking
                    self.logger.warning(
                        f"Direction mismatch for {symbol}: existing={existing_signed_size:.4f} ({existing_side}), "
                        f"target={target_size:.4f}. Position should be closed first. Skipping to prevent stacking."
                    )
                    return {
                        'status': 'skipped',
                        'symbol': symbol,
                        'reason': 'position_exists_opposite_direction',
                        'current_size': existing_signed_size,
                        'target_size': target_size
                    }
        
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
            
            order_id = order.get('id')
            order_status = order.get('status')
            
            # CCXT may return status=None for market orders that fill instantly
            # This is normal - we'll verify by checking position existence
            status_str = str(order_status) if order_status is not None else 'None (likely filled instantly)'
            
            self.logger.info(
                f"Executed {side} order for {symbol}: {size} contracts @ {entry_price}, "
                f"order_id={order_id}, status={status_str}"
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
            
            # CRITICAL: Wait for entry order to be filled before placing SL/TP orders
            # Market orders should fill immediately, but position may take a moment to appear
            # CCXT may return status=None for market orders that fill instantly
            # So we check position existence rather than relying on order status
            
            # Wait and retry position verification (up to 20 seconds total for small positions)
            # Small positions (like BTC minimum 0.001) may take longer to appear in portfolio
            position_verified = False
            actual_size = size  # Use target size as default
            max_retries = 10  # Increased from 5 to handle small positions
            retry_delay = 1.0  # Start with 1 second delay
            
            for attempt in range(max_retries):
                if attempt > 0:
                    self.logger.debug(
                        f"Position verification attempt {attempt + 1}/{max_retries} for {symbol} "
                        f"(waiting {retry_delay}s)..."
                    )
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 3.0)  # Exponential backoff, max 3s
                
                # Try portfolio state first (cached)
                if portfolio_state:
                    portfolio_state.update()  # Refresh positions from exchange
                    position_verified = portfolio_state.has_position(symbol)
                    if position_verified:
                        pos = portfolio_state.get_position(symbol)
                        actual_size = abs(pos.get('contracts', size))
                        self.logger.info(
                            f"Position for {symbol} verified via portfolio state: "
                            f"{actual_size:.6f} contracts"
                        )
                        break
                
                # Check exchange directly as fallback
                if not position_verified:
                    try:
                        positions = self.exchange.fetch_positions(symbol=symbol)
                        for pos in positions:
                            # Normalize symbol format: CCXT returns "SOL/USDT:USDT" for perpetual futures
                            # Convert to "SOLUSDT" for comparison (same logic as portfolio.py)
                            pos_symbol_raw = pos.get('symbol', '')
                            
                            # First, remove :USDT suffix if present (for perpetual futures)
                            if ':USDT' in pos_symbol_raw:
                                pos_symbol = pos_symbol_raw.replace(':USDT', '')  # "SOL/USDT:USDT" → "SOL/USDT"
                            else:
                                pos_symbol = pos_symbol_raw
                            
                            # Then, replace /USDT with USDT (handles both spot and futures after removing :USDT)
                            if '/USDT' in pos_symbol:
                                pos_symbol = pos_symbol.replace('/USDT', 'USDT')  # "SOL/USDT" → "SOLUSDT"
                            # else: already in BASEUSDT format, use as-is
                            
                            self.logger.debug(
                                f"Position verification: comparing {pos_symbol_raw} (normalized: {pos_symbol}) "
                                f"with target {symbol}"
                            )
                            
                            contracts = float(pos.get('contracts', 0))
                            if pos_symbol == symbol and abs(contracts) >= 0.001:
                                position_verified = True
                                actual_size = abs(contracts)
                                self.logger.info(
                                    f"Position for {symbol} verified via exchange: "
                                    f"{actual_size:.6f} contracts"
                                )
                                break
                    except Exception as e:
                        self.logger.warning(
                            f"Error fetching positions during verification attempt {attempt + 1}: {e}"
                        )
                
                if position_verified:
                    break
            
            if not position_verified:
                # Calculate total wait time (sum of all delays)
                total_wait_time = sum(min(1.0 * (1.5 ** i), 3.0) for i in range(max_retries - 1))
                self.logger.warning(
                    f"Position for {symbol} not verified after {max_retries} attempts (~{total_wait_time:.1f}s total). "
                    f"Entry order {order_id} was placed, but position not yet visible. "
                    "Stop-loss will be placed when position appears on next cycle."
                )
                # Store the entry info so we can place SL/TP when position appears
                if portfolio_state:
                    portfolio_state.set_position_metadata(
                        symbol,
                        entry_time=datetime.now(timezone.utc),
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                    )
                return {
                    'status': 'partial',
                    'order': order,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': entry_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'signal': signal,
                    'message': 'Entry order placed, but position not yet verified. SL/TP will be placed on next cycle.'
                }
            
            self.logger.info(f"Position for {symbol} verified ({actual_size:.6f} contracts). Proceeding to place SL/TP orders...")
            
            # Place stop-loss order if configured and stop price provided
            stop_order_id = None
            tp_order_id = None
            
            # Determine position side from signal
            position_side = signal  # 'long' or 'short'
            
            # actual_size is already set during position verification above
            
            # Place stop-loss order
            if stop_loss_price:
                stop_order_id = self._place_stop_loss_order(
                    symbol,
                    position_side,
                    actual_size,  # Use actual position size
                    stop_loss_price,
                    portfolio_state,
                    entry_price,  # Pass entry price for validation
                )
            else:
                self.logger.warning(f"No stop-loss price provided for {symbol}, skipping SL order")
            
            # Place take-profit order if configured
            if take_profit_price and self.strategy_config:
                tp_order_id = self._place_take_profit_order(
                    symbol,
                    position_side,
                    actual_size,  # Use actual position size
                    take_profit_price,
                    portfolio_state,
                )
            else:
                if not take_profit_price:
                    take_profit_rr_val = None
                    if self.strategy_config:
                        take_profit_rr_val = self.strategy_config.take_profit_rr
                    self.logger.info(
                        f"Take-profit not configured for {symbol} "
                        f"(take_profit_rr={take_profit_rr_val}). "
                        f"To enable TP orders, set strategy.trend.take_profit_rr in config.yaml (e.g., 2.0 for 2x risk-reward). "
                        f"Skipping TP order."
                    )
                elif not self.strategy_config:
                    self.logger.warning(f"No strategy_config available for {symbol}, skipping TP order")
            
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
            if abs(contracts) >= 0.001:  # Include minimum order size (0.001 for BTC)
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
            
            # CRITICAL: If a position already exists for this symbol, skip placing a new order
            # This prevents stacking positions on each iteration, regardless of size differences
            # We only act if:
            # 1. Signal changed direction (handled by checking sign mismatch below)
            # 2. Signal changed to flat (symbol not in target_positions - handled above)
            # 3. Position needs to be closed (target_size == 0)
            if abs(current_size) >= 0.001:  # Position already exists (>= to include minimum order size)
                # Check if direction matches (same sign = same direction)
                same_direction = (
                    (target_size > 0 and current_size > 0) or
                    (target_size < 0 and current_size < 0)
                )
                
                if same_direction:
                    # Position exists with same direction - skip entirely to prevent stacking
                    # Size differences (due to profit, equity changes, etc.) should NOT trigger new orders
                    self.logger.info(
                        f"Skipping {symbol}: position already exists (same direction). "
                        f"Current={current_size:.4f}, Target={target_size:.4f}. "
                        f"No new order will be placed to prevent stacking."
                    )
                    continue
                else:
                    # Direction changed - need to close existing position first
                    # This will be handled by closing the position (it's not in target), then opening new one
                    self.logger.warning(
                        f"Direction mismatch for {symbol}: existing={current_size:.4f}, "
                        f"target={target_size:.4f}. Position should be closed first."
                    )
                    # Close existing position first - it will be handled in the close loop above
                    continue
            
            # Only execute if no position exists (new position to open)
            if abs(target_size) < 0.001:
                # Target size is zero, skip
                continue
            
            # Open new position
            self.logger.info(
                f"Opening new position for {symbol}: "
                f"target={target_size:.4f} ({target.get('signal', 'long').upper()})"
            )
            
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

