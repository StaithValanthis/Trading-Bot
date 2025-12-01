"""Order execution logic."""

import time
from typing import Dict, Optional, List
from datetime import datetime, timezone
import ccxt

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
    
    def _cancel_stop_orders(
        self, 
        symbol: str, 
        portfolio_state: Optional[PortfolioState] = None,
        force_refresh: bool = False
    ) -> Dict[str, bool]:
        """
        Cancel all stop-loss and take-profit orders for a symbol.
        
        Enhanced version that:
        - Checks exchange for SL/TP orders not in metadata
        - Handles OrderNotFound gracefully
        - Returns cancellation status
        
        Args:
            symbol: Trading pair symbol
            portfolio_state: Optional portfolio state to get order IDs from
            force_refresh: If True, refresh portfolio state before cancellation
            
        Returns:
            Dictionary with cancellation results:
            {
                'stop_cancelled': bool,
                'tp_cancelled': bool,
                'stop_order_id': str | None,
                'tp_order_id': str | None,
                'errors': List[str]
            }
        """
        results = {
            'stop_cancelled': False,
            'tp_cancelled': False,
            'stop_order_id': None,
            'tp_order_id': None,
            'errors': []
        }
        
        # Refresh portfolio state if requested
        if force_refresh and portfolio_state:
            portfolio_state.update()
        
        # Get order IDs from portfolio state
        stop_order_id = None
        tp_order_id = None
        
        if portfolio_state:
            pos = portfolio_state.positions.get(symbol)
            if pos:
                stop_order_id = pos.get('stop_order_id')
                tp_order_id = pos.get('tp_order_id')
        
        results['stop_order_id'] = stop_order_id
        results['tp_order_id'] = tp_order_id
        
        # CRITICAL: Check exchange directly for ALL SL/TP orders and cancel them all
        # This ensures we cancel orders even if they're not in portfolio state metadata
        # This is especially important when positions are closed and SL orders are orphaned
        orders_to_cancel = []  # List of (order_id, order_type) tuples
        found_order_ids = set()
        
        # Add orders from portfolio state to cancellation list
        if stop_order_id:
            orders_to_cancel.append((stop_order_id, 'stop_loss'))
            found_order_ids.add(stop_order_id)
        if tp_order_id:
            orders_to_cancel.append((tp_order_id, 'take_profit'))
            found_order_ids.add(tp_order_id)
        
        # Fetch ALL open orders from exchange and find ALL SL/TP orders
        # CRITICAL: For Bybit, we need to explicitly fetch conditional orders (stop-loss/take-profit)
        # These may not be returned by regular fetch_open_orders without proper parameters
        open_orders = []
        try:
            # Try fetching conditional orders specifically first (Bybit v5 specific)
            # This is more reliable for finding SL/TP orders
            try:
                conditional_orders = self.exchange.fetch_open_orders(
                    symbol, 
                    params={'category': 'linear', 'orderFilter': 'Stop'}
                )
                open_orders.extend(conditional_orders)
                self.logger.debug(f"Found {len(conditional_orders)} conditional orders for {symbol}")
            except Exception as e1:
                self.logger.debug(f"Could not fetch conditional orders with Stop filter for {symbol}: {e1}")
            
            # Also try fetching all open orders (including regular and conditional)
            try:
                all_orders = self.exchange.fetch_open_orders(symbol, params={'category': 'linear'})
                # Add orders that aren't already in our list
                existing_ids = {o.get('id') for o in open_orders if o.get('id')}
                for order in all_orders:
                    if order.get('id') not in existing_ids:
                        open_orders.append(order)
                self.logger.debug(f"Found {len(all_orders)} total orders (added {len([o for o in all_orders if o.get('id') not in existing_ids])} new ones)")
            except Exception as e2:
                # Fallback: try without params
                try:
                    all_orders = self.exchange.fetch_open_orders(symbol)
                    existing_ids = {o.get('id') for o in open_orders if o.get('id')}
                    for order in all_orders:
                        if order.get('id') not in existing_ids:
                            open_orders.append(order)
                    self.logger.debug(f"Found {len(all_orders)} orders without params (added {len([o for o in all_orders if o.get('id') not in existing_ids])} new ones)")
                except Exception as e3:
                    self.logger.warning(f"Could not fetch open orders for {symbol}: {e2}, {e3}")
            
            self.logger.debug(f"Found {len(open_orders)} total open orders for {symbol} on exchange (including conditionals)")
            
            # Log all order types found for debugging
            if open_orders:
                order_types_found = set()
                for order in open_orders:
                    order_type = order.get('type', 'unknown')
                    order_info = order.get('info', {})
                    stop_order_type = order_info.get('stopOrderType', 'N/A')
                    order_types_found.add(f"{order_type}/{stop_order_type}")
                self.logger.debug(f"Order types found for {symbol}: {order_types_found}")
            
            
            for order in open_orders:
                order_id = order.get('id')
                if not order_id:
                    continue
                    
                order_type = order.get('type', '').lower()
                order_info = order.get('info', {})
                order_type_info = order_info.get('stopOrderType', '').lower() if isinstance(order_info.get('stopOrderType'), str) else ''
                
                # Check if it's a stop-loss order (more robust detection)
                is_stop_order = (
                    ('stop' in order_type and 'take_profit' not in order_type and 'takeprofit' not in order_type) or
                    ('stop' in order_type_info and 'takeprofit' not in order_type_info) or
                    order_info.get('stopOrderType') in ['StopMarket', 'StopLimit', 'Stop'] or
                    order_info.get('orderType') == 'Stop' or
                    order_info.get('triggerPrice') is not None and 'TakeProfit' not in str(order_info.get('stopOrderType', ''))
                )
                
                # Check if it's a take-profit order
                is_tp_order = (
                    'take_profit' in order_type or
                    'takeprofit' in order_type_info or
                    order_info.get('stopOrderType') in ['TakeProfit', 'TakeProfitMarket', 'TakeProfitLimit'] or
                    order_info.get('orderType') == 'TakeProfit'
                )
                
                # Add to cancellation list if not already there
                if is_stop_order and order_id not in found_order_ids:
                    orders_to_cancel.append((order_id, 'stop_loss'))
                    found_order_ids.add(order_id)
                    self.logger.info(
                        f"Found SL order {order_id} on exchange for {symbol} "
                        "(not in local metadata). Will cancel."
                    )
                elif is_tp_order and order_id not in found_order_ids:
                    orders_to_cancel.append((order_id, 'take_profit'))
                    found_order_ids.add(order_id)
                    self.logger.info(
                        f"Found TP order {order_id} on exchange for {symbol} "
                        "(not in local metadata). Will cancel."
                    )
        except Exception as e:
            self.logger.warning(f"Could not fetch open orders for {symbol} to verify SL/TP: {e}")
            results['errors'].append(f"Order fetch error: {e}")
        
        # Log summary of orders to cancel
        if orders_to_cancel:
            order_list = ', '.join([f'{oid} ({otype})' for oid, otype in orders_to_cancel])
            self.logger.info(
                f"Found {len(orders_to_cancel)} SL/TP order(s) to cancel for {symbol}: {order_list}"
            )
        else:
            # Log why no orders were found
            orders_from_state = []
            if stop_order_id:
                orders_from_state.append(f"SL={stop_order_id}")
            if tp_order_id:
                orders_from_state.append(f"TP={tp_order_id}")
            
            exchange_order_count = len([o for o in open_orders if o.get('id')])
            self.logger.debug(
                f"No SL/TP orders found to cancel for {symbol}. "
                f"Orders in state: {', '.join(orders_from_state) if orders_from_state else 'none'}. "
                f"Orders found on exchange: {exchange_order_count}"
            )
        
        # Cancel ALL stop-loss and take-profit orders found
        cancelled_stop_count = 0
        cancelled_tp_count = 0
        
        for order_id, order_type_str in orders_to_cancel:
            try:
                # For conditional orders (SL/TP), pass params to help Bybit identify them correctly
                # Bybit v5 API requires category='linear' and orderFilter='Stop' for conditional orders
                cancel_params = {
                    'category': 'linear',
                    'orderFilter': 'Stop'  # 'Stop' covers both stop-loss and take-profit conditional orders
                }
                
                self.logger.debug(
                    f"Attempting to cancel {order_type_str} order {order_id} for {symbol} "
                    f"(using params: {cancel_params})"
                )
                
                self.exchange.cancel_order(order_id, symbol, params=cancel_params)
                
                if order_type_str == 'stop_loss':
                    cancelled_stop_count += 1
                    self.logger.info(f"✅ Cancelled stop-loss order {order_id} for {symbol}")
                else:
                    cancelled_tp_count += 1
                    self.logger.info(f"✅ Cancelled take-profit order {order_id} for {symbol}")
                
                # Update portfolio state to clear the order ID
                if portfolio_state:
                    if order_type_str == 'stop_loss':
                        portfolio_state.set_position_metadata(symbol, stop_order_id=None)
                    else:
                        portfolio_state.set_position_metadata(symbol, tp_order_id=None)
                        
            except ccxt.OrderNotFound:
                self.logger.debug(
                    f"{order_type_str.capitalize()} order {order_id} already cancelled/not found for {symbol}"
                )
                # Consider it cancelled
                if order_type_str == 'stop_loss':
                    cancelled_stop_count += 1
                else:
                    cancelled_tp_count += 1
                    
                # Still update portfolio state to clear the order ID
                if portfolio_state:
                    if order_type_str == 'stop_loss':
                        portfolio_state.set_position_metadata(symbol, stop_order_id=None)
                    else:
                        portfolio_state.set_position_metadata(symbol, tp_order_id=None)
                        
            except Exception as e:
                error_msg = f"Error cancelling {order_type_str} order {order_id} for {symbol}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                # Continue anyway - don't block position close
        
        # Update results
        results['stop_cancelled'] = cancelled_stop_count > 0
        results['tp_cancelled'] = cancelled_tp_count > 0
        
        # Log summary
        if cancelled_stop_count > 0 or cancelled_tp_count > 0:
            self.logger.info(
                f"Cancelled {cancelled_stop_count} stop-loss and {cancelled_tp_count} take-profit "
                f"order(s) for {symbol}"
            )
        elif not orders_to_cancel:
            self.logger.debug(f"No SL/TP orders found to cancel for {symbol}")
        
        # Warn if any errors occurred
        if results['errors']:
            self.logger.warning(
                f"Some errors during SL/TP cancellation for {symbol}: {results['errors']}"
            )
        
        return results
    
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
        # CRITICAL: Check exchange DIRECTLY for existing positions BEFORE placing any order
        # This is the most reliable way to prevent stacking - query exchange in real-time
        try:
            # Fetch positions directly from exchange (bypass portfolio state cache)
            exchange_positions = self.exchange.fetch_positions(symbol=symbol)
            for pos in exchange_positions:
                pos_symbol = pos.get("symbol", "")
                # Normalize symbol for comparison
                pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                
                if pos_symbol_normalized == symbol_normalized:
                    contracts = float(pos.get("contracts", 0))
                    if abs(contracts) >= 0.001:  # Position exists (including stacked positions)
                        side = pos.get("side", "long")
                        # Convert to signed size
                        existing_signed_size = abs(contracts) if side == "long" else -abs(contracts)
                        
                        # Check if direction matches
                        same_direction = (
                            (target_size > 0 and existing_signed_size > 0) or
                            (target_size < 0 and existing_signed_size < 0)
                        )
                        
                        if same_direction:
                            # CRITICAL: Position exists with same direction - ALWAYS ABORT to prevent stacking
                            # Don't compare sizes - if ANY position exists in same direction, skip the order
                            # This prevents stacking even if position size doesn't match target (e.g., due to minimum order bumps)
                            self.logger.warning(
                                f"🚫 BLOCKING ORDER for {symbol}: Position already exists on exchange "
                                f"({side} {abs(contracts):.6f} contracts, signed={existing_signed_size:.6f}). "
                                f"Target={target_size:.6f}. NO ORDER will be placed to prevent stacking. "
                                f"(Note: Size mismatch may indicate prior stacking - existing position will remain.)"
                            )
                            return {
                                'status': 'skipped',
                                'symbol': symbol,
                                'reason': 'position_exists_on_exchange_same_direction',
                                'current_size': existing_signed_size,
                                'target_size': target_size,
                                'message': f'Position exists in same direction ({side} {abs(contracts):.6f}), blocking order to prevent stacking'
                            }
                        else:
                            # Direction mismatch - should have been closed first
                            self.logger.warning(
                                f"🚫 BLOCKING ORDER for {symbol}: Position exists with opposite direction "
                                f"({side} {abs(contracts):.6f} contracts, signed={existing_signed_size:.6f}). "
                                f"Target={target_size:.6f}. Position must be closed first. NO ORDER placed."
                            )
                            return {
                                'status': 'skipped',
                                'symbol': symbol,
                                'reason': 'position_exists_on_exchange_opposite_direction',
                                'current_size': existing_signed_size,
                                'target_size': target_size
                            }
        except Exception as e:
            # If exchange check fails, fall back to portfolio state check
            self.logger.warning(f"Error checking exchange directly for {symbol} position: {e}. Using portfolio state check.")
            
            # Fallback: Check portfolio state (may be stale but better than nothing)
            if portfolio_state:
                try:
                    portfolio_state.update()
                    self.logger.debug(f"Portfolio state refreshed in execute_position_change for {symbol}")
                except Exception as e2:
                    self.logger.warning(f"Error refreshing portfolio state in execute_position_change for {symbol}: {e2}")
                
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
                        self.logger.warning(
                            f"⚠️ BLOCKING ORDER for {symbol}: Position exists in portfolio state "
                            f"({existing_side} {abs(existing_contracts):.6f} contracts, signed={existing_signed_size:.6f}). "
                            f"Target={target_size:.6f}. NO ORDER will be placed to prevent stacking."
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
                            f"⚠️ BLOCKING ORDER for {symbol}: Position exists with opposite direction "
                            f"({existing_side} {abs(existing_contracts):.6f} contracts, signed={existing_signed_size:.6f}). "
                            f"Target={target_size:.6f}. Position must be closed first. NO ORDER placed."
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
        
        # FINAL SAFETY CHECK: Double-check exchange directly right before placing order
        # This is the last line of defense against stacking
        try:
            exchange_positions = self.exchange.fetch_positions(symbol=symbol)
            for pos in exchange_positions:
                pos_symbol = pos.get("symbol", "")
                pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                
                if pos_symbol_normalized == symbol_normalized:
                    contracts = float(pos.get("contracts", 0))
                    if abs(contracts) >= 0.001:  # Position exists!
                        side_str = pos.get("side", "long")
                        existing_signed = abs(contracts) if side_str == "long" else -abs(contracts)
                        same_dir = ((target_size > 0 and existing_signed > 0) or (target_size < 0 and existing_signed < 0))
                        
                        if same_dir:
                            # CRITICAL: Position exists right before placing order - ABORT!
                            self.logger.error(
                                f"🚨 CRITICAL: Position for {symbol} detected on exchange RIGHT BEFORE placing order! "
                                f"Existing: {side_str} {abs(contracts):.6f} (signed={existing_signed:.6f}), "
                                f"Target: {target_size:.6f}. ORDER ABORTED to prevent stacking!"
                            )
                            return {
                                'status': 'skipped',
                                'symbol': symbol,
                                'reason': 'position_detected_right_before_order',
                                'current_size': existing_signed,
                                'target_size': target_size
                            }
        except Exception as e:
            self.logger.warning(f"Final safety check failed for {symbol} before placing order: {e}. Proceeding with caution.")
        
        try:
            # Place market order (can be changed to limit if needed)
            self.logger.info(
                f"🔵 Placing {side} order for {symbol}: {size:.6f} contracts @ {entry_price:.2f} "
                f"(target_size={target_size:.6f})"
            )
            order = self.exchange.create_order(
                symbol,
                side,
                size,
                order_type='market',
                price=None
            )
            
            order_id = order.get('id')
            self.recent_orders[symbol] = current_time
            
            # CRITICAL: After placing order, immediately verify position to detect stacking
            # Wait a moment for position to appear, then check
            time.sleep(0.5)  # Brief delay for position to propagate
            
            # Verify position was created/updated correctly
            try:
                verify_positions = self.exchange.fetch_positions(symbol=symbol)
                aggregated_verify_size = 0.0
                verify_side = None
                
                for verify_pos in verify_positions:
                    verify_symbol = verify_pos.get("symbol", "")
                    verify_symbol_normalized = verify_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                    symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                    
                    if verify_symbol_normalized == symbol_normalized:
                        verify_contracts = float(verify_pos.get("contracts", 0))
                        verify_side = verify_pos.get("side", "long")
                        # Aggregate all positions for the same symbol (in case of stacking)
                        aggregated_verify_size += abs(verify_contracts)
                
                # Check if this is a stacking situation (position larger than expected)
                # For minimum order sizes (e.g., 0.001 BTC), if we see 0.002+ BTC, it's likely stacking
                expected_max_size = size * 1.05  # Allow 5% tolerance for rounding/slippage
                if aggregated_verify_size > expected_max_size:
                    self.logger.error(
                        f"🚨 CRITICAL: Position stacking detected for {symbol} after order! "
                        f"Ordered: {size:.6f}, Total position on exchange: {verify_side} {aggregated_verify_size:.6f}. "
                        f"This indicates multiple orders were placed (stacking). "
                        f"Expected max: {expected_max_size:.6f}, Actual: {aggregated_verify_size:.6f}"
                    )
                    # Log warning but don't fail - the order already went through
                    # The stacking prevention checks should prevent future orders
            except Exception as e:
                self.logger.warning(f"Could not verify position after order for {symbol}: {e}")
            
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
                # Get source and metadata from target_positions if available (passed via reconcile_positions)
                # This is a workaround - ideally execute_position_change would accept these as parameters
                # For now, we'll set them after position is opened via a separate call
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
        # Fetch all positions (more reliable than filtering by symbol)
        # CCXT may not recognize internal symbol format (BTCUSDT vs BTC/USDT:USDT)
        all_positions = self.exchange.fetch_positions()
        
        # Filter and AGGREGATE positions that match symbol (handle various CCXT formats)
        # CRITICAL: Sum all positions for the same symbol in case there are multiple entries
        # This can happen if positions were stacked or opened separately
        aggregated_positions = {}
        for pos in all_positions:
            pos_symbol = pos.get("symbol", "")
            # Normalize both symbols for comparison (remove / and :USDT variations)
            pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
            symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
            
            if pos_symbol_normalized == symbol_normalized:
                contracts = float(pos.get("contracts", 0))
                side = pos.get("side", "long" if contracts > 0 else "short")
                
                # Aggregate positions for the same symbol
                # Store by normalized symbol and side
                key = f"{symbol_normalized}_{side}"
                if key not in aggregated_positions:
                    aggregated_positions[key] = {
                        "contracts": 0.0,
                        "side": side,
                        "positions": []  # Track individual positions for logging
                    }
                
                # Sum contracts (can be positive for long, negative for short)
                aggregated_positions[key]["contracts"] += contracts
                aggregated_positions[key]["positions"].append(pos)
                
                self.logger.debug(
                    f"Found position {pos_symbol} ({side} {abs(contracts):.6f}) for {symbol}. "
                    f"Total aggregated: {side} {abs(aggregated_positions[key]['contracts']):.6f}"
                )
        
        # Process aggregated positions (should only be one per symbol, but aggregate if multiple)
        if not aggregated_positions:
            self.logger.warning(
                f"⚠️ Could not find position for {symbol} to close. "
                f"Searched {len(all_positions)} position(s) from exchange. "
                "Position may already be closed or symbol format mismatch."
            )
            return {"status": "no_position", "symbol": symbol, "message": "Position not found"}
        
        # Get the aggregated position (there should only be one for a given symbol)
        agg_pos = list(aggregated_positions.values())[0]  # Take first (should be only one)
        side = agg_pos["side"]
        total_contracts = agg_pos["contracts"]
        
        # Log if multiple position entries were aggregated
        if len(agg_pos["positions"]) > 1:
            self.logger.warning(
                f"⚠️ Multiple position entries found for {symbol} ({side}): "
                f"{len(agg_pos['positions'])} entries aggregated to total {abs(total_contracts):.6f} contracts. "
                f"This may indicate position stacking occurred. Will close the full aggregated position."
            )
        
        # Use the most recent position entry for metadata (entry price, etc.)
        latest_pos = agg_pos["positions"][-1]  # Use last one (usually most recent)
        
        # Create aggregated position object
        pos = latest_pos.copy()  # Copy to preserve original
        pos["contracts"] = total_contracts  # Replace with aggregated total
        pos_symbol = pos.get("symbol", "")
        
        # Symbol already matched in filter above, proceed with closing
        contracts = float(pos.get("contracts", 0))
        
        if abs(contracts) < 0.001:  # Already closed
            self.logger.info(f"Position {symbol} already closed (contracts={contracts:.6f})")
            return {"status": "no_position", "symbol": symbol, "message": "Position already closed"}
        
        # CRITICAL: Use the EXACT position size from exchange (not portfolio state)
        # This ensures we close the full position, not a partial or stale size.
        # For Bybit linear perps, CCXT typically returns contracts as a positive number
        # and uses the `side` field ('long'/'short') to indicate direction.
        # Therefore we MUST use `position_side` to choose the closing side, not the sign of `contracts`.
        size_from_exchange = abs(total_contracts)
        
        # Determine position side from exchange data
        position_side = pos.get("side", "long").lower()  # 'long' or 'short'
        
        # Determine side to close based on position side (NOT on contracts sign!)
        # - To close a LONG we SELL
        # - To close a SHORT we BUY
        close_side = "sell" if position_side == "long" else "buy"
        size = size_from_exchange  # Use aggregated exchange position size directly
        
        self.logger.info(
            f"Closing {symbol}: Found position on exchange - {position_side.upper()} "
            f"{size_from_exchange:.6f} contracts (aggregated from {len(agg_pos['positions'])} entries). "
            f"Will place {close_side.upper()} order for {size:.6f} contracts to fully close position."
        )
        
        entry_price = float(pos.get("entryPrice", 0.0))
        mark_price = float(pos.get("markPrice", entry_price))
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
            # For closing positions, we don't use reduceOnly=True
            # reduceOnly is for conditional orders (stop-loss/take-profit), not regular closing orders
            # By placing an order in the opposite direction of the position, Bybit automatically closes it
            # Using reduceOnly=True can cause error 110017: "reduce-only order has same side with current position"
            self.logger.info(
                f"Closing {symbol} position: placing {close_side} order for {size:.6f} contracts "
                f"to close {position_side.upper()} position"
            )
            order = self.exchange.create_order(
                symbol,
                close_side,
                size,
                order_type="market"
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
                        side=close_side,
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

        except ccxt.InsufficientFunds as e:
            error_msg = str(e)
            # When closing positions, insufficient funds might indicate:
            # 1. Account truly has no available balance (unlikely for reduce-only orders)
            # 2. Exchange requires some available balance even for reduce-only orders
            # 3. Position size is too large relative to available margin
            self.logger.error(
                f"❌ Insufficient funds when closing position for {symbol}: {error_msg}\n"
                f"   Position: {position_side.upper()} {size} contracts @ entry {entry_price:.2f}\n"
                f"   This may indicate the account needs more available balance to close the position.\n"
                f"   Try closing positions individually or reduce position sizes."
            )
            return {
                "status": "error",
                "symbol": symbol,
                "error": f"Insufficient funds: {error_msg}",
                "error_type": "insufficient_funds"
            }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"❌ Error closing position for {symbol}: {error_msg}",
                exc_info=True
            )
            return {"status": "error", "symbol": symbol, "error": error_msg}
    
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
        
        # CRITICAL: Refresh portfolio state FIRST to get latest positions from exchange
        # This prevents stacking by ensuring we have the most up-to-date position data
        try:
            portfolio_state.update()
            self.logger.debug("Portfolio state refreshed before reconciliation")
        except Exception as e:
            self.logger.warning(f"Error refreshing portfolio state before reconciliation: {e}")
            # Continue anyway - will use stale data, but better than failing completely
        
        # Get current positions from portfolio state (now freshly updated)
        # ALSO fetch directly from exchange to ensure we have the latest data
        current_positions = {}
        try:
            # Fetch positions directly from exchange for more reliable data
            exchange_positions = self.exchange.fetch_positions()
            for pos in exchange_positions:
                pos_symbol = pos.get("symbol", "")
                # Normalize symbol
                pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                
                contracts = float(pos.get("contracts", 0)) if pos.get("contracts") is not None else 0.0
                if abs(contracts) >= 0.001:  # Include minimum order size (0.001 for BTC)
                    side = pos.get("side", "long" if contracts > 0 else "short")
                    # Store signed contracts (positive for long, negative for short)
                    signed_contracts = abs(contracts) if side == 'long' else -abs(contracts)
                    current_positions[pos_symbol_normalized] = signed_contracts
                    
                    self.logger.debug(
                        f"Found position on exchange for {pos_symbol_normalized}: {side} {abs(contracts):.6f} contracts "
                        f"(signed={signed_contracts:.6f})"
                    )
        except Exception as e:
            self.logger.warning(f"Error fetching positions directly from exchange in reconcile_positions: {e}. Using portfolio state.")
            # Fallback to portfolio state if exchange fetch fails
            for symbol, pos in portfolio_state.positions.items():
                contracts = pos.get('contracts', 0)
                side = pos.get('side', 'long')
                
                # Convert contracts to float if needed
                try:
                    contracts = float(contracts) if contracts is not None else 0.0
                except (TypeError, ValueError):
                    contracts = 0.0
                
                if abs(contracts) >= 0.001:  # Include minimum order size (0.001 for BTC)
                    # Store signed contracts (positive for long, negative for short)
                    signed_contracts = abs(contracts) if side == 'long' else -abs(contracts)
                    current_positions[symbol] = signed_contracts
        
        # Log reconciliation details
        for symbol in current_positions:
            if symbol in target_positions:
                target = target_positions[symbol]
                target_size = target.get('size', 0)
                current_size = current_positions[symbol]
                self.logger.info(
                    f"Reconciling {symbol}: current={current_size:.6f}, "
                    f"target={target_size:.6f}"
                )
        
        # Log positions that will be closed (not in target)
        positions_to_close = [s for s in current_positions if s not in target_positions]
        if positions_to_close:
            self.logger.info(
                f"Closing {len(positions_to_close)} position(s) not in target portfolio: {positions_to_close}"
            )
        
        # Close positions not in target
        for symbol in positions_to_close:
            # CRITICAL: Double-check exchange directly to ensure position actually exists before closing
            # This prevents closing positions that were already closed in a previous iteration
            try:
                exchange_positions = self.exchange.fetch_positions()
                position_found = False
                actual_size = 0.0
                actual_side = None
                
                for pos in exchange_positions:
                    pos_symbol = pos.get("symbol", "")
                    pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                    symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                    
                    if pos_symbol_normalized == symbol_normalized:
                        contracts = float(pos.get("contracts", 0)) if pos.get("contracts") is not None else 0.0
                        if abs(contracts) >= 0.001:  # Position actually exists
                            position_found = True
                            actual_size = abs(contracts)
                            actual_side = pos.get("side", "long" if contracts > 0 else "short")
                            break
                
                if not position_found:
                    # Position already closed (or was a false positive from stale portfolio state)
                    self.logger.info(
                        f"Skipping close for {symbol}: Position not found on exchange "
                        "(may have been closed already or never existed)"
                    )
                    results.append({
                        'status': 'skipped',
                        'symbol': symbol,
                        'reason': 'position_already_closed_on_exchange',
                        'message': 'Position not found on exchange, skipping close'
                    })
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Error verifying position existence for {symbol} before close: {e}. Proceeding with close.")
            
            # Position confirmed to exist on exchange, proceed with closing
            pos = portfolio_state.positions.get(symbol, {})
            side = pos.get('side', 'long')
            contracts = abs(pos.get('contracts', 0))
            self.logger.info(
                f"Closing {symbol}: {side.upper()} {contracts:.4f} contracts "
                f"(not in target portfolio)"
            )
            
            # CRITICAL: Cancel ALL stop/TP orders before closing (enhanced version)
            # This prevents orphaned SL/TP orders after position closure
            self.logger.info(f"Cancelling all SL/TP orders for {symbol} before closing position...")
            cancel_results = self._cancel_stop_orders(symbol, portfolio_state, force_refresh=True)
            
            if cancel_results.get('stop_cancelled') or cancel_results.get('tp_cancelled'):
                self.logger.info(
                    f"Successfully cancelled orders for {symbol}: "
                    f"SL={cancel_results.get('stop_cancelled')}, TP={cancel_results.get('tp_cancelled')}"
                )
            
            if cancel_results.get('errors'):
                self.logger.warning(
                    f"Errors cancelling SL/TP for {symbol}: {cancel_results['errors']}. "
                    "Proceeding with position close."
                )
            
            if not cancel_results.get('stop_cancelled') and not cancel_results.get('tp_cancelled'):
                self.logger.debug(
                    f"No SL/TP orders found to cancel for {symbol} "
                    "(may have been cancelled already or none exist)"
                )
            
            result = self.close_position(symbol)
            results.append(result)
            
            # CRITICAL: After closing, verify position was actually closed and SL was canceled
            # Double-check for any orphaned SL/TP orders that may remain
            if result.get('status') == 'closed':
                # Wait a moment for exchange to process the close
                time.sleep(0.5)
                
                # Verify position is closed and check for orphaned SL orders
                try:
                    exchange_positions = self.exchange.fetch_positions()
                    position_still_exists = False
                    for pos in exchange_positions:
                        pos_symbol = pos.get("symbol", "")
                        pos_symbol_normalized = pos_symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                        symbol_normalized = symbol.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT").replace(":USDT", "USDT")
                        if pos_symbol_normalized == symbol_normalized:
                            contracts = float(pos.get("contracts", 0)) if pos.get("contracts") is not None else 0.0
                            if abs(contracts) >= 0.001:
                                position_still_exists = True
                                break
                    
                    if position_still_exists:
                        self.logger.warning(
                            f"Position {symbol} still exists after close attempt. "
                            "This may indicate a partial fill or exchange delay."
                        )
                    else:
                        # Position is closed, verify no orphaned SL orders remain
                        final_cancel_results = self._cancel_stop_orders(symbol, portfolio_state, force_refresh=True)
                        if final_cancel_results.get('stop_cancelled') or final_cancel_results.get('tp_cancelled'):
                            self.logger.info(
                                f"Cleaned up orphaned orders for {symbol}: "
                                f"SL={final_cancel_results.get('stop_cancelled')}, "
                                f"TP={final_cancel_results.get('tp_cancelled')}"
                            )
                except Exception as e:
                    self.logger.warning(f"Error verifying position close for {symbol}: {e}")
            
            # CRITICAL: After closing, wait a moment and verify position was actually closed
            # Also check for and cancel any orphaned SL/TP orders that may remain
            if result.get('status') == 'closed':
                time.sleep(0.5)  # Brief delay for close order to settle
                
                # FINAL SAFETY CHECK: Cancel any remaining SL/TP orders after position closure
                # This handles cases where orders weren't cancelled before closing or remain orphaned
                self.logger.debug(f"Performing final cleanup: checking for orphaned SL/TP orders for {symbol}...")
                final_cancel_results = self._cancel_stop_orders(symbol, portfolio_state, force_refresh=True)
                if final_cancel_results.get('stop_cancelled') or final_cancel_results.get('tp_cancelled'):
                    self.logger.warning(
                        f"⚠️ Found and cancelled orphaned SL/TP orders for {symbol} after position closure: "
                        f"SL={final_cancel_results.get('stop_cancelled')}, TP={final_cancel_results.get('tp_cancelled')}"
                    )
                
                try:
                    # Refresh portfolio state to get latest positions
                    portfolio_state.update()
                    # Check if position still exists
                    still_open = portfolio_state.get_position(symbol)
                    if still_open:
                        contracts_remaining = abs(still_open.get('contracts', 0))
                        if contracts_remaining >= 0.001:
                            self.logger.warning(
                                f"⚠️ Position {symbol} still appears open after close order "
                                f"({still_open.get('side', 'unknown')} {contracts_remaining:.6f} contracts). "
                                f"This may indicate the close order hasn't settled yet, or there was a stacking issue."
                            )
                        else:
                            self.logger.debug(f"Position {symbol} successfully closed (contracts={contracts_remaining:.6f} < 0.001)")
                    else:
                        self.logger.debug(f"Position {symbol} successfully closed (not found in portfolio state)")
                except Exception as e:
                    self.logger.warning(f"Error verifying position closure for {symbol}: {e}")
        
        # Open or adjust positions
        for symbol, target in target_positions.items():
            target_size = target.get('size', 0)
            current_size = current_positions.get(symbol, 0)
            
            # Log for debugging
            self.logger.debug(
                f"Checking {symbol}: current_size={current_size:.6f}, target_size={target_size:.6f}, "
                f"abs(current_size)={abs(current_size):.6f}"
            )
            
            # CRITICAL: If a position already exists for this symbol, skip placing a new order
            # This prevents stacking positions on each iteration, regardless of size differences
            # We only act if:
            # 1. Signal changed direction (handled by checking sign mismatch below)
            # 2. Signal changed to flat (symbol not in target_positions - handled above)
            # 3. Position needs to be closed (target_size == 0)
            # CRITICAL: Use tolerance for floating point comparison, especially important for minimum order sizes
            size_tolerance = 0.0001  # Small tolerance for floating point comparison
            if abs(current_size) >= 0.001:  # Position already exists (>= to include minimum order size)
                # Check if direction matches (same sign = same direction)
                same_direction = (
                    (target_size > 0 and current_size > 0) or
                    (target_size < 0 and current_size < 0)
                )
                
                if same_direction:
                    # Calculate size difference percentage to determine if adjustment is needed
                    size_diff_abs = abs(target_size - current_size)
                    size_diff_pct = (size_diff_abs / abs(current_size) * 100) if current_size != 0 else float('inf')
                    
                    # If size difference is small (<10%), it's likely due to min order size bumps or rounding
                    # Allow these small mismatches without blocking (softened stacking prevention)
                    size_tolerance_pct = 10.0  # 10% tolerance for small size differences
                    
                    if size_diff_pct < size_tolerance_pct:
                        # Small size difference - skip but don't warn (likely expected due to min order size)
                        self.logger.info(
                            f"Position {symbol} exists with similar size. "
                            f"Current={current_size:.6f}, Target={target_size:.6f} "
                            f"(diff={size_diff_pct:.1f}%). Skipping order (size within tolerance)."
                        )
                        results.append({
                            'status': 'skipped',
                            'symbol': symbol,
                            'reason': 'position_already_exists_same_direction_size_match',
                            'current_size': current_size,
                            'target_size': target_size,
                            'size_diff_pct': size_diff_pct,
                            'message': f'Position exists with similar size (diff={size_diff_pct:.1f}%), skipping'
                        })
                    else:
                        # Large size difference - block to prevent stacking
                        # CRITICAL: Position exists with same direction and SIGNIFICANT size difference
                        # This prevents stacking when:
                        # - Positions were previously stacked (e.g., 0.002 BTC from two 0.001 BTC orders)
                        # - Size differences due to profit/equity changes that would double position
                        self.logger.warning(
                            f"⚠️ SKIPPING {symbol}: Position already exists (same direction) with large size difference. "
                            f"Current={current_size:.6f}, Target={target_size:.6f} "
                            f"(diff={size_diff_pct:.1f}%). NO NEW ORDER will be placed to prevent stacking. "
                            f"(Size mismatch may indicate prior stacking)"
                        )
                        # Return a skipped result to track this
                        results.append({
                            'status': 'skipped',
                            'symbol': symbol,
                            'reason': 'position_already_exists_same_direction_large_diff',
                            'current_size': current_size,
                            'target_size': target_size,
                            'size_diff_pct': size_diff_pct,
                            'message': f'Position exists in same direction with large size difference (diff={size_diff_pct:.1f}%), blocking order to prevent stacking'
                        })
                    continue
                else:
                    # Direction changed - need to close existing position first
                    # This will be handled by closing the position (it's not in target), then opening new one
                    self.logger.warning(
                        f"⚠️ Direction mismatch for {symbol}: existing={current_size:.6f}, "
                        f"target={target_size:.6f}. Position should be closed first. "
                        f"Skipping new order to prevent stacking."
                    )
                    # Return a skipped result
                    results.append({
                        'status': 'skipped',
                        'symbol': symbol,
                        'reason': 'direction_mismatch_close_first',
                        'current_size': current_size,
                        'target_size': target_size
                    })
                    continue
            
            # Only execute if no position exists (new position to open)
            if abs(target_size) < 0.001:
                # Target size is zero, skip
                continue
            
            # CRITICAL: Before opening a new position, cancel any existing SL/TP orders
            # This prevents orphaned stop-loss orders from previous positions
            # This is especially important when:
            # - A position was closed and then reopened in the same direction
            # - A position was partially closed and then reopened
            # - Old SL orders were left behind from previous iterations
            self.logger.debug(f"Checking for existing SL/TP orders for {symbol} before opening new position...")
            cancel_results = self._cancel_stop_orders(symbol, portfolio_state, force_refresh=True)
            if cancel_results.get('stop_cancelled') or cancel_results.get('tp_cancelled'):
                self.logger.info(
                    f"Cancelled existing orders for {symbol} before opening new position: "
                    f"SL={cancel_results.get('stop_cancelled')}, TP={cancel_results.get('tp_cancelled')}"
                )
            if cancel_results.get('errors'):
                self.logger.warning(
                    f"Errors cancelling old orders for {symbol}: {cancel_results['errors']}. "
                    "Proceeding with new position, but old orders may need manual cleanup."
                )
            
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
            
            # Store source and metadata if position was successfully opened
            if result.get('status') == 'filled' and portfolio_state:
                source = target.get('source', 'main_strategy')
                metadata = target.get('metadata', {})
                portfolio_state.set_position_metadata(
                    symbol,
                    source=source,
                    metadata=metadata
                )
            
            results.append(result)
        
        return results
    
    def ensure_protective_stops_for_all_positions(
        self,
        portfolio_state: PortfolioState,
        strategy_config: Optional[TrendStrategyConfig] = None,
    ) -> Dict[str, Dict]:
        """
        Ensure every open position has a valid server-side stop-loss order.
        
        This is a safety mechanism that:
        - Checks all open positions from exchange
        - Verifies SL orders exist on exchange (not just in metadata)
        - Recalculates SL prices if metadata is missing
        - Creates missing SL orders
        
        Args:
            portfolio_state: Current portfolio state
            strategy_config: Optional strategy config for ATR-based SL calculation
            
        Returns:
            Dictionary mapping symbol to result dict:
            {
                'status': 'ok' | 'created' | 'failed',
                'stop_order_id': str | None,
                'stop_loss_price': float | None,
                'message': str
            }
        """
        results = {}
        
        if not self.risk_config or not self.risk_config.use_server_side_stops:
            return results
        
        # Refresh portfolio state to get latest positions
        portfolio_state.update()
        
        for symbol, pos in portfolio_state.positions.items():
            contracts = abs(pos.get('contracts', 0))
            if contracts < 0.001:  # Skip zero/dust positions
                continue
            
            side = pos.get('side', 'long')
            entry_price = pos.get('entry_price')
            mark_price = pos.get('mark_price', entry_price)
            stored_stop_order_id = pos.get('stop_order_id')
            stored_stop_loss_price = pos.get('stop_loss_price')
            
            # Step 1: Check ALL open stop orders for this symbol on exchange
            # This prevents creating duplicate SLs and gives us order data without needing fetch_order
            # Bybit's fetchOrder() has limitations (only last 500 orders), so we use fetch_open_orders instead
            existing_stop_orders = []  # List of stop order dicts with full order data
            stored_order_found_in_open_orders = None  # Full order data if stored order is found
            fetch_orders_failed = False
            
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    order_id = order.get('id')
                    order_type = order.get('type', '').lower()
                    order_info = order.get('info', {})
                    order_type_info = order_info.get('stopOrderType', '').lower()
                    
                    # Check if it's a stop-loss order (stop market/limit)
                    is_stop_order = (
                        'stop' in order_type or
                        'stop' in order_type_info or
                        order_info.get('stopOrderType') in ['StopMarket', 'StopLimit', 'Stop']
                    )
                    
                    if is_stop_order:
                        order_dict = {
                            'id': order_id,
                            'type': order_type,
                            'info': order_info,
                            'full_order': order  # Store full order data for later use
                        }
                        existing_stop_orders.append(order_dict)
                        
                        # Check if this is the stored order we're looking for
                        if stored_stop_order_id and order_id == stored_stop_order_id:
                            stored_order_found_in_open_orders = order_dict
                        
                        self.logger.info(
                            f"Found existing stop order {order_id} for {symbol}: {order_type}"
                        )
            except Exception as e:
                fetch_orders_failed = True
                self.logger.warning(
                    f"⚠️ Error fetching open orders for {symbol} to check for duplicates: {e}"
                )
            
            # Step 2: If we have a stored order ID and found it in open orders, we're done
            # This is faster and avoids the fetchOrder() limitation
            if stored_stop_order_id and stored_order_found_in_open_orders:
                stored_order_verified = True
                self.logger.debug(
                    f"✅ Stored SL order {stored_stop_order_id} verified on exchange for {symbol} "
                    "(found in open orders)"
                )
                # If stored order exists, we're done - no need to check for duplicates
                results[symbol] = {
                    'status': 'ok',
                    'stop_order_id': stored_stop_order_id,
                    'stop_loss_price': stored_stop_loss_price,
                    'message': "Stop-loss order verified on exchange"
                }
                continue  # Skip the rest of the checks for this symbol
            
            # Determine if stop order exists on exchange
            stop_order_exists_on_exchange = len(existing_stop_orders) > 0
            
            # If we found existing stop orders on exchange, use the correct one and cancel duplicates
            if existing_stop_orders:
                # CRITICAL: If we have a stored order ID, prioritize it (it's the "official" one)
                # Otherwise, use the first existing order
                if stored_stop_order_id:
                    # Find the stored order in the list of existing orders
                    stored_order_in_list = next(
                        (o for o in existing_stop_orders if o['id'] == stored_stop_order_id),
                        None
                    )
                    if stored_order_in_list:
                        # Stored order exists - use it and cancel all others
                        # CRITICAL: Only cancel duplicates if stored order is verified to exist
                        existing_order_id = stored_stop_order_id
                        orders_to_cancel = [o for o in existing_stop_orders if o['id'] != stored_stop_order_id]
                        self.logger.info(
                            f"✅ Found stored SL order {existing_order_id} for {symbol} on exchange. "
                            f"Using this and will cancel {len(orders_to_cancel)} verified duplicate(s)."
                        )
                        
                        # CRITICAL: Double-check we're not accidentally cancelling the stored order
                        if existing_order_id in [o['id'] for o in orders_to_cancel]:
                            self.logger.error(
                                f"❌ CRITICAL: Logic error! Stored SL order {existing_order_id} found in duplicate list for {symbol}! "
                                "Aborting duplicate cancellation to prevent position closure."
                            )
                            orders_to_cancel = []  # Clear the list to prevent cancellation
                    else:
                        # Stored order doesn't exist, but we have other orders
                        # Use the first one and cancel all others (including trying to cancel the stored one)
                        existing_order_id = existing_stop_orders[0]['id']
                        orders_to_cancel = existing_stop_orders[1:]  # Cancel all except first
                        self.logger.warning(
                            f"⚠️ Stored SL order {stored_stop_order_id} not found for {symbol}. "
                            f"Using {existing_order_id} instead and cancelling {len(orders_to_cancel)} duplicate(s)."
                        )
                        # Also try to cancel the stored order if it's different (orphaned/stale order)
                        if stored_stop_order_id != existing_order_id:
                            try:
                                # CRITICAL: Pass params for conditional orders to avoid cancelling wrong orders
                                cancel_params = {
                                    'category': 'linear',
                                    'orderFilter': 'Stop'
                                }
                                self.exchange.cancel_order(stored_stop_order_id, symbol, params=cancel_params)
                                self.logger.debug(f"Cancelled orphaned stored SL order {stored_stop_order_id} for {symbol}")
                            except (ccxt.OrderNotFound, ccxt.ExchangeError):
                                pass  # Already cancelled/doesn't exist - this is fine
                else:
                    # No stored order ID - use first existing and cancel rest
                    existing_order_id = existing_stop_orders[0]['id']
                    orders_to_cancel = existing_stop_orders[1:]
                    self.logger.info(
                        f"✅ Found existing SL order {existing_order_id} for {symbol} on exchange. "
                        f"Using this and cancelling {len(orders_to_cancel)} duplicate(s)."
                    )
                
                # Cancel all duplicate/extra stop orders
                # CRITICAL: Use proper params for conditional orders to avoid cancelling wrong orders or closing positions
                # CRITICAL: Double-check that we're NOT cancelling the active SL order
                if orders_to_cancel:
                    # CRITICAL: Verify we're not accidentally trying to cancel the active SL
                    orders_to_cancel_filtered = []
                    for duplicate_order in orders_to_cancel:
                        duplicate_order_id = duplicate_order['id']
                        # Ensure we're not cancelling the active SL order
                        if duplicate_order_id == existing_order_id:
                            self.logger.error(
                                f"⚠️ CRITICAL: Attempted to cancel active SL order {duplicate_order_id} for {symbol}! "
                                "Skipping cancellation to prevent position closure."
                            )
                            continue
                        
                        # Also double-check it's not the stored order if we have one
                        if stored_stop_order_id and duplicate_order_id == stored_stop_order_id:
                            self.logger.warning(
                                f"⚠️ Attempted to cancel stored SL order {duplicate_order_id} for {symbol} as duplicate. "
                                "This should have been handled separately. Skipping."
                            )
                            continue
                        
                        orders_to_cancel_filtered.append(duplicate_order)
                    
                    if orders_to_cancel_filtered:
                        self.logger.warning(
                            f"⚠️ Cancelling {len(orders_to_cancel_filtered)} duplicate stop-loss order(s) for {symbol} "
                            f"(active SL {existing_order_id} will be preserved)..."
                        )
                        for duplicate_order in orders_to_cancel_filtered:
                            duplicate_order_id = duplicate_order['id']
                            try:
                                # CRITICAL: Pass params for conditional orders to ensure we cancel the right order
                                # Without these params, Bybit might cancel the wrong order or even close positions
                                cancel_params = {
                                    'category': 'linear',
                                    'orderFilter': 'Stop'  # 'Stop' for conditional orders (SL/TP)
                                }
                                
                                self.logger.debug(
                                    f"Cancelling duplicate SL order {duplicate_order_id} for {symbol} "
                                    f"(using params: {cancel_params}). Active SL {existing_order_id} will remain."
                                )
                                
                                self.exchange.cancel_order(duplicate_order_id, symbol, params=cancel_params)
                                
                                self.logger.info(
                                    f"✅ Cancelled duplicate stop order {duplicate_order_id} for {symbol} "
                                    f"(active SL {existing_order_id} preserved)"
                                )
                                
                                # CRITICAL: Verify position and active SL are still open after cancellation
                                # This prevents accidentally closing positions when cancelling duplicates
                                time.sleep(0.5)  # Brief delay for cancellation to settle
                                try:
                                    # Verify position is still open
                                    if portfolio_state:
                                        portfolio_state.update()
                                        pos_after_cancel = portfolio_state.get_position(symbol)
                                        if not pos_after_cancel:
                                            self.logger.error(
                                                f"❌ CRITICAL: Position {symbol} was closed after cancelling duplicate SL {duplicate_order_id}! "
                                                "This should not happen. Please investigate."
                                            )
                                        else:
                                            contracts_after = abs(pos_after_cancel.get('contracts', 0))
                                            if contracts_after < 0.001:
                                                self.logger.error(
                                                    f"❌ CRITICAL: Position {symbol} size dropped to {contracts_after:.6f} after cancelling duplicate SL! "
                                                    "This should not happen. Please investigate."
                                                )
                                            else:
                                                self.logger.debug(
                                                    f"✅ Verified: Position {symbol} still open ({contracts_after:.6f} contracts) after cancelling duplicate SL"
                                                )
                                    
                                    # Verify active SL is still open (use fetch_open_orders instead of fetch_order)
                                    # Bybit's fetchOrder() has limitations (only last 500 orders), so we check via fetch_open_orders
                                    try:
                                        open_orders_after = self.exchange.fetch_open_orders(symbol)
                                        active_sl_found = any(
                                            o.get('id') == existing_order_id for o in open_orders_after
                                        )
                                        if active_sl_found:
                                            self.logger.debug(
                                                f"✅ Verified: Active SL order {existing_order_id} still open for {symbol}"
                                            )
                                        else:
                                            self.logger.error(
                                                f"❌ CRITICAL: Active SL order {existing_order_id} not found in open orders "
                                                f"after cancelling duplicate SL {duplicate_order_id} for {symbol}! "
                                                "This should not happen. Please investigate."
                                            )
                                    except Exception as verify_order_error:
                                        # If we can't verify, log warning but don't fail - position check is more important
                                        self.logger.warning(
                                            f"Could not verify active SL order {existing_order_id} status after cancelling duplicate: {verify_order_error}"
                                        )
                                except Exception as verify_error:
                                    self.logger.warning(
                                        f"Could not verify position/SL status after cancelling duplicate SL {duplicate_order_id} for {symbol}: {verify_error}"
                                    )
                                
                            except (ccxt.OrderNotFound, ccxt.ExchangeError) as e:
                                # Order already doesn't exist - this is fine, just log
                                self.logger.debug(
                                    f"Duplicate stop order {duplicate_order_id} already cancelled/doesn't exist: {e}"
                                )
                            except Exception as e:
                                error_msg = f"Error cancelling duplicate stop order {duplicate_order_id} for {symbol}: {e}"
                                self.logger.error(error_msg)
                                # Don't fail completely if one duplicate cancellation fails - continue with others
                    else:
                        self.logger.warning(
                            f"⚠️ All duplicate orders filtered out for {symbol} (would have cancelled active SL). "
                            "No duplicate orders will be cancelled."
                        )
                
                # Update portfolio state with the existing order ID if it's different from stored
                if existing_order_id != stored_stop_order_id and portfolio_state:
                    # Get the stop loss price from the order data we already have from fetch_open_orders
                    # No need to call fetch_order (which has Bybit limitations) - we already have the data
                    trigger_price = stored_stop_loss_price  # Default to stored price
                    
                    # Find the order in our existing_stop_orders list
                    existing_order_data = next(
                        (o for o in existing_stop_orders if o.get('id') == existing_order_id),
                        None
                    )
                    
                    if existing_order_data and existing_order_data.get('full_order'):
                        # Extract trigger price from the order data we already have
                        full_order = existing_order_data['full_order']
                        trigger_price = (
                            full_order.get('triggerPrice') or
                            full_order.get('stopPrice') or
                            full_order.get('price') or
                            existing_order_data.get('info', {}).get('triggerPrice') or
                            existing_order_data.get('info', {}).get('stopPrice') or
                            stored_stop_loss_price
                        )
                    
                    # Update portfolio state with the order ID and trigger price
                    if portfolio_state:
                        portfolio_state.set_position_metadata(
                            symbol,
                            stop_order_id=existing_order_id,
                            stop_loss_price=float(trigger_price) if trigger_price else stored_stop_loss_price,
                        )
                
                # We found and are using existing SL order, we're done
                results[symbol] = {
                    'status': 'ok',
                    'stop_order_id': existing_order_id,
                    'stop_loss_price': stored_stop_loss_price,  # Keep existing price or update above
                    'message': f"Using existing stop-loss order {existing_order_id} found on exchange"
                }
                continue  # Skip creating new SL order
            
            # If we couldn't fetch open orders and don't have a verified stored order, 
            # we should skip creating a new SL to avoid duplicates
            if fetch_orders_failed and not stop_order_exists_on_exchange:
                self.logger.error(
                    f"❌ Cannot verify existing SL orders for {symbol} (fetch_open_orders failed). "
                    f"Skipping SL creation to avoid duplicates. Please check manually."
                )
                results[symbol] = {
                    'status': 'failed',
                    'stop_order_id': None,
                    'stop_loss_price': stored_stop_loss_price,
                    'message': "Cannot verify existing orders - skipping to avoid duplicates"
                }
                continue  # Skip creating new SL order
            
            
            # Step 2: If SL order missing, determine stop-loss price
            if not stop_order_exists_on_exchange:
                stop_loss_price = stored_stop_loss_price
                
                # If stored price missing, use fallback percentage-based calculation
                if stop_loss_price is None:
                    if entry_price:
                        try:
                            # Conservative fallback: 5% below/above entry
                            if side == 'long':
                                stop_loss_price = entry_price * 0.95
                            else:
                                stop_loss_price = entry_price * 1.05
                            
                            self.logger.warning(
                                f"No stored stop-loss price for {symbol}. "
                                f"Using fallback: {stop_loss_price:.2f} "
                                f"({'5% below' if side == 'long' else '5% above'} entry)"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Could not calculate fallback SL for {symbol}: {e}"
                            )
                            results[symbol] = {
                                'status': 'failed',
                                'stop_order_id': None,
                                'stop_loss_price': None,
                                'message': f"Could not determine stop-loss price: {e}"
                            }
                            continue
                    else:
                        results[symbol] = {
                            'status': 'failed',
                            'stop_order_id': None,
                            'stop_loss_price': None,
                            'message': "No stored stop-loss price and cannot recalculate (missing entry_price)"
                        }
                        continue
                
                # Step 3: Place missing SL order with retry logic
                self.logger.warning(
                    f"⚠️ Position {symbol} ({side.upper()} {contracts:.4f}) has NO stop-loss order. "
                    f"Creating protective SL at {stop_loss_price:.2f}..."
                )
                
                # Retry logic: try up to 3 times with delays
                max_retries = 3
                retry_delay = 2.0  # seconds
                new_stop_order_id = None
                
                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            self.logger.info(
                                f"Retrying SL creation for {symbol} (attempt {attempt}/{max_retries})..."
                            )
                            time.sleep(retry_delay)
                        
                        new_stop_order_id = self._place_stop_loss_order(
                            symbol,
                            side,
                            contracts,
                            stop_loss_price,
                            portfolio_state,
                            entry_price,
                        )
                        
                        if new_stop_order_id:
                            results[symbol] = {
                                'status': 'created',
                                'stop_order_id': new_stop_order_id,
                                'stop_loss_price': stop_loss_price,
                                'message': f"Created missing stop-loss order: {new_stop_order_id} (attempt {attempt})"
                            }
                            self.logger.info(
                                f"✅ Successfully created protective SL for {symbol}: order_id={new_stop_order_id} (attempt {attempt})"
                            )
                            break  # Success, exit retry loop
                        else:
                            if attempt < max_retries:
                                self.logger.warning(
                                    f"SL creation for {symbol} returned None (attempt {attempt}/{max_retries}). Will retry..."
                                )
                            else:
                                results[symbol] = {
                                    'status': 'failed',
                                    'stop_order_id': None,
                                    'stop_loss_price': stop_loss_price,
                                    'message': f"Failed to create stop-loss order after {max_retries} attempts (returned None)"
                                }
                                self.logger.error(
                                    f"❌ Failed to create protective SL for {symbol} after {max_retries} attempts"
                                )
                    
                    except Exception as e:
                        if attempt < max_retries:
                            self.logger.warning(
                                f"Exception creating SL for {symbol} (attempt {attempt}/{max_retries}): {e}. Will retry..."
                            )
                        else:
                            results[symbol] = {
                                'status': 'failed',
                                'stop_order_id': None,
                                'stop_loss_price': stop_loss_price,
                                'message': f"Exception creating stop-loss after {max_retries} attempts: {e}"
                            }
                            self.logger.error(
                                f"❌ Exception creating protective SL for {symbol} after {max_retries} attempts: {e}",
                                exc_info=True
                            )
        
        return results

