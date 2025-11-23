"""Vectorized backtester for historical data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date

from ..config import BotConfig
from ..signals.trend import TrendSignalGenerator, calculate_atr
from ..signals.cross_sectional import CrossSectionalSignalGenerator
from ..risk.position_sizing import PositionSizer
from ..risk.portfolio_limits import PortfolioLimits
from ..logging_utils import get_logger

logger = get_logger(__name__)

_TIMEFRAME_TO_HOURS = {
    "1m": 1 / 60,
    "3m": 3 / 60,
    "5m": 5 / 60,
    "15m": 15 / 60,
    "30m": 30 / 60,
    "45m": 45 / 60,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "12h": 12.0,
    "1d": 24.0,
}


def parse_timeframe_to_hours(timeframe: str) -> float:
    """Convert timeframe string (e.g. '4h') to hours."""
    tf = timeframe.lower().strip()
    if tf in _TIMEFRAME_TO_HOURS:
        return _TIMEFRAME_TO_HOURS[tf]
    if tf.endswith("m"):
        return int(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(tf[:-1])
    if tf.endswith("d"):
        return float(tf[:-1]) * 24.0
    raise ValueError(f"Unsupported timeframe: {timeframe}")


class _SimulatedExchange:
    """
    Minimal simulated exchange for backtesting risk logic.
    
    Provides the subset of methods used by PositionSizer / PortfolioLimits.
    Assumes contractSize = 1 for all symbols and no rounding constraints.
    """

    def get_market_info(self, symbol: str) -> Dict:
        return {
            "precision": {"price": 8, "amount": 8},
            "limits": {
                "amount": {"min": 0.0, "max": None},
                "cost": {"min": 0.0},
            },
            "contractSize": 1.0,
        }

    def round_amount(self, symbol: str, amount: float) -> float:
        # For backtests we don't need exchange-specific rounding
        return float(amount)
    
    def validate_order_size(self, symbol: str, amount: float, price: float) -> Tuple[bool, Optional[str]]:
        """
        Validate order size meets exchange requirements.
        
        For backtests, we're lenient - only reject truly invalid orders.
        Returns:
            (is_valid, error_message)
        """
        market_info = self.get_market_info(symbol)
        limits = market_info['limits']
        contract_size = market_info['contractSize']
        
        # Check minimum amount
        min_amount = limits['amount']['min']
        if min_amount is not None and min_amount > 0:
            if amount < min_amount:
                return False, f"Amount {amount} below minimum {min_amount}"
        
        # Check minimum cost (notional)
        min_cost = limits['cost']['min']
        if min_cost is not None and min_cost > 0:
            cost = amount * price * contract_size
            if cost < min_cost:
                return False, f"Order cost {cost} below minimum {min_cost}"
        
        return True, None


class _BacktestPortfolioState:
    """Lightweight portfolio state for backtesting risk checks."""

    def __init__(self, equity: float, positions: Dict[str, Dict]):
        self.equity = equity
        self.positions = positions

    def get_total_notional(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            total += abs(pos.get("notional", 0.0))
        return total

    def get_leverage(self) -> float:
        if self.equity <= 0:
            return 0.0
        return self.get_total_notional() / self.equity


class Backtester:
    """Vectorized backtester for trading strategies."""

    def __init__(self, config: BotConfig):
        """
        Initialize backtester.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.trend_gen = TrendSignalGenerator(config.strategy.trend)
        self.cross_sectional_gen = CrossSectionalSignalGenerator(
            config.strategy.cross_sectional
        )
        # Use simulated exchange + live risk configs so sizing behavior matches live
        self._sim_exchange = _SimulatedExchange()
        self.position_sizer = PositionSizer(config.risk, self._sim_exchange)
        self.portfolio_limits = PortfolioLimits(config.risk, self._sim_exchange)

    def backtest(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        taker_fee: float = 0.00055,  # Bybit taker fee
        universe_history: Optional[Dict[date, Set[str]]] = None,
        funding_rate_per_8h: float = 0.0,  # Optional constant funding rate approximation
        stop_slippage_bps: float = 10.0,  # Slippage in basis points when stop is hit
        tp_slippage_bps: float = 5.0,  # Slippage for take-profit orders
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            initial_capital: Starting capital
            taker_fee: Trading fee (0.00055 = 0.055% for Bybit)
        
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        equity_history = [capital]
        timestamps_list = []  # Track timestamps for metrics calculation
        trades = []
        funding_pnl_total = 0.0
        # symbol -> {size, contracts, entry_price, stop_loss, signal, entry_time, notional}
        positions: Dict[str, Dict] = {}
        
        # Get all timestamps
        all_timestamps = set()
        for df in symbol_data.values():
            if not df.empty:
                all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        if not timestamps:
            return {'error': 'No data to backtest'}
        
        # Track cross-sectional rebalance time
        last_rebalance_time = None
        rebalance_frequency_hours = self.config.strategy.cross_sectional.rebalance_frequency_hours
        hours_in_bar = parse_timeframe_to_hours(self.config.exchange.timeframe)
        
        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            current_equity = capital
            unrealized_pnl = 0.0

            # Calculate unrealized PnL from open positions and update notionals
            for symbol, pos in list(positions.items()):
                if symbol not in symbol_data:
                    continue

                df_sym = symbol_data[symbol]
                if timestamp not in df_sym.index:
                    continue

                current_price = df_sym.loc[timestamp, "close"]
                high_price = df_sym.loc[timestamp, "high"]
                low_price = df_sym.loc[timestamp, "low"]
                entry_price = pos["entry_price"]
                size = pos["size"]
                signal = pos["signal"]

                # Update notional at current mark
                pos["notional"] = abs(size) * current_price

                # Calculate PnL
                if signal == "long":
                    pnl = (current_price - entry_price) * size
                else:  # short
                    pnl = (entry_price - current_price) * size

                unrealized_pnl += pnl

                # Update trailing stop if enabled
                stop_loss = pos.get("stop_loss")
                if stop_loss and self.config.strategy.trend.use_trailing_stop:
                    # Update highest/lowest price for trailing
                    highest = pos.get("highest_price", entry_price)
                    lowest = pos.get("lowest_price", entry_price)
                    
                    if signal == "long":
                        if current_price > highest:
                            pos["highest_price"] = current_price
                            highest = current_price
                    else:  # short
                        if current_price < lowest:
                            pos["lowest_price"] = current_price
                            lowest = current_price
                    
                    # Check if trailing should activate
                    if signal == "long":
                        profit_pct = (current_price - entry_price) / entry_price
                        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
                        if profit_pct >= (stop_distance_pct * self.config.strategy.trend.trailing_stop_activation_rr):
                            # Calculate new trailing stop
                            atr_estimate = current_price * 0.01  # Rough ATR approximation
                            new_stop = current_price - (atr_estimate * self.config.strategy.trend.trailing_stop_atr_multiplier)
                            # Only move stop up, never down
                            if new_stop > stop_loss:
                                pos["stop_loss"] = new_stop
                                stop_loss = new_stop
                    # Similar for short (trail down)
                    elif signal == "short":
                        profit_pct = (entry_price - current_price) / entry_price
                        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
                        if profit_pct >= (stop_distance_pct * self.config.strategy.trend.trailing_stop_activation_rr):
                            atr_estimate = current_price * 0.01
                            new_stop = current_price + (atr_estimate * self.config.strategy.trend.trailing_stop_atr_multiplier)
                            if new_stop < stop_loss or stop_loss == 0:  # First trailing stop
                                pos["stop_loss"] = new_stop
                                stop_loss = new_stop

                # Check stop loss using bar high/low (more realistic)
                stop_loss = pos.get("stop_loss")
                stop_hit = False
                exit_price = None
                exit_reason = None
                
                if stop_loss:
                    # Use high/low to check if stop was hit intrabar
                    if signal == "long":
                        # Long: stop is below entry, check if low touched stop
                        if low_price <= stop_loss:
                            stop_hit = True
                            # Model slippage: assume we fill slightly worse than stop
                            slippage = stop_loss * (stop_slippage_bps / 10000.0)
                            exit_price = stop_loss - slippage  # Worse fill for long stop
                            exit_reason = "stop_loss"
                    else:  # short
                        # Short: stop is above entry, check if high touched stop
                        if high_price >= stop_loss:
                            stop_hit = True
                            # Model slippage: assume we fill slightly worse than stop
                            slippage = stop_loss * (stop_slippage_bps / 10000.0)
                            exit_price = stop_loss + slippage  # Worse fill for short stop
                            exit_reason = "stop_loss"
                
                # Check take-profit if configured
                take_profit = pos.get("take_profit")
                if not stop_hit and take_profit:
                    if signal == "long":
                        # Long: TP is above entry, check if high touched TP
                        if high_price >= take_profit:
                            stop_hit = True  # Reuse flag for exit
                            # TP usually has less slippage (limit order)
                            slippage = take_profit * (tp_slippage_bps / 10000.0)
                            exit_price = take_profit - slippage  # Slightly better than TP
                            exit_reason = "take_profit"
                    else:  # short
                        # Short: TP is below entry, check if low touched TP
                        if low_price <= take_profit:
                            stop_hit = True
                            slippage = take_profit * (tp_slippage_bps / 10000.0)
                            exit_price = take_profit + slippage
                            exit_reason = "take_profit"
                
                # Check time-based exit
                if not stop_hit and self.config.strategy.trend.max_holding_hours:
                    entry_time = pos.get("entry_time")
                    if entry_time:
                        hours_held = (timestamp - entry_time).total_seconds() / 3600
                        if hours_held >= self.config.strategy.trend.max_holding_hours:
                            stop_hit = True
                            exit_price = current_price  # Exit at close
                            exit_reason = "max_holding_period"
                
                if stop_hit and exit_price:
                    # Realize PnL at exit
                    if signal == "long":
                        pnl = (exit_price - entry_price) * size
                    else:
                        pnl = (entry_price - exit_price) * size

                    fee = abs(size) * exit_price * taker_fee
                    current_equity += pnl - fee
                    trades.append(
                        {
                            "symbol": symbol,
                            "entry_time": pos["entry_time"],
                            "exit_time": timestamp,
                            "side": signal,
                            "size": size,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "return_pct": (pnl / (entry_price * size)) * 100,
                            "reason": exit_reason or "stop_loss",
                        }
                    )
                    del positions[symbol]
                    continue  # Skip rest of position update for this symbol

            # Apply funding PnL approximation for open positions (if configured)
            if funding_rate_per_8h != 0.0 and positions:
                funding_pnl_bar = 0.0
                for symbol, pos in positions.items():
                    notional_here = abs(pos.get("notional", 0.0))
                    if notional_here <= 0:
                        continue
                    funding_pnl_symbol = (
                        notional_here * funding_rate_per_8h * (hours_in_bar / 8.0)
                    )
                    funding_pnl_bar += funding_pnl_symbol
                current_equity += funding_pnl_bar
                funding_pnl_total += funding_pnl_bar

            current_equity += unrealized_pnl
            equity_history.append(current_equity)
            
            # Check if we should rebalance (cross-sectional)
            should_rebalance = False
            if last_rebalance_time is None:
                should_rebalance = True
            else:
                hours_diff = (timestamp - last_rebalance_time).total_seconds() / 3600
                if hours_diff >= rebalance_frequency_hours:
                    should_rebalance = True
            
            # Determine allowed symbols based on universe (if provided)
            allowed_symbols: Optional[Set[str]] = None
            if universe_history is not None:
                as_of_date = timestamp.date()
                allowed_symbols = universe_history.get(as_of_date, set())

            # Generate signals for symbols with data at this timestamp
            symbol_signals: Dict[str, Dict] = {}
            symbol_data_at_timestamp: Dict[str, pd.DataFrame] = {}

            for symbol, df in symbol_data.items():
                if allowed_symbols is not None and symbol not in allowed_symbols:
                    continue
                if timestamp not in df.index:
                    continue

                df_up_to_now = df.loc[:timestamp]
                if len(df_up_to_now) < self.config.strategy.trend.ma_long:
                    continue

                symbol_data_at_timestamp[symbol] = df_up_to_now

                # Generate trend signal
                trend_signal = self.trend_gen.generate_signal(df_up_to_now)
                symbol_signals[symbol] = trend_signal
            
            # Cross-sectional selection
            if should_rebalance and symbol_data_at_timestamp:
                selected_symbols = self.cross_sectional_gen.select_top_symbols(
                    symbol_data_at_timestamp,
                    symbol_signals,
                    self.config.strategy.cross_sectional.require_trend_alignment
                )
                
                # Close positions not in selected symbols
                symbols_to_close = [s for s in positions.keys() if s not in selected_symbols]
                for symbol in symbols_to_close:
                    pos = positions[symbol]
                    df_sym = symbol_data[symbol]
                    if timestamp in df_sym.index:
                        exit_price = df_sym.loc[timestamp, "close"]
                        entry_price = pos["entry_price"]
                        size = pos["size"]
                        signal = pos["signal"]

                        if signal == "long":
                            pnl = (exit_price - entry_price) * size
                        else:
                            pnl = (entry_price - exit_price) * size

                        fee = abs(size) * exit_price * taker_fee
                        current_equity += pnl - fee
                        trades.append(
                            {
                                "symbol": symbol,
                                "entry_time": pos["entry_time"],
                                "exit_time": timestamp,
                                "side": signal,
                                "size": size,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "pnl": pnl,
                                "return_pct": (pnl / (entry_price * size)) * 100,
                                "reason": "rebalance",
                            }
                        )
                        del positions[symbol]

                # Open new positions for selected symbols
                for symbol in selected_symbols:
                    if symbol in positions:
                        # Already have position, skip
                        continue

                    signal_dict = symbol_signals.get(symbol)
                    if not signal_dict or signal_dict["signal"] == "flat":
                        continue

                    signal = signal_dict["signal"]
                    entry_price = signal_dict["entry_price"]
                    stop_loss = signal_dict["stop_loss"]

                    if not stop_loss:
                        continue

                    # Position sizing via PositionSizer (uses per-trade risk fraction and Kelly)
                    size, err = self.position_sizer.calculate_position_size(
                        symbol=symbol,
                        equity=current_equity,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        signal=signal,
                    )
                    if size <= 0:
                        continue

                    # Apply portfolio-level limits (leverage, concentration, max positions)
                    sim_portfolio_state = _BacktestPortfolioState(current_equity, positions)
                    adjusted_size, scale_reason = self.portfolio_limits.scale_position_for_limits(
                        sim_portfolio_state,
                        symbol,
                        size,
                        entry_price,
                        signal,
                    )
                    if adjusted_size <= 0:
                        continue

                    # Check max positions again (scale_position_for_limits may not enforce count)
                    within_max, max_error = self.portfolio_limits.check_max_positions(
                        sim_portfolio_state,
                        symbol,
                    )
                    if not within_max:
                        continue

                    # Calculate take-profit price if configured
                    take_profit = None
                    if self.config.strategy.trend.take_profit_rr and stop_loss:
                        stop_distance = abs(entry_price - stop_loss)
                        if signal == "long":
                            take_profit = entry_price + (stop_distance * self.config.strategy.trend.take_profit_rr)
                        else:  # short
                            take_profit = entry_price - (stop_distance * self.config.strategy.trend.take_profit_rr)
                    
                    # Open position
                    notional = abs(adjusted_size) * entry_price
                    positions[symbol] = {
                        "size": adjusted_size,
                        "contracts": adjusted_size,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "signal": signal,
                        "entry_time": timestamp,
                        "notional": notional,
                        # For trailing stop
                        "highest_price": entry_price if signal == "long" else None,
                        "lowest_price": entry_price if signal == "short" else None,
                    }

                    # Pay entry fee
                    current_equity -= abs(adjusted_size) * entry_price * taker_fee
                
                last_rebalance_time = timestamp
            
            capital = current_equity - unrealized_pnl  # Update capital for next iteration
        
        # Close all remaining positions at end
        for symbol, pos in positions.items():
            if symbol not in symbol_data:
                continue
            
            df = symbol_data[symbol]
            if df.empty:
                continue
            
            exit_price = df['close'].iloc[-1]
            entry_price = pos['entry_price']
            size = pos['size']
            signal = pos['signal']
            
            if signal == 'long':
                pnl = (exit_price - entry_price) * size
            else:
                pnl = (entry_price - exit_price) * size
            
            trades.append({
                'symbol': symbol,
                'entry_time': pos['entry_time'],
                'exit_time': df.index[-1],
                'side': signal,
                'size': size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': (pnl / (entry_price * size)) * 100,
                'reason': 'end_of_backtest'
            })
        
        # Calculate metrics
        final_equity = equity_history[-1] if equity_history else initial_capital
        total_return_pct = (final_equity - initial_capital) / initial_capital * 100
        total_return = total_return_pct / 100.0  # As decimal for consistency
        
        # Calculate time period for annualization
        if timestamps:
            start_time = timestamps[0]
            end_time = timestamps[-1]
            time_diff = (end_time - start_time).total_seconds() / (365.25 * 24 * 3600)  # Years
            days_diff = (end_time - start_time).total_seconds() / (24 * 3600)  # Days
        else:
            time_diff = 0
            days_diff = 0
        
        # Calculate returns
        equity_series = pd.Series(equity_history)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe ratio (annualized)
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation only)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (returns.mean() / downside_returns.std() * np.sqrt(252 * 24))
            else:
                sortino_ratio = float('inf') if returns.mean() > 0 else 0
            
            max_drawdown = self._calculate_max_drawdown(equity_series)
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
            
            # Annualized return
            if time_diff > 0:
                annualized_return = ((final_equity / initial_capital) ** (1.0 / time_diff) - 1.0) if time_diff > 0 else total_return
            else:
                annualized_return = 0.0
            
            # Trades per day
            trades_per_day = len(trades) / days_diff if days_diff > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            win_rate = 0
            profit_factor = 0
            annualized_return = 0.0
            trades_per_day = 0
        
        # Calculate total fees (sum of all fees from trades)
        total_fees = sum([abs(t['size']) * t['exit_price'] * taker_fee for t in trades if 'exit_price' in t])
        total_fees += sum([abs(t['size']) * t['entry_price'] * taker_fee for t in trades if 'entry_price' in t])
        
        # Calculate average leverage (average position size / capital)
        if trades:
            avg_position_size = np.mean([abs(t['size']) * t['entry_price'] for t in trades if 'entry_price' in t])
            avg_leverage = (avg_position_size / initial_capital) if initial_capital > 0 else 0
        else:
            avg_leverage = 0
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,  # As decimal
            'total_return_pct': total_return_pct,  # As percentage for backwards compatibility
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_per_day': trades_per_day,
            'total_fees': total_fees,
            'avg_leverage': avg_leverage,
            'trades': trades,
            'equity_history': equity_history,
            'funding_pnl_total': funding_pnl_total,
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        return drawdown.min()

