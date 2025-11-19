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

                # Check stop loss
                stop_loss = pos.get("stop_loss")
                if stop_loss:
                    stop_hit = (signal == "long" and current_price <= stop_loss) or (
                        signal == "short" and current_price >= stop_loss
                    )
                    if stop_hit:
                        # Realize PnL at stop
                        exit_price = stop_loss
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
                                "reason": "stop_loss",
                            }
                        )
                        del positions[symbol]

            # Apply funding PnL approximation for open positions (if configured)
            if funding_rate_per_8h != 0.0 and positions:
                hours_in_bar = 1.0  # Assumes 1h bars; for other timeframes this should be adjusted
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

                    # Open position
                    notional = abs(adjusted_size) * entry_price
                    positions[symbol] = {
                        "size": adjusted_size,
                        "contracts": adjusted_size,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "signal": signal,
                        "entry_time": timestamp,
                        "notional": notional,
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
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate returns
        equity_series = pd.Series(equity_history)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 0:
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() > 0 else 0  # Annualized for hourly data
            max_drawdown = self._calculate_max_drawdown(equity_series)
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            profit_factor = 0
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': trades,
            'equity_history': equity_history,
            'funding_pnl_total': funding_pnl_total,
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        return drawdown.min()

