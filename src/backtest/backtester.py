"""Vectorized backtester for historical data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date
from unittest.mock import Mock

from ..config import BotConfig
from ..signals.trend import TrendSignalGenerator, calculate_atr
from ..signals.cross_sectional import CrossSectionalSignalGenerator
from ..signals.funding_opportunity import FundingOpportunityGenerator
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
        
        # Initialize funding opportunity generator if enabled
        self.funding_opportunity_gen = None
        # Defensive check: ensure funding_opportunity config exists and is enabled
        if (hasattr(config.strategy, 'funding_opportunity') and 
            hasattr(config.strategy.funding_opportunity, 'enabled') and
            config.strategy.funding_opportunity.enabled):
            # Create mock exchange for funding rates (backtest uses constant/approximated rates)
            mock_exchange = Mock()
            # Default funding rate (can be overridden per symbol via funding_rate_history)
            mock_exchange.fetch_funding_rate = Mock(return_value={'fundingRate': 0.0})
            self.funding_opportunity_gen = FundingOpportunityGenerator(
                config.strategy.funding_opportunity,
                mock_exchange
            )

    def backtest(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        taker_fee: float = 0.00055,  # Bybit taker fee
        universe_history: Optional[Dict[date, Set[str]]] = None,
        funding_rate_per_8h: float = 0.0,  # Optional constant funding rate approximation
        funding_rate_history: Optional[Dict[str, pd.Series]] = None,  # Optional per-symbol time-varying funding rates
        stop_slippage_bps: float = 10.0,  # Slippage in basis points when stop is hit
        tp_slippage_bps: float = 5.0,  # Slippage for take-profit orders
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            symbol_data: Dictionary mapping symbol to OHLCV DataFrame
            initial_capital: Starting capital
            taker_fee: Trading fee (0.00055 = 0.055% for Bybit)
            universe_history: Optional dictionary mapping date to set of tradable symbols
            funding_rate_per_8h: Constant funding rate approximation (per 8h period)
            funding_rate_history: Optional dictionary mapping symbol to Series of funding rates over time
            stop_slippage_bps: Slippage in basis points when stop-loss is hit
            tp_slippage_bps: Slippage in basis points for take-profit orders
        
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
                # Try exact date match first
                allowed_symbols = universe_history.get(as_of_date)
                
                # If no exact match, find nearest prior date (universe membership is forward-looking)
                if allowed_symbols is None and universe_history:
                    # Find the most recent date <= as_of_date
                    prior_dates = [d for d in universe_history.keys() if d <= as_of_date]
                    if prior_dates:
                        nearest_date = max(prior_dates)
                        allowed_symbols = universe_history[nearest_date]
                    else:
                        # No prior dates found - use all symbols in symbol_data as fallback
                        # This handles edge cases where universe_history doesn't cover early dates
                        allowed_symbols = set(symbol_data.keys())
                        if i == 0:  # Log once at start
                            logger.debug(
                                f"Universe history missing date {as_of_date}, using all symbols as fallback "
                                f"(universe_history has {len(universe_history)} dates, earliest: {min(universe_history.keys()) if universe_history else 'N/A'})"
                            )
                
                # If still None (empty universe_history dict), use all symbols
                if allowed_symbols is None:
                    allowed_symbols = set(symbol_data.keys())
                    if i == 0:
                        logger.debug("Universe history is empty, using all symbols as fallback")

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
            
            # Generate funding opportunity signals if enabled
            funding_signals = {}
            funding_selected = []
            confluence_symbols = set()  # Initialize confluence symbols (not yet implemented in backtester)
            if (self.config.strategy.funding_opportunity.enabled and 
                self.funding_opportunity_gen and 
                should_rebalance):
                
                # Get funding rates for symbols (from history or constant approximation)
                funding_universe = list(symbol_data_at_timestamp.keys())
                
                # Update mock exchange to return funding rates
                if funding_rate_history:
                    def get_funding_rate(symbol: str):
                        if symbol in funding_rate_history:
                            series = funding_rate_history[symbol]
                            # Find rate for current timestamp
                            try:
                                if timestamp in series.index:
                                    return {'fundingRate': float(series.loc[timestamp])}
                                # Find nearest prior
                                prior = series.index[series.index <= timestamp]
                                if len(prior) > 0:
                                    return {'fundingRate': float(series.loc[prior[-1]])}
                            except:
                                pass
                        # Fallback to constant
                        return {'fundingRate': funding_rate_per_8h}
                else:
                    def get_funding_rate(symbol: str):
                        return {'fundingRate': funding_rate_per_8h}
                
                self.funding_opportunity_gen.exchange.fetch_funding_rate = Mock(side_effect=get_funding_rate)
                
                # Scan for funding opportunities
                try:
                    funding_opportunities = self.funding_opportunity_gen.scan_opportunities(
                        funding_universe,
                        symbol_data_at_timestamp
                    )
                    
                    # Select top opportunities
                    top_opportunities = funding_opportunities[:self.config.strategy.funding_opportunity.max_positions]
                    funding_selected = [opp.symbol for opp in top_opportunities]
                    
                    # Generate signals
                    for opp in top_opportunities:
                        signal = self.funding_opportunity_gen.generate_signal(
                            opp.symbol,
                            opp.funding_rate,
                            symbol_data_at_timestamp.get(opp.symbol)
                        )
                        signal['source'] = 'funding_opportunity'
                        signal['metadata'] = opp.metadata
                        funding_signals[opp.symbol] = signal
                except Exception as e:
                    self.logger.warning(f"Error generating funding signals in backtest: {e}")
                    funding_signals = {}
                    funding_selected = []
            
            # Cross-sectional selection (main strategy)
            if should_rebalance and symbol_data_at_timestamp:
                selected_symbols = self.cross_sectional_gen.select_top_symbols(
                    symbol_data_at_timestamp,
                    symbol_signals,
                    self.config.strategy.cross_sectional.require_trend_alignment
                )
            else:
                selected_symbols = []
            
            # Close positions not in selected symbols (main strategy)
            # Also close funding positions if they're not in funding_selected
            if should_rebalance:
                symbols_to_close = [s for s in positions.keys() if s not in selected_symbols]
                if self.config.strategy.funding_opportunity.enabled:
                    # Close funding positions not in funding_selected
                    funding_positions_to_close = [
                        s for s in positions.keys()
                        if positions[s].get('source') in ['funding_opportunity'] and s not in funding_selected
                    ]
                    symbols_to_close.extend(funding_positions_to_close)
                
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
                        # Source tracking (Phase 1: basic support)
                        "source": signal_dict.get("source", "main_strategy"),
                        "metadata": signal_dict.get("metadata", {}),
                    }

                    # Pay entry fee
                    current_equity -= abs(adjusted_size) * entry_price * taker_fee
                
                # Process funding strategy positions (excluding confluence)
                if (self.config.strategy.funding_opportunity.enabled and 
                    self.funding_opportunity_gen):
                    funding_strategy_symbols = [s for s in funding_selected if s not in confluence_symbols]
                    
                    # Calculate total funding exposure
                    total_funding_exposure = sum(
                        abs(pos.get('notional', 0)) for pos in positions.values()
                        if pos.get('source') in ['funding_opportunity', 'confluence']
                    )
                    
                    for symbol in funding_strategy_symbols:
                        if symbol in positions:
                            continue
                        
                        signal_dict = funding_signals.get(symbol)
                        if not signal_dict or signal_dict["signal"] == "flat":
                            continue
                        
                        signal = signal_dict["signal"]
                        entry_price = signal_dict["entry_price"]
                        stop_loss = signal_dict["stop_loss"]
                        
                        if not stop_loss or not entry_price:
                            continue
                        
                        # Funding-specific sizing
                        base_size_fraction = self.config.strategy.funding_opportunity.sizing.base_size_fraction
                        funding_rate = signal_dict.get('metadata', {}).get('funding_rate', 0.0)
                        abs_funding = abs(funding_rate)
                        
                        size_multiplier = self.config.strategy.funding_opportunity.sizing.size_multiplier
                        funding_multiplier = 1.0 + (abs_funding * size_multiplier)
                        target_size_fraction = min(
                            base_size_fraction * funding_multiplier,
                            self.config.strategy.funding_opportunity.sizing.max_position_size
                        )
                        
                        target_notional = current_equity * target_size_fraction
                        size = target_notional / entry_price
                        
                        # Apply stop loss risk limit
                        stop_distance = abs(entry_price - stop_loss) if signal == 'long' else abs(stop_loss - entry_price)
                        if stop_distance > 0:
                            risk_at_size = size * stop_distance
                            max_risk = current_equity * self.config.risk.per_trade_risk_fraction
                            if risk_at_size > max_risk:
                                size = (max_risk / stop_distance)
                        
                        # Check funding exposure limit
                        notional = size * entry_price
                        max_allowed_exposure = current_equity * self.config.strategy.funding_opportunity.risk.max_total_funding_exposure
                        if total_funding_exposure + notional > max_allowed_exposure:
                            max_additional = max_allowed_exposure - total_funding_exposure
                            if max_additional <= 0:
                                continue
                            size = min(size, max_additional / entry_price)
                            notional = size * entry_price
                        
                        if size <= 0:
                            continue
                        
                        # Check leverage
                        sim_portfolio_state = _BacktestPortfolioState(current_equity, positions)
                        adjusted_size, _ = self.portfolio_limits.scale_position_for_limits(
                            sim_portfolio_state,
                            symbol,
                            size,
                            entry_price,
                            signal,
                        )
                        if adjusted_size <= 0:
                            continue
                        
                        # Check max positions
                        within_max, _ = self.portfolio_limits.check_max_positions(
                            sim_portfolio_state,
                            symbol,
                        )
                        if not within_max:
                            continue
                        
                        # Calculate take-profit
                        take_profit = None
                        if self.config.strategy.funding_opportunity.exit.take_profit_rr and stop_loss:
                            stop_distance = abs(entry_price - stop_loss)
                            if signal == "long":
                                take_profit = entry_price + (stop_distance * self.config.strategy.funding_opportunity.exit.take_profit_rr)
                            else:
                                take_profit = entry_price - (stop_distance * self.config.strategy.funding_opportunity.exit.take_profit_rr)
                        
                        # Open funding position
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
                            "highest_price": entry_price if signal == "long" else None,
                            "lowest_price": entry_price if signal == "short" else None,
                            "source": "funding_opportunity",
                            "metadata": signal_dict.get("metadata", {}),
                        }
                        
                        current_equity -= abs(adjusted_size) * entry_price * taker_fee
                        total_funding_exposure += notional
                
                # Process confluence positions (simplified - would need full confluence logic)
                # For now, confluence positions are handled by main strategy with source='confluence'
                
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
                'reason': 'end_of_backtest',
                'source': pos.get('source', 'main_strategy'),  # Preserve source
                'metadata': pos.get('metadata', {}),  # Preserve metadata
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
            'timestamps': timestamps,  # Add timestamps for time-based calculations
        }
        
        # Calculate funding-specific metrics
        if self.config.strategy.funding_opportunity.enabled:
            funding_metrics = self._calculate_funding_metrics(trades, timestamps, initial_capital)
            result['funding_metrics'] = funding_metrics
    
    def _calculate_funding_metrics(
        self,
        trades: List[Dict],
        timestamps: List,
        initial_capital: float
    ) -> Dict:
        """Calculate funding-specific metrics from trades."""
        funding_trades = [
            t for t in trades
            if t.get('source') in ['funding_opportunity', 'confluence', 'confluence_prefer_funding', 'confluence_prefer_main']
        ]
        
        metrics = {
            'total_funding_trades': len(funding_trades),
            'funding_trades_per_year': 0.0,
            'holding_times_hours': [],
            'entry_funding_rates': {'long': [], 'short': []},
            'max_concurrent_funding_positions': 0,  # Would need position history
            'max_funding_exposure_pct': 0.0,  # Would need position history
        }
        
        if funding_trades and timestamps:
            # Calculate holding times
            for trade in funding_trades:
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')
                if entry_time and exit_time:
                    if isinstance(entry_time, str):
                        entry_time = pd.to_datetime(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = pd.to_datetime(exit_time)
                    hours = (exit_time - entry_time).total_seconds() / 3600
                    metrics['holding_times_hours'].append(hours)
                
                # Extract entry funding rate from metadata
                metadata = trade.get('metadata', {})
                funding_rate = metadata.get('funding_rate')
                if funding_rate is not None:
                    signal = trade.get('side', 'long')
                    metrics['entry_funding_rates'][signal].append(funding_rate)
            
            # Calculate trades per year
            first_trade = min(t.get('entry_time', timestamps[0]) for t in funding_trades if t.get('entry_time'))
            last_trade = max(t.get('exit_time', timestamps[-1]) for t in funding_trades if t.get('exit_time'))
            if isinstance(first_trade, str):
                first_trade = pd.to_datetime(first_trade)
            if isinstance(last_trade, str):
                last_trade = pd.to_datetime(last_trade)
            years = (last_trade - first_trade).total_seconds() / (365.25 * 24 * 3600)
            if years > 0:
                metrics['funding_trades_per_year'] = len(funding_trades) / years
            
            # Calculate holding time statistics
            if metrics['holding_times_hours']:
                holding_times = sorted(metrics['holding_times_hours'])
                metrics['holding_time_min'] = holding_times[0]
                metrics['holding_time_median'] = holding_times[len(holding_times) // 2]
                metrics['holding_time_90th_pct'] = holding_times[int(len(holding_times) * 0.9)] if len(holding_times) > 10 else holding_times[-1]
                metrics['holding_time_max'] = holding_times[-1]
            
            # Calculate average entry funding rates
            if metrics['entry_funding_rates']['long']:
                metrics['avg_entry_funding_rate_long'] = np.mean(metrics['entry_funding_rates']['long'])
            if metrics['entry_funding_rates']['short']:
                metrics['avg_entry_funding_rate_short'] = np.mean(metrics['entry_funding_rates']['short'])
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        return drawdown.min()

