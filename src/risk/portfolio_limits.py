"""Portfolio-level risk limits and checks."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from ..config import RiskConfig
from ..exchange.bybit_client import BybitClient
from ..state.portfolio import PortfolioState
from ..logging_utils import get_logger

logger = get_logger(__name__)


class PortfolioLimits:
    """Enforce portfolio-level risk limits."""
    
    def __init__(self, config: RiskConfig, exchange_client: BybitClient):
        """
        Initialize portfolio limits checker.
        
        Args:
            config: Risk configuration
            exchange_client: Exchange client
        """
        self.config = config
        self.exchange = exchange_client
        self.logger = get_logger(__name__)
        self.daily_start_equity: Optional[float] = None
        self.daily_start_date: Optional[str] = None
    
    def update_daily_start(self, equity: float):
        """Update daily start equity (call at start of each UTC day)."""
        today = datetime.now(timezone.utc).date().isoformat()
        
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_equity = equity
            self.logger.info(f"Daily start equity updated: {equity:.2f}")
    
    def check_daily_loss_limits(
        self,
        current_equity: float,
        realized_pnl: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check if daily loss limits are breached.
        
        Args:
            current_equity: Current account equity
            realized_pnl: Realized PnL for today
        
        Returns:
            Tuple of (can_trade, reason)
        """
        if self.daily_start_equity is None:
            self.update_daily_start(current_equity)
            return True, ""
        
        # Calculate daily PnL
        daily_pnl = current_equity - self.daily_start_equity + realized_pnl
        daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        # Hard loss cap
        if daily_pnl_pct <= self.config.daily_hard_loss_pct:
            reason = f"Hard daily loss cap breached: {daily_pnl_pct:.2f}% <= {self.config.daily_hard_loss_pct}%"
            self.logger.warning(reason)
            return False, reason
        
        # Soft loss cap
        if daily_pnl_pct <= self.config.daily_soft_loss_pct:
            reason = f"Soft daily loss cap reached: {daily_pnl_pct:.2f}% <= {self.config.daily_soft_loss_pct}%"
            self.logger.warning(reason)
            # Still allow trading but at reduced risk
            return True, reason
        
        return True, ""
    
    def check_leverage_limit(
        self,
        portfolio_state: PortfolioState,
        new_position_notional: float,
        new_position_side: str  # 'long' or 'short'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a new position would exceed max leverage.
        
        Args:
            portfolio_state: Current portfolio state
            new_position_notional: Notional value of new position
            new_position_side: Side of new position
        
        Returns:
            Tuple of (within_limit, error_message)
        """
        current_notional = 0.0
        
        # Sum absolute notional of existing positions
        for pos in portfolio_state.positions.values():
            notional = abs(pos.get('notional', 0))
            current_notional += notional
        
        # Add new position notional
        total_notional = current_notional + new_position_notional
        
        # Calculate leverage
        equity = portfolio_state.equity
        if equity <= 0:
            return False, "Equity must be positive"
        
        leverage = total_notional / equity
        
        if leverage > self.config.max_leverage:
            return False, f"Leverage {leverage:.2f}x exceeds max {self.config.max_leverage}x"
        
        return True, None
    
    def check_symbol_concentration(
        self,
        portfolio_state: PortfolioState,
        symbol: str,
        new_position_notional: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position in symbol exceeds max symbol fraction.
        
        Args:
            portfolio_state: Current portfolio state
            symbol: Symbol name
            new_position_notional: Notional value of new position
        
        Returns:
            Tuple of (within_limit, error_message)
        """
        # Calculate total notional for this symbol including new position
        existing_position = portfolio_state.positions.get(symbol, {})
        existing_notional = abs(existing_position.get('notional', 0))
        
        # Assume new position replaces existing or adds to it
        # For simplicity, take max of existing or new
        total_symbol_notional = max(existing_notional, new_position_notional)
        
        equity = portfolio_state.equity
        if equity <= 0:
            return False, "Equity must be positive"
        
        symbol_fraction = total_symbol_notional / equity
        
        if symbol_fraction > self.config.max_symbol_fraction:
            return False, (
                f"Symbol {symbol} concentration {symbol_fraction*100:.1f}% "
                f"exceeds max {self.config.max_symbol_fraction*100:.1f}%"
            )
        
        return True, None
    
    def check_max_positions(
        self,
        portfolio_state: PortfolioState,
        new_symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a new position would exceed max positions.
        
        Args:
            portfolio_state: Current portfolio state
            new_symbol: Symbol of new position
        
        Returns:
            Tuple of (within_limit, error_message)
        """
        # Count distinct symbols with open positions
        open_symbols = set()
        for symbol, pos in portfolio_state.positions.items():
            contracts = abs(pos.get('contracts', 0))
            if contracts > 0:
                open_symbols.add(symbol)
        
        # If new symbol is already in portfolio, no new position count
        if new_symbol not in open_symbols:
            if len(open_symbols) >= self.config.max_positions:
                return False, (
                    f"Max positions {self.config.max_positions} reached. "
                    f"Open symbols: {open_symbols}"
                )
        
        return True, None
    
    def scale_position_for_limits(
        self,
        portfolio_state: PortfolioState,
        symbol: str,
        desired_size: float,
        entry_price: float,
        signal: str
    ) -> Tuple[float, str]:
        """
        Scale down position size to fit within risk limits if needed.
        
        Args:
            portfolio_state: Current portfolio state
            symbol: Symbol name
            desired_size: Desired position size in contracts
            entry_price: Entry price
            signal: Signal direction
        
        Returns:
            Tuple of (adjusted_size, reason)
        """
        if desired_size == 0:
            return 0.0, ""
        
        # Calculate notional
        market_info = self.exchange.get_market_info(symbol)
        contract_size = market_info.get('contractSize', 1.0)
        notional = desired_size * entry_price * contract_size
        
        # Check leverage limit
        within_leverage, leverage_error = self.check_leverage_limit(
            portfolio_state,
            notional,
            signal
        )
        
        if not within_leverage:
            # Scale down to fit within leverage
            max_notional = portfolio_state.equity * self.config.max_leverage
            
            # Calculate current notional
            current_notional = 0.0
            for pos in portfolio_state.positions.values():
                current_notional += abs(pos.get('notional', 0))
            
            # Available notional
            available_notional = max(0, max_notional - current_notional)
            
            # Scale down size
            if available_notional > 0:
                scaled_notional = min(notional, available_notional)
                scaled_size = scaled_notional / (entry_price * contract_size)
                scaled_size = self.exchange.round_amount(symbol, scaled_size)
                return scaled_size, f"Scaled for leverage: {leverage_error}"
            else:
                return 0.0, f"Cannot add position: {leverage_error}"
        
        # Check symbol concentration
        within_concentration, conc_error = self.check_symbol_concentration(
            portfolio_state,
            symbol,
            notional
        )
        
        if not within_concentration:
            # Scale down to fit within symbol fraction
            max_symbol_notional = portfolio_state.equity * self.config.max_symbol_fraction
            existing_position = portfolio_state.positions.get(symbol, {})
            existing_notional = abs(existing_position.get('notional', 0))
            available_symbol_notional = max(0, max_symbol_notional - existing_notional)
            
            if available_symbol_notional > 0:
                scaled_notional = min(notional, available_symbol_notional)
                scaled_size = scaled_notional / (entry_price * contract_size)
                scaled_size = self.exchange.round_amount(symbol, scaled_size)
                return scaled_size, f"Scaled for concentration: {conc_error}"
            else:
                return 0.0, f"Cannot add position: {conc_error}"
        
        # Approximate liquidation-distance check based on effective leverage.
        # We approximate liquidation distance as 100 / effective_leverage (% move to wipe equity),
        # which is conservative and does not use exact exchange formulas.
        total_notional = 0.0
        for pos in portfolio_state.positions.values():
            total_notional += abs(pos.get('notional', 0))
        total_notional += notional

        equity = portfolio_state.equity
        if equity > 0 and total_notional > 0:
            effective_leverage = total_notional / equity
            if effective_leverage > 0:
                liq_distance_pct = 100.0 / effective_leverage
                min_liq = getattr(self.config, "min_liquidation_distance_pct", 0.0)
                if min_liq > 0 and liq_distance_pct < min_liq:
                    # Scale down notional so that 100 / leverage >= min_liq
                    max_leverage_for_liq = 100.0 / min_liq
                    max_total_notional = max_leverage_for_liq * equity
                    available_notional_for_liq = max(0.0, max_total_notional - (total_notional - notional))
                    if available_notional_for_liq <= 0:
                        return 0.0, (
                            f"Cannot add position: effective liquidation distance {liq_distance_pct:.1f}% "
                            f"below minimum {min_liq:.1f}%"
                        )
                    scaled_notional = min(notional, available_notional_for_liq)
                    scaled_size = scaled_notional / (entry_price * contract_size)
                    scaled_size = self.exchange.round_amount(symbol, scaled_size)
                    return scaled_size, (
                        f"Scaled for liquidation buffer: "
                        f"liq_distance {liq_distance_pct:.1f}% < min {min_liq:.1f}%"
                    )

        return desired_size, ""

