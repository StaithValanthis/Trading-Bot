"""Safety checks and defensive behaviors for funding opportunity strategy."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..config import BotConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class FundingSafetyChecker:
    """Safety checks and defensive behaviors for funding strategy."""
    
    def __init__(self, config: BotConfig):
        """Initialize safety checker."""
        self.config = config
        self.funding_pnl_window = []  # Track recent funding P&L
        self.funding_pnl_window_days = 90  # 90-day trailing window
        
    def check_funding_data_availability(
        self,
        symbol: str,
        funding_rate: Optional[float],
        error: Optional[Exception] = None
    ) -> Tuple[bool, str]:
        """
        Check if funding data is available and valid.
        
        Returns:
            (is_safe: bool, message: str)
        """
        if error is not None:
            logger.warning(f"Funding data error for {symbol}: {error}")
            return False, f"funding_data_error: {str(error)}"
        
        if funding_rate is None:
            logger.warning(f"No funding data available for {symbol}")
            return False, "no_funding_data"
        
        # Check if funding rate is within reasonable bounds
        if abs(funding_rate) > self.config.strategy.funding_opportunity.risk.max_funding_rate:
            logger.warning(
                f"Funding rate {funding_rate*100:.4f}% exceeds max {self.config.strategy.funding_opportunity.risk.max_funding_rate*100:.4f}% for {symbol}"
            )
            return False, f"funding_rate_exceeds_max: {funding_rate}"
        
        return True, "ok"
    
    def check_total_funding_exposure(
        self,
        current_exposure: float,
        new_position_notional: float,
        equity: float
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Check if adding a new funding position would exceed exposure limits.
        
        Returns:
            (is_safe: bool, message: str, max_allowed_notional: Optional[float])
        """
        max_exposure = equity * self.config.strategy.funding_opportunity.risk.max_total_funding_exposure
        total_after = current_exposure + new_position_notional
        
        if total_after > max_exposure:
            max_allowed = max_exposure - current_exposure
            if max_allowed <= 0:
                return False, "max_funding_exposure_reached", None
            return False, "would_exceed_max_funding_exposure", max_allowed
        
        return True, "ok", None
    
    def check_funding_pnl_trailing_window(
        self,
        recent_funding_pnl: List[float],
        window_days: int = 90
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Check if recent funding P&L is strongly negative.
        
        Returns:
            (is_safe: bool, message: str, suggested_size_reduction: Optional[float])
        """
        if len(recent_funding_pnl) < 10:  # Need some data
            return True, "insufficient_data", None
        
        # Calculate trailing window P&L
        window_pnl = sum(recent_funding_pnl[-window_days:]) if len(recent_funding_pnl) >= window_days else sum(recent_funding_pnl)
        
        # If strongly negative (e.g., > 5% of equity lost), suggest scaling down
        # This is a placeholder - would need equity context
        if window_pnl < -0.05:  # -5% threshold
            reduction_factor = 0.5  # Reduce to 50% of base size
            logger.warning(
                f"Recent funding P&L is strongly negative ({window_pnl*100:.2f}%). "
                f"Consider scaling down base_size_fraction to {reduction_factor*100:.0f}% of current value."
            )
            return False, "negative_funding_pnl_window", reduction_factor
        
        return True, "ok", None
    
    def validate_position_source_tracking(
        self,
        positions: Dict[str, Dict]
    ) -> List[str]:
        """
        Validate that all positions have proper source tracking.
        
        Returns:
            List of warnings/errors
        """
        issues = []
        
        for symbol, pos in positions.items():
            source = pos.get('source')
            if not source:
                issues.append(f"Position {symbol} missing 'source' field")
            elif source not in ['main_strategy', 'funding_opportunity', 'confluence', 'confluence_prefer_funding', 'confluence_prefer_main']:
                issues.append(f"Position {symbol} has invalid source: {source}")
        
        return issues
    
    def check_leverage_breaches(
        self,
        portfolio_notional: float,
        equity: float,
        max_leverage: float
    ) -> Tuple[bool, str]:
        """
        Check if total portfolio leverage exceeds limits.
        
        Returns:
            (is_safe: bool, message: str)
        """
        current_leverage = portfolio_notional / equity if equity > 0 else 0
        
        if current_leverage > max_leverage:
            return False, f"leverage_breach: {current_leverage:.2f}x > {max_leverage:.2f}x"
        
        return True, "ok"
    
    def log_safety_summary(
        self,
        funding_positions: int,
        total_funding_exposure: float,
        equity: float,
        max_concurrent: int = 0,
        exposure_breaches: int = 0
    ):
        """Log safety summary."""
        exposure_pct = (total_funding_exposure / equity * 100) if equity > 0 else 0
        max_exposure_pct = self.config.strategy.funding_opportunity.risk.max_total_funding_exposure * 100
        
        logger.info("=" * 60)
        logger.info("FUNDING STRATEGY SAFETY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Current Funding Positions: {funding_positions}")
        logger.info(f"Max Concurrent (this run): {max_concurrent}")
        logger.info(f"Total Funding Exposure: ${total_funding_exposure:,.2f} ({exposure_pct:.1f}% of equity)")
        logger.info(f"Max Allowed Exposure: {max_exposure_pct:.1f}% of equity")
        logger.info(f"Exposure Breaches: {exposure_breaches}")
        
        if exposure_breaches > 0:
            logger.warning(f"⚠️ {exposure_breaches} exposure limit breach(es) detected!")
        else:
            logger.info("✅ No exposure breaches detected")
        
        logger.info("=" * 60)

