"""Funding rate opportunity finder - generates signals based on funding rates."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..config import FundingOpportunityConfig
from ..exchange.bybit_client import BybitClient
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FundingOpportunity:
    """Represents a funding rate opportunity."""
    symbol: str
    funding_rate: float  # Per 8h
    signal: str  # 'long' or 'short'
    expected_funding_per_day: float  # funding_rate * 3 (3 funding periods per day)
    confidence: float  # 0-1, based on funding magnitude and trend alignment
    entry_price: float
    stop_loss: Optional[float]
    metadata: Dict


class FundingOpportunityGenerator:
    """
    Generate signals based on funding rate opportunities.
    
    This is an INDEPENDENT strategy that:
    - Uses its own universe (separate from main strategy)
    - Operates autonomously
    - Can detect confluence with main strategy when both agree on symbols
    """
    
    def __init__(self, config: FundingOpportunityConfig, exchange: BybitClient):
        """
        Initialize funding opportunity generator.
        
        Args:
            config: Funding opportunity configuration
            exchange: Exchange client for fetching funding rates
        """
        self.config = config
        self.exchange = exchange
        self.logger = get_logger(__name__)
    
    def detect_confluence(
        self,
        funding_symbols: List[str],
        funding_signals: Dict[str, Dict],
        main_symbols: List[str],
        main_signals: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Detect confluence between funding strategy and main strategy.
        
        Args:
            funding_symbols: Symbols selected by funding strategy
            funding_signals: Funding strategy signals
            main_symbols: Symbols selected by main strategy
            main_signals: Main strategy signals
        
        Returns:
            Dictionary mapping symbol to confluence info:
            {
                'symbol': {
                    'main_signal': 'long',
                    'funding_signal': 'long',
                    'aligned': True,
                    'confluence_type': 'both_long' or 'both_short'
                }
            }
        """
        confluence = {}
        overlap_symbols = set(funding_symbols) & set(main_symbols)
        
        for symbol in overlap_symbols:
            funding_sig = funding_signals.get(symbol, {}).get('signal', 'flat')
            main_sig = main_signals.get(symbol, {}).get('signal', 'flat')
            
            # Check if signals are aligned (same direction)
            aligned = (funding_sig == main_sig and funding_sig != 'flat')
            
            if aligned:
                confluence[symbol] = {
                    'main_signal': main_sig,
                    'funding_signal': funding_sig,
                    'aligned': True,
                    'confluence_type': f'both_{main_sig}'
                }
            else:
                # Symbols overlap but signals don't align
                confluence[symbol] = {
                    'main_signal': main_sig,
                    'funding_signal': funding_sig,
                    'aligned': False,
                    'confluence_type': 'misaligned'
                }
        
        return confluence
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Funding rate (per 8h), or None if error
        """
        try:
            funding_info = self.exchange.fetch_funding_rate(symbol)
            return funding_info.get('fundingRate', 0.0)
        except Exception as e:
            self.logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def scan_opportunities(
        self,
        symbols: List[str],
        symbol_data: Dict[str, pd.DataFrame] = None
    ) -> List[FundingOpportunity]:
        """
        Scan symbols for funding rate opportunities.
        
        Args:
            symbols: List of symbols to scan
            symbol_data: Optional price data for trend alignment checks
        
        Returns:
            List of FundingOpportunity objects, sorted by attractiveness (best first)
        """
        opportunities = []
        
        for symbol in symbols:
            funding_rate = self.get_funding_rate(symbol)
            
            if funding_rate is None:
                continue
            
            # Check minimum funding rate threshold
            abs_funding = abs(funding_rate)
            if abs_funding < self.config.min_funding_rate:
                continue
            
            # Check maximum funding rate (avoid extreme rates that may reverse)
            if abs_funding > self.config.risk.max_funding_rate:
                self.logger.debug(
                    f"Skipping {symbol}: funding rate {funding_rate:.6f} exceeds max "
                    f"{self.config.risk.max_funding_rate:.6f}"
                )
                continue
            
            # Determine signal direction based on funding rate
            # Negative funding = we receive funding to hold long
            # Positive funding = we receive funding to hold short
            if funding_rate < 0:
                signal = 'long'
                expected_funding = abs(funding_rate) * 3.0  # 3 funding periods per day
            elif funding_rate > 0:
                signal = 'short'
                expected_funding = abs(funding_rate) * 3.0
            else:
                continue  # Zero funding, skip
            
            # Get price data for entry/exit calculations
            price_data = symbol_data.get(symbol) if symbol_data else None
            entry_price = None
            stop_loss = None
            confidence = 0.0
            
            if price_data is not None and len(price_data) > 0:
                entry_price = float(price_data['close'].iloc[-1])
                
                # Calculate stop loss using ATR if available
                if len(price_data) >= 14:  # Need enough data for ATR
                    high = price_data['high']
                    low = price_data['low']
                    close = price_data['close']
                    
                    # Calculate ATR
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean().iloc[-1]
                    
                    if not pd.isna(atr) and atr > 0:
                        stop_distance = atr * self.config.exit.stop_loss_atr_multiplier
                        if signal == 'long':
                            stop_loss = entry_price - stop_distance
                        else:  # short
                            stop_loss = entry_price + stop_distance
                
                # Calculate confidence based on funding magnitude and trend alignment
                confidence = self._calculate_confidence(
                    funding_rate,
                    signal,
                    price_data
                )
            else:
                # No price data, use current price from exchange (would need to fetch)
                # For now, skip if no price data
                self.logger.debug(f"Skipping {symbol}: no price data available")
                continue
            
            if entry_price is None or entry_price <= 0:
                continue
            
            # Safety check: Skip if funding data error or rate exceeds max
            try:
                if abs(funding_rate) > self.config.risk.max_funding_rate:
                    self.logger.debug(f"Skipping {symbol}: funding rate {funding_rate*100:.4f}% exceeds max {self.config.risk.max_funding_rate*100:.4f}%")
                    continue
            except Exception as e:
                self.logger.warning(f"Error checking funding rate limits for {symbol}: {e}. Skipping.")
                continue
            
            opportunity = FundingOpportunity(
                symbol=symbol,
                funding_rate=funding_rate,
                signal=signal,
                expected_funding_per_day=expected_funding,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                metadata={
                    'abs_funding_rate': abs_funding,
                    'funding_rate_pct_per_day': expected_funding * 100,
                }
            )
            
            opportunities.append(opportunity)
        
        # Sort by confidence (highest first), then by funding rate magnitude
        opportunities.sort(
            key=lambda x: (x.confidence, abs(x.funding_rate)),
            reverse=True
        )
        
        return opportunities
    
    def _calculate_confidence(
        self,
        funding_rate: float,
        signal: str,
        price_data: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score (0-1) for an opportunity.
        
        Args:
            funding_rate: Current funding rate
            signal: Signal direction ('long' or 'short')
            price_data: Price history DataFrame
        
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from funding rate magnitude
        abs_funding = abs(funding_rate)
        funding_confidence = min(1.0, abs_funding / (self.config.min_funding_rate * 2))
        
        # Trend alignment confidence (if required)
        trend_confidence = 1.0
        if self.config.entry.require_trend_alignment and len(price_data) >= 20:
            close = price_data['close']
            # Simple momentum check
            momentum = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            
            if signal == 'long':
                # Prefer positive or neutral momentum for longs
                if momentum < -0.05:  # Strongly negative
                    trend_confidence = 0.3
                elif momentum < 0:
                    trend_confidence = 0.7
                else:
                    trend_confidence = 1.0
            else:  # short
                # Prefer negative or neutral momentum for shorts
                if momentum > 0.05:  # Strongly positive
                    trend_confidence = 0.3
                elif momentum > 0:
                    trend_confidence = 0.7
                else:
                    trend_confidence = 1.0
        
        # Combine confidences
        confidence = funding_confidence * 0.7 + trend_confidence * 0.3
        
        return confidence
    
    def generate_signal(
        self,
        symbol: str,
        funding_rate: float,
        price_data: pd.DataFrame = None
    ) -> Dict[str, any]:
        """
        Generate entry signal for a funding opportunity.
        
        Args:
            symbol: Trading pair symbol
            funding_rate: Current funding rate
            price_data: Optional price history
        
        Returns:
            Dictionary with signal, entry_price, stop_loss, confidence, etc.
        """
        # Determine signal direction
        if funding_rate < 0:
            signal = 'long'
        elif funding_rate > 0:
            signal = 'short'
        else:
            return {
                'signal': 'flat',
                'entry_price': None,
                'stop_loss': None,
                'confidence': 0.0,
                'metadata': {'reason': 'zero_funding_rate'}
            }
        
        entry_price = None
        stop_loss = None
        
        if price_data is not None and len(price_data) > 0:
            entry_price = float(price_data['close'].iloc[-1])
            
            # Calculate stop loss
            if len(price_data) >= 14:
                high = price_data['high']
                low = price_data['low']
                close = price_data['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]
                
                if not pd.isna(atr) and atr > 0:
                    stop_distance = atr * self.config.exit.stop_loss_atr_multiplier
                    if signal == 'long':
                        stop_loss = entry_price - stop_distance
                    else:
                        stop_loss = entry_price + stop_distance
        
        confidence = 0.0
        if price_data is not None:
            confidence = self._calculate_confidence(funding_rate, signal, price_data)
        
        return {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'metadata': {
                'funding_rate': funding_rate,
                'expected_funding_per_day': abs(funding_rate) * 3.0,
                'source': 'funding_opportunity'
            }
        }
    
    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_funding_rate: float,
        price_data: pd.DataFrame = None
    ) -> Tuple[bool, str]:
        """
        Determine if a funding position should be exited.
        
        Args:
            symbol: Trading pair symbol
            position: Position dictionary (from portfolio)
            current_funding_rate: Current funding rate
            price_data: Optional price history
        
        Returns:
            (should_exit: bool, reason: str)
        """
        position_signal = position.get('signal')
        entry_funding_rate = position.get('metadata', {}).get('funding_rate')
        
        if entry_funding_rate is None:
            # Can't determine exit without entry funding rate
            return False, "no_entry_funding_rate"
        
        # Check if funding rate flipped sign
        # For funding opportunities:
        # - LONG positions: we entered when funding was NEGATIVE (we receive funding)
        # - SHORT positions: we entered when funding was POSITIVE (we receive funding)
        # A flip occurs when the sign changes, meaning we'd now pay instead of receive
        if self.config.exit.exit_on_funding_flip:
            entry_funding_sign = 1 if entry_funding_rate > 0 else -1 if entry_funding_rate < 0 else 0
            current_funding_sign = 1 if current_funding_rate > 0 else -1 if current_funding_rate < 0 else 0
            
            # Exit if funding sign flipped (from negative to positive for longs, or positive to negative for shorts)
            if entry_funding_sign != 0 and current_funding_sign != 0 and entry_funding_sign != current_funding_sign:
                return True, f"funding_flipped_from_{entry_funding_rate:.6f}_to_{current_funding_rate:.6f}"
        
        # Check if funding rate dropped below threshold (with hysteresis buffer)
        abs_current_funding = abs(current_funding_rate)
        exit_threshold = self.config.exit.exit_funding_threshold
        # Add 20% buffer to prevent churn from small oscillations
        exit_threshold_with_buffer = exit_threshold * 0.8
        if abs_current_funding < exit_threshold_with_buffer:
            return True, f"funding_below_threshold_{abs_current_funding:.6f}"
        
        # Check maximum holding period
        if self.config.exit.max_holding_hours:
            entry_time = position.get('entry_time')
            if entry_time:
                from datetime import datetime, timezone
                # Handle both datetime objects and ISO strings
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    except:
                        return False, "invalid_entry_time_format"
                
                if isinstance(entry_time, datetime):
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    
                    hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                    if hours_held >= self.config.exit.max_holding_hours:
                        return True, f"max_holding_period_exceeded_{hours_held:.1f}h"
        
        # Check stop loss (handled by backtester/main loop)
        # Check take profit (handled by backtester/main loop)
        
        return False, "no_exit_condition"

