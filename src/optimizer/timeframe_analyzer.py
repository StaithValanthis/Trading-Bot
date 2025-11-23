"""
Timeframe comparison and analysis tool.

Systematically compares strategy performance across different timeframes
to determine optimal trading timeframe for the strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..config import BotConfig, TrendStrategyConfig, CrossSectionalStrategyConfig
from ..backtest.backtester import Backtester
from ..data.ohlcv_store import OHLCVStore
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TimeframeResult:
    """Results for a single timeframe backtest."""
    timeframe: str
    # Performance metrics
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_abs: float
    profit_factor: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_r_multiple: float
    # Operational metrics
    total_trades: int
    trades_per_day: float
    avg_holding_hours: float
    # Cost metrics
    total_fees_pct: float
    total_funding_pct: float
    net_return_after_costs_pct: float
    # Universe metrics
    avg_universe_size: float
    # Regime breakdown (if provided)
    bull_return: Optional[float] = None
    bear_return: Optional[float] = None
    chop_return: Optional[float] = None
    # Out-of-sample metrics (if IS/OOS split)
    oos_return_pct: Optional[float] = None
    oos_sharpe: Optional[float] = None
    oos_drawdown_pct: Optional[float] = None
    # Timeframe-specific metadata
    bars_per_day: int = field(default=24)  # e.g., 24 for 1h, 4 for 6h
    hours_per_bar: float = field(default=1.0)  # e.g., 1.0 for 1h, 4.0 for 4h


def parse_timeframe_to_hours(tf: str) -> float:
    """
    Parse timeframe string to hours.
    
    Examples:
        '15m' -> 0.25
        '30m' -> 0.5
        '1h' -> 1.0
        '2h' -> 2.0
        '4h' -> 4.0
        '6h' -> 6.0
        '1d' -> 24.0
    """
    tf_lower = tf.lower().strip()
    if tf_lower.endswith('m'):
        minutes = int(tf_lower[:-1])
        return minutes / 60.0
    elif tf_lower.endswith('h'):
        hours = int(tf_lower[:-1])
        return float(hours)
    elif tf_lower.endswith('d'):
        days = int(tf_lower[:-1])
        return float(days * 24)
    else:
        raise ValueError(f"Unknown timeframe format: {tf}")


def adjust_parameters_for_timeframe(
    base_config: BotConfig,
    target_timeframe: str
) -> BotConfig:
    """
    Adjust strategy parameters for a different timeframe.
    
    Two approaches:
    1. Time-normalized: Keep same time horizons (e.g., 48h momentum)
    2. Bar-normalized: Keep same number of bars (e.g., 24 bars momentum)
    
    Currently implements time-normalized approach for consistency.
    
    Args:
        base_config: Base configuration (assumed to be for 1h)
        target_timeframe: Target timeframe string (e.g., '4h')
    
    Returns:
        New BotConfig with adjusted parameters
    """
    base_tf_hours = parse_timeframe_to_hours(base_config.exchange.timeframe)
    target_tf_hours = parse_timeframe_to_hours(target_timeframe)
    ratio = base_tf_hours / target_tf_hours  # e.g., 1h -> 4h: ratio = 0.25
    
    # Create a copy of config
    import copy
    new_config = copy.deepcopy(base_config)
    new_config.exchange.timeframe = target_timeframe
    
    # Adjust trend parameters (time-normalized)
    # e.g., 24 bars * 1h = 24h momentum -> 6 bars * 4h = 24h momentum
    trend = new_config.strategy.trend
    trend.momentum_lookback = max(1, int(trend.momentum_lookback * ratio))
    
    # MA periods: keep same time horizon
    # e.g., 100 bars * 1h = 100h -> 25 bars * 4h = 100h
    trend.ma_short = max(5, int(trend.ma_short * ratio))
    trend.ma_long = max(20, int(trend.ma_long * ratio))
    
    # ATR period: keep same time horizon (14 bars * 1h = 14h -> 3.5 bars * 4h, round to 4)
    trend.atr_period = max(5, int(round(trend.atr_period * ratio)))
    
    # Cross-sectional parameters
    cross_sec = new_config.strategy.cross_sectional
    # Ranking window: keep same time horizon (e.g., 72h -> 18 bars * 4h)
    cross_sec.ranking_window = max(10, int(cross_sec.ranking_window * ratio))
    
    # Rebalance frequency: adjust in hours
    # e.g., 4h rebalance on 1h bars -> same 4h on 4h bars means 1 bar
    # But we keep it in hours for consistency
    # This will be handled in the backtester/rebalancing logic
    
    # Max holding hours: keep same (absolute time)
    # No adjustment needed
    
    logger.debug(
        f"Adjusted parameters for {target_timeframe}: "
        f"MA short: {trend.ma_short}, MA long: {trend.ma_long}, "
        f"Momentum: {trend.momentum_lookback}, ATR: {trend.atr_period}, "
        f"Ranking window: {cross_sec.ranking_window}"
    )
    
    return new_config


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation only)."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return 0.0
    return (excess_returns.mean() * 252) / downside_std


def classify_regime(price_data: pd.Series, window: int = 720) -> pd.Series:
    """
    Simple regime classification: bull, bear, or chop.
    
    Args:
        price_data: Series of prices
        window: Lookback window in bars
    
    Returns:
        Series of regime labels ('bull', 'bear', 'chop')
    """
    returns = price_data.pct_change(window)
    volatility = price_data.pct_change().rolling(window).std()
    
    # Bull: positive trend, low volatility relative to trend
    # Bear: negative trend
    # Chop: low trend magnitude relative to volatility
    trend_strength = returns.abs()
    vol_level = volatility * np.sqrt(window)
    
    regimes = pd.Series(index=price_data.index, dtype=object)
    
    for i in range(len(price_data)):
        if i < window:
            regimes.iloc[i] = 'chop'
            continue
        
        ret = returns.iloc[i]
        ts = trend_strength.iloc[i]
        vol = vol_level.iloc[i]
        
        if pd.isna(ret) or pd.isna(ts) or pd.isna(vol):
            regimes.iloc[i] = 'chop'
        elif ret > 0.05 and ts > vol * 0.5:  # Strong uptrend
            regimes.iloc[i] = 'bull'
        elif ret < -0.05 and ts > vol * 0.5:  # Strong downtrend
            regimes.iloc[i] = 'bear'
        else:
            regimes.iloc[i] = 'chop'
    
    return regimes


class TimeframeAnalyzer:
    """Analyze and compare strategy performance across different timeframes."""
    
    def __init__(self, config: BotConfig, ohlcv_store: OHLCVStore):
        """
        Initialize timeframe analyzer.
        
        Args:
            config: Base bot configuration (typically for 1h)
            ohlcv_store: OHLCV data store
        """
        self.config = config
        self.store = ohlcv_store
        self.logger = get_logger(__name__)
    
    def compare_timeframes(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_oos_split: float = 0.7,
        classify_regimes: bool = True
    ) -> List[TimeframeResult]:
        """
        Compare strategy performance across multiple timeframes.
        
        Args:
            symbols: List of symbols to test
            timeframes: List of timeframe strings (e.g., ['15m', '30m', '1h', '2h', '4h', '1d'])
            start_date: Start date for backtest (default: 1 year ago)
            end_date: End date for backtest (default: now)
            is_oos_split: Fraction for in-sample (default: 0.7 = 70% IS, 30% OOS)
            classify_regimes: If True, calculate regime-specific returns
        
        Returns:
            List of TimeframeResult objects, one per timeframe
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # 1 year default
        
        self.logger.info(
            f"Starting timeframe comparison for {len(symbols)} symbols, "
            f"{len(timeframes)} timeframes, from {start_date.date()} to {end_date.date()}"
        )
        
        results = []
        
        for tf in timeframes:
            try:
                self.logger.info(f"Testing timeframe: {tf}")
                
                # Adjust config for this timeframe
                tf_config = adjust_parameters_for_timeframe(self.config, tf)
                
                # Load data for all symbols at this timeframe
                since_ms = int(start_date.timestamp() * 1000)
                symbol_data = {}
                
                for symbol in symbols:
                    try:
                        df = self.store.get_ohlcv(symbol, tf, since=since_ms)
                        if not df.empty and len(df) > 100:  # Need enough bars
                            symbol_data[symbol] = df
                            self.logger.debug(f"Loaded {len(df)} {tf} bars for {symbol}")
                    except Exception as e:
                        self.logger.warning(f"Error loading {tf} data for {symbol}: {e}")
                        continue
                
                if not symbol_data:
                    self.logger.warning(f"No data available for timeframe {tf}, skipping")
                    continue
                
                # Run backtest
                backtester = Backtester(tf_config)
                
                # Filter symbol data to start_date and end_date if provided
                if start_date or end_date:
                    filtered_symbol_data = {}
                    for symbol, df in symbol_data.items():
                        filtered_df = df.copy()
                        if start_date:
                            filtered_df = filtered_df[filtered_df.index >= start_date]
                        if end_date:
                            filtered_df = filtered_df[filtered_df.index <= end_date]
                        if not filtered_df.empty and len(filtered_df) >= 100:
                            filtered_symbol_data[symbol] = filtered_df
                    symbol_data = filtered_symbol_data
                
                # Find common timestamp range
                all_timestamps = set()
                for df in symbol_data.values():
                    all_timestamps.update(df.index)
                common_timestamps = sorted(all_timestamps)
                
                if len(common_timestamps) < 100:
                    self.logger.warning(f"Insufficient common timestamps for {tf}, skipping")
                    continue
                
                # Run backtest (it uses all timestamps from symbol_data)
                backtest_result = backtester.backtest(symbol_data)
                
                if not backtest_result or backtest_result.get('total_trades', 0) == 0:
                    self.logger.warning(f"No trades for timeframe {tf}, skipping")
                    continue
                
                # Calculate additional metrics
                equity_history = backtest_result.get('equity_history', [])
                tf_result = self._calculate_metrics(
                    backtest_result,
                    tf,
                    symbol_data,
                    common_timestamps,
                    is_oos_split,
                    classify_regimes,
                    equity_history
                )
                
                results.append(tf_result)
                self.logger.info(
                    f"Completed {tf}: Return={tf_result.total_return_pct:.2f}%, "
                    f"Sharpe={tf_result.sharpe_ratio:.2f}, Trades={tf_result.total_trades}"
                )
                
            except Exception as e:
                self.logger.error(f"Error testing timeframe {tf}: {e}", exc_info=True)
                continue
        
        # Sort by Sharpe ratio (descending)
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        return results
    
    def _calculate_metrics(
        self,
        backtest_result: Dict,
        timeframe: str,
        symbol_data: Dict[str, pd.DataFrame],
        timestamps: List[datetime],
        is_oos_split: float,
        classify_regimes: bool,
        equity_history: List[float]
    ) -> TimeframeResult:
        """Calculate comprehensive metrics from backtest result."""
        hours_per_bar = parse_timeframe_to_hours(timeframe)
        bars_per_day = int(24 / hours_per_bar)
        
        # Extract equity curve
        equity_curve = pd.Series(equity_history)
        if len(equity_curve) == 0:
            equity_curve = pd.Series([backtest_result.get('initial_equity', 1000)])
        
        # Calculate returns
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            total_return_pct = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
            
            # Annualized return
            days_elapsed = (timestamps[-1] - timestamps[0]).days
            if days_elapsed > 0:
                annualized_return_pct = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.0 / days_elapsed) - 1) * 100
            else:
                annualized_return_pct = total_return_pct
            
            # Sharpe ratio (assuming daily returns)
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * np.sqrt(bars_per_day * 365)) / (returns.std() * np.sqrt(bars_per_day * 365))
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio
            sortino_ratio = calculate_sortino_ratio(returns)
            
            # Max drawdown
            cumulative = equity_curve / equity_curve.iloc[0]
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown_pct = drawdown.min()
            max_drawdown_abs = (equity_curve.iloc[drawdown.idxmin()] - running_max.iloc[drawdown.idxmin()]) if len(equity_curve) > 0 else 0.0
            
        else:
            total_return_pct = 0.0
            annualized_return_pct = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown_pct = 0.0
            max_drawdown_abs = 0.0
            returns = pd.Series()
        
        # Trade metrics
        trades = backtest_result.get('trades', [])
        total_trades = len(trades)
        
        if total_trades > 0:
            trade_returns = [t.get('return_pct', 0) for t in trades]
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r < 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
            avg_win_pct = np.mean(wins) if wins else 0.0
            avg_loss_pct = np.mean(losses) if losses else 0.0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0.0
            total_losses = abs(sum(losses)) if losses else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else (total_wins if total_wins > 0 else 0.0)
            
            # Average R multiple (simplified as return / risk)
            avg_r_multiple = np.mean([abs(r / avg_loss_pct) if avg_loss_pct < 0 else 0.0 for r in trade_returns]) if avg_loss_pct < 0 else 0.0
            
            # Average holding period
            holding_periods = []
            for trade in trades:
                entry = trade.get('entry_time')
                exit_t = trade.get('exit_time')
                if entry and exit_t:
                    if isinstance(entry, str):
                        entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
                    if isinstance(exit_t, str):
                        exit_t = datetime.fromisoformat(exit_t.replace('Z', '+00:00'))
                    if isinstance(entry, datetime) and isinstance(exit_t, datetime):
                        hours = (exit_t - entry).total_seconds() / 3600
                        holding_periods.append(hours)
            
            avg_holding_hours = np.mean(holding_periods) if holding_periods else 0.0
            trades_per_day = total_trades / ((timestamps[-1] - timestamps[0]).days or 1)
        else:
            win_rate = 0.0
            avg_win_pct = 0.0
            avg_loss_pct = 0.0
            profit_factor = 0.0
            avg_r_multiple = 0.0
            avg_holding_hours = 0.0
            trades_per_day = 0.0
        
        # Cost metrics
        total_fees_pct = backtest_result.get('total_fees', 0.0) / backtest_result.get('initial_equity', 1.0) * 100 if backtest_result.get('initial_equity', 0) > 0 else 0.0
        total_funding_pct = backtest_result.get('total_funding_pnl', 0.0) / backtest_result.get('initial_equity', 1.0) * 100 if backtest_result.get('initial_equity', 0) > 0 else 0.0
        net_return_after_costs_pct = total_return_pct - total_fees_pct + total_funding_pct
        
        # Average universe size (simplified - use number of symbols)
        avg_universe_size = float(len(symbol_data))
        
        # IS/OOS split
        if len(equity_curve) > 10 and is_oos_split < 1.0:
            split_idx = int(len(equity_curve) * is_oos_split)
            is_curve = equity_curve.iloc[:split_idx]
            oos_curve = equity_curve.iloc[split_idx:]
            
            if len(oos_curve) > 1:
                oos_return_pct = ((oos_curve.iloc[-1] / oos_curve.iloc[0]) - 1) * 100
                oos_returns = oos_curve.pct_change().dropna()
                if len(oos_returns) > 0 and oos_returns.std() > 0:
                    oos_sharpe = (oos_returns.mean() * np.sqrt(bars_per_day * 365)) / (oos_returns.std() * np.sqrt(bars_per_day * 365))
                else:
                    oos_sharpe = 0.0
                
                oos_cumulative = oos_curve / oos_curve.iloc[0]
                oos_running_max = oos_cumulative.cummax()
                oos_drawdown = (oos_cumulative - oos_running_max) / oos_running_max * 100
                oos_drawdown_pct = oos_drawdown.min()
            else:
                oos_return_pct = None
                oos_sharpe = None
                oos_drawdown_pct = None
        else:
            oos_return_pct = None
            oos_sharpe = None
            oos_drawdown_pct = None
        
        # Regime classification (simplified - using first symbol as proxy)
        bull_return = None
        bear_return = None
        chop_return = None
        
        if classify_regimes and symbol_data:
            first_symbol = list(symbol_data.keys())[0]
            price_data = symbol_data[first_symbol]['close']
            regimes = classify_regime(price_data, window=min(720, len(price_data) // 10))
            
            # Calculate returns per regime (simplified - using equity curve alignment)
            # This is approximate since regimes are based on price, not equity
            # For a more accurate analysis, we'd need to match trades to regimes
            bull_return = None  # Placeholder - would require trade-level regime matching
            bear_return = None
            chop_return = None
        
        return TimeframeResult(
            timeframe=timeframe,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_abs=max_drawdown_abs,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            avg_r_multiple=avg_r_multiple,
            total_trades=total_trades,
            trades_per_day=trades_per_day,
            avg_holding_hours=avg_holding_hours,
            total_fees_pct=total_fees_pct,
            total_funding_pct=total_funding_pct,
            net_return_after_costs_pct=net_return_after_costs_pct,
            avg_universe_size=avg_universe_size,
            bull_return=bull_return,
            bear_return=bear_return,
            chop_return=chop_return,
            oos_return_pct=oos_return_pct,
            oos_sharpe=oos_sharpe,
            oos_drawdown_pct=oos_drawdown_pct,
            bars_per_day=bars_per_day,
            hours_per_bar=hours_per_bar
        )
    
    def save_results(
        self,
        results: List[TimeframeResult],
        output_path: str
    ):
        """Save timeframe comparison results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = []
        for r in results:
            result_dict = {
                'timeframe': r.timeframe,
                'total_return_pct': r.total_return_pct,
                'annualized_return_pct': r.annualized_return_pct,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'max_drawdown_pct': r.max_drawdown_pct,
                'profit_factor': r.profit_factor,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'trades_per_day': r.trades_per_day,
                'avg_holding_hours': r.avg_holding_hours,
                'total_fees_pct': r.total_fees_pct,
                'net_return_after_costs_pct': r.net_return_after_costs_pct,
                'oos_return_pct': r.oos_return_pct,
                'oos_sharpe': r.oos_sharpe,
                'bars_per_day': r.bars_per_day,
                'hours_per_bar': r.hours_per_bar
            }
            results_dict.append(result_dict)
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved timeframe comparison results to {output_file}")

