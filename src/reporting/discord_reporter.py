"""Discord reporting via webhook."""

import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import sqlite3

from ..config import ReportingConfig, BotConfig
from ..state.portfolio import PortfolioState
from ..exchange.bybit_client import BybitClient
from ..universe.selector import UniverseSelector
from ..universe.store import UniverseStore
from ..data.ohlcv_store import OHLCVStore
from ..data.trades_store import TradesStore
from ..logging_utils import get_logger

logger = get_logger(__name__)


class DiscordReporter:
    """Send daily reports to Discord via webhook."""
    
    def __init__(self, config: ReportingConfig, exchange_client: BybitClient):
        """
        Initialize Discord reporter.
        
        Args:
            config: Reporting configuration
            exchange_client: Exchange client
        """
        self.config = config
        self.exchange = exchange_client
        self.logger = get_logger(__name__)
        self.webhook_url = config.discord_webhook_url
    
    def send_daily_report(
        self,
        portfolio_state: PortfolioState,
        db_path: str,
        config: BotConfig = None
    ) -> bool:
        """
        Send daily performance and risk report to Discord.
        
        Args:
            portfolio_state: Current portfolio state
            db_path: Path to database for historical data
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.webhook_url:
            self.logger.warning("Discord webhook URL not configured, skipping report")
            return False
        
        try:
            # Calculate PnL metrics
            pnl_metrics = self._calculate_pnl_metrics(portfolio_state, db_path)
            
            # Get open positions
            positions = self._format_positions(portfolio_state.positions)
            
            # Get risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_state)
            
            # Get strategy performance (from trades in DB)
            strategy_performance = self._get_strategy_performance(db_path)
            
            # Get optimizer changes
            optimizer_changes = self._get_optimizer_changes(db_path)
            
            # Get universe stats (if config provided)
            universe_stats = None
            universe_changes = None
            if config:
                try:
                    universe_store = UniverseStore(db_path)
                    ohlcv_store = OHLCVStore(db_path)
                    selector = UniverseSelector(config.universe, self.exchange, ohlcv_store, universe_store)
                    universe_stats = selector.get_universe_stats()
                    
                    # Get recent universe changes (last 24h)
                    from datetime import timedelta
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=1)
                    changes = universe_store.get_changes(start_date, end_date)
                    if changes:
                        universe_changes = changes
                except Exception as e:
                    self.logger.warning(f"Error getting universe stats: {e}")
            
            # Get recent errors
            recent_errors = self._get_recent_errors()
            
            # Format and send message
            embed = self._create_embed(
                portfolio_state,
                pnl_metrics,
                positions,
                risk_metrics,
                strategy_performance,
                optimizer_changes,
                recent_errors,
                universe_stats,
                universe_changes
            )
            
            response = requests.post(
                self.webhook_url,
                json={'embeds': [embed]},
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info("Daily report sent to Discord successfully")
                return True
            else:
                self.logger.error(f"Error sending Discord report: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating Discord report: {e}")
            return False
    
    def _calculate_pnl_metrics(
        self,
        portfolio_state: PortfolioState,
        db_path: str
    ) -> Dict:
        """Calculate PnL metrics (daily, weekly, monthly) using trades table if available.

        Note: uses UTC day boundaries for consistency with risk logic.
        """
        now = datetime.now(tz=timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        current_equity = portfolio_state.equity

        trades_store = TradesStore(db_path)

        # Helper to sum PnL over a period
        def sum_pnl(start: datetime, end: datetime) -> float:
            trades = trades_store.get_trades_between(start, end)
            return float(sum(t["pnl"] for t in trades))

        daily_pnl = sum_pnl(today_start, now)
        weekly_pnl = sum_pnl(week_start, now)
        monthly_pnl = sum_pnl(month_start, now)

        # Percent returns relative to current equity (approximation)
        daily_pnl_pct = (daily_pnl / current_equity * 100) if current_equity > 0 else 0.0
        weekly_pnl_pct = (weekly_pnl / current_equity * 100) if current_equity > 0 else 0.0
        monthly_pnl_pct = (monthly_pnl / current_equity * 100) if current_equity > 0 else 0.0

        return {
            "current_equity": current_equity,
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "monthly_pnl": monthly_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "weekly_pnl_pct": weekly_pnl_pct,
            "monthly_pnl_pct": monthly_pnl_pct,
        }
    
    def _format_positions(self, positions: Dict) -> List[Dict]:
        """Format positions for display."""
        formatted = []
        
        for symbol, pos in positions.items():
            formatted.append({
                'symbol': symbol,
                'side': pos.get('side', 'unknown'),
                'size': pos.get('contracts', 0),
                'entry_price': pos.get('entry_price', 0),
                'mark_price': pos.get('mark_price', 0),
                'unrealized_pnl': pos.get('unrealized_pnl', 0),
                'unrealized_pnl_pct': (
                    (pos.get('unrealized_pnl', 0) / (pos.get('entry_price', 1) * pos.get('contracts', 1))) * 100
                    if pos.get('contracts', 0) != 0 else 0
                )
            })
        
        return formatted
    
    def _calculate_risk_metrics(self, portfolio_state: PortfolioState) -> Dict:
        """Calculate risk metrics."""
        total_notional = portfolio_state.get_total_notional()
        leverage = portfolio_state.get_leverage()

        # Estimate one-day funding PnL using current funding rates (approximate)
        estimated_funding_pnl_24h = 0.0
        for symbol, pos in portfolio_state.positions.items():
            notional = abs(pos.get('notional', 0))
            if notional <= 0:
                continue
            try:
                funding_info = self.exchange.fetch_funding_rate(symbol)
                funding_rate = float(funding_info.get('fundingRate', 0.0))
                # fundingRate is per 8h; approximate 24h PnL
                estimated_funding_pnl_24h += notional * funding_rate * 3.0
            except Exception as e:
                self.logger.warning(f"Error estimating funding for {symbol}: {e}")
                continue
        
        # Largest position concentration
        max_concentration = 0.0
        if portfolio_state.equity > 0:
            for pos in portfolio_state.positions.values():
                notional = abs(pos.get('notional', 0))
                concentration = (notional / portfolio_state.equity) * 100
                max_concentration = max(max_concentration, concentration)
        
        return {
            'max_leverage': leverage,
            'total_notional': total_notional,
            'largest_position_concentration_pct': max_concentration,
            'num_open_positions': len(portfolio_state.positions),
            'estimated_funding_pnl_24h': estimated_funding_pnl_24h,
        }
    
    def _get_strategy_performance(self, db_path: str) -> Dict:
        """Get strategy performance breakdown from trades table."""
        trades_store = TradesStore(db_path)
        # For now, consider all trades in DB
        start = datetime.fromtimestamp(0)
        end = datetime.now()
        trades = trades_store.get_trades_between(start, end)

        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "by_symbol": {},
            }

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        by_symbol: Dict[str, Dict[str, float]] = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"trades": 0, "pnl": 0.0}
            by_symbol[sym]["trades"] += 1
            by_symbol[sym]["pnl"] += t["pnl"]

        by_symbol = dict(sorted(by_symbol.items(), key=lambda kv: kv[1]["pnl"], reverse=True))

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "by_symbol": by_symbol,
        }
    
    def _get_optimizer_changes(self, db_path: str) -> Optional[Dict]:
        """Get recent optimizer parameter changes."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get latest optimization result
            cursor.execute("""
                SELECT timestamp, params_changed
                FROM optimization_results
                WHERE should_update = 1
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                timestamp, params_changed_json = result
                params_changed = json.loads(params_changed_json) if params_changed_json else {}
                
                return {
                    'timestamp': timestamp,
                    'params_changed': params_changed
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting optimizer changes: {e}")
            return None
    
    def _get_recent_errors(self) -> List[str]:
        """Get recent errors from logs or database."""
        # TODO: Query error log or database for recent errors
        return []
    
    def _create_embed(
        self,
        portfolio_state: PortfolioState,
        pnl_metrics: Dict,
        positions: List[Dict],
        risk_metrics: Dict,
        strategy_performance: Dict,
        optimizer_changes: Optional[Dict],
        recent_errors: List[str],
        universe_stats: Optional[Dict] = None,
        universe_changes: Optional[Dict] = None
    ) -> Dict:
        """Create Discord embed message."""
        now = datetime.now()
        
        # Title
        title = f"📊 Daily Trading Report - {now.strftime('%Y-%m-%d')}"
        
        # Description
        description = f"""
**Current Equity:** ${pnl_metrics['current_equity']:,.2f}

**PnL Summary:**
• Daily: ${pnl_metrics['daily_pnl']:,.2f} ({pnl_metrics['daily_pnl_pct']:+.2f}%)
• Weekly: ${pnl_metrics['weekly_pnl']:,.2f} ({pnl_metrics['weekly_pnl_pct']:+.2f}%)
• Monthly: ${pnl_metrics['monthly_pnl']:,.2f} ({pnl_metrics['monthly_pnl_pct']:+.2f}%)
"""
        
        # Fields
        fields = []
        
        # Open Positions
        if positions:
            pos_text = "\n".join([
                f"**{p['symbol']}** {p['side'].upper()}: {abs(p['size']):.4f} @ ${p['entry_price']:,.2f} "
                f"(PnL: ${p['unrealized_pnl']:,.2f} / {p['unrealized_pnl_pct']:+.2f}%)"
                for p in positions[:10]  # Limit to 10 positions
            ])
            fields.append({
                'name': '📈 Open Positions',
                'value': pos_text if pos_text else 'None',
                'inline': False
            })
        else:
            fields.append({
                'name': '📈 Open Positions',
                'value': 'None',
                'inline': False
            })
        
        # Risk Metrics
        risk_text = f"""
• Max Leverage: {risk_metrics['max_leverage']:.2f}x
• Total Notional: ${risk_metrics['total_notional']:,.2f}
• Largest Position: {risk_metrics['largest_position_concentration_pct']:.1f}%
• Open Positions: {risk_metrics['num_open_positions']}
• Est. Daily Funding PnL: ${risk_metrics.get('estimated_funding_pnl_24h', 0.0):,.2f}
"""
        fields.append({
            'name': '⚠️ Risk Metrics',
            'value': risk_text,
            'inline': False
        })
        
        # Strategy Performance
        if strategy_performance.get('total_trades', 0) > 0:
            perf_text = f"""
• Total Trades: {strategy_performance['total_trades']}
• Win Rate: {strategy_performance['win_rate']*100:.1f}%
• Profit Factor: {strategy_performance['profit_factor']:.2f}
"""
            fields.append({
                'name': '📊 Strategy Performance',
                'value': perf_text,
                'inline': False
            })
        
        # Universe Stats
        if universe_stats:
            universe_text = f"""
• Size: {universe_stats['size']} symbols
• Avg 24h Volume: ${universe_stats['avg_volume_24h']:,.2f} USDT
"""
            if universe_stats.get('top_5_by_volume'):
                top_5 = ", ".join([f"{item['symbol']}" for item in universe_stats['top_5_by_volume'][:5]])
                universe_text += f"• Top 5: {top_5}"
            
            fields.append({
                'name': '🌐 Trading Universe',
                'value': universe_text,
                'inline': False
            })
        
        # Universe Changes (last 24h)
        if universe_changes:
            additions = [s for s, c in universe_changes.items() if any(change['action'] == 'added' for change in c)]
            removals = [s for s, c in universe_changes.items() if any(change['action'] == 'removed' for change in c)]
            
            if additions or removals:
                changes_text = ""
                if additions:
                    changes_text += f"**Added**: {', '.join(additions[:10])}\n"
                if removals:
                    changes_text += f"**Removed**: {', '.join(removals[:10])}"
                
                if changes_text:
                    fields.append({
                        'name': '🔄 Universe Changes (24h)',
                        'value': changes_text,
                        'inline': False
                    })
        
        # Optimizer Changes
        if optimizer_changes:
            params_text = "\n".join([
                f"• **{param}**: {change['old']} → {change['new']}"
                for param, change in optimizer_changes['params_changed'].items()
            ])
            fields.append({
                'name': '🔧 Parameter Updates',
                'value': params_text,
                'inline': False
            })
        
        # Errors
        if recent_errors:
            error_text = "\n".join(recent_errors[:5])  # Limit to 5 errors
            fields.append({
                'name': '⚠️ Recent Errors',
                'value': error_text[:1024],  # Discord limit
                'inline': False
            })
        
        # Color (green if positive PnL, red if negative)
        color = 0x00ff00 if pnl_metrics['daily_pnl'] >= 0 else 0xff0000
        
        embed = {
            'title': title,
            'description': description,
            'fields': fields,
            'color': color,
            'timestamp': now.isoformat(),
            'footer': {
                'text': 'Bybit Trading Bot'
            }
        }
        
        return embed

