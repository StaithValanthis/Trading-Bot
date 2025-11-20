"""Main CLI entrypoint."""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List
import ccxt

from ..config import BotConfig
from ..logging_utils import setup_logging, get_logger
from ..exchange.bybit_client import BybitClient
from ..data.ohlcv_store import OHLCVStore
from ..data.trades_store import TradesStore
from ..data.orders_store import OrdersStore
from ..data.downloader import DataDownloader
from ..signals.trend import TrendSignalGenerator
from ..signals.cross_sectional import CrossSectionalSignalGenerator
from ..signals.funding_carry import FundingBiasGenerator
from ..risk.position_sizing import PositionSizer
from ..risk.portfolio_limits import PortfolioLimits
from ..execution.executor import OrderExecutor
from ..state.portfolio import PortfolioState
from ..backtest.backtester import Backtester
from ..optimizer.optimizer import Optimizer
from ..optimizer.universe_optimizer import UniverseOptimizer
from ..optimizer.timeframe_analyzer import TimeframeAnalyzer
from ..reporting.discord_reporter import DiscordReporter
from ..universe.selector import UniverseSelector
from ..universe.store import UniverseStore
import sqlite3


def run_universe_build(config_path: str):
    """Build/update the universe."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="universe",
        force=True,
    )
    
    # Now get logger (will inherit from root logger)
    logger = get_logger(__name__)
    logger.info("Building universe")
    
    # Initialize components
    exchange = BybitClient(config.exchange)
    ohlcv_store = OHLCVStore(config.data.db_path)
    universe_store = UniverseStore(config.data.db_path)
    selector = UniverseSelector(config.universe, exchange, ohlcv_store, universe_store)
    
    # Build universe
    universe, changes = selector.build_universe(config.exchange.timeframe)
    
    # Print results
    print("\n" + "="*60)
    print("UNIVERSE BUILD RESULTS")
    print("="*60)
    print(f"Universe size: {len(universe)} symbols")
    print(f"\nAdditions: {len([c for c in changes.values() if c['action'] == 'added'])}")
    print(f"Removals: {len([c for c in changes.values() if c['action'] == 'removed'])}")
    
    if changes:
        print("\nChanges:")
        for symbol, change in changes.items():
            print(f"  {symbol}: {change['action']} - {change['reason']}")
    
    print(f"\nUniverse symbols: {sorted(universe)}")
    print("="*60)


def run_universe_show(config_path: str):
    """Show current universe."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="universe",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    
    # Initialize components
    exchange = BybitClient(config.exchange)
    ohlcv_store = OHLCVStore(config.data.db_path)
    universe_store = UniverseStore(config.data.db_path)
    selector = UniverseSelector(config.universe, exchange, ohlcv_store, universe_store)
    
    # Get universe
    universe = selector.get_universe()
    stats = selector.get_universe_stats()
    
    # Print
    print("\n" + "="*60)
    print("CURRENT UNIVERSE")
    print("="*60)
    print(f"Size: {len(universe)} symbols")
    print(f"Average 24h volume: ${stats['avg_volume_24h']:,.2f} USDT")
    
    if stats['top_5_by_volume']:
        print("\nTop 5 by volume:")
        for item in stats['top_5_by_volume']:
            print(f"  {item['symbol']}: ${item['volume_24h']:,.2f} USDT")
    
    print(f"\nSymbols: {sorted(universe)}")
    print("="*60)


def run_universe_history(config_path: str, symbol: str = None):
    """Show universe history."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="universe",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    
    # Initialize components
    universe_store = UniverseStore(config.data.db_path)
    
    if symbol:
        # Get history for specific symbol
        history = universe_store.get_history(symbol)
        
        print(f"\nUniverse history for {symbol}:")
        print("="*60)
        for record in history:
            print(f"{record['date']}: {record['action']} - {record['reason']}")
            if record.get('volume_24h'):
                print(f"  24h volume: ${record['volume_24h']:,.2f} USDT")
        print("="*60)
    else:
        # Get recent changes (last 30 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        changes = universe_store.get_changes(start_date, end_date)
        
        print(f"\nUniverse changes (last 30 days):")
        print("="*60)
        for symbol_name, symbol_changes in sorted(changes.items()):
            print(f"\n{symbol_name}:")
            for change in symbol_changes:
                print(f"  {change['date']}: {change['action']} - {change['reason']}")
        print("="*60)


def run_live(config_path: str):
    """Run live trading bot."""
    # Load config FIRST (before any logging that depends on config)
    config = BotConfig.from_yaml(config_path)
    errors = config.validate()
    if errors:
        # Use basic logging for early errors
        import logging
        logging.error(f"Config validation errors: {errors}")
        sys.exit(1)
    
    # Setup logging with config (this configures root logger)
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="live",  # Separate log file for live service
        force=True,  # Force reinitialization
    )
    
    # Now get logger (will inherit from root logger)
    logger = get_logger(__name__)
    logger.info("Starting live trading bot")
    
    # Validate API credentials early
    if config.exchange.mode != "paper":
        # Debug: Check if env vars are loaded (without exposing values)
        import os
        env_key_raw = os.getenv("BYBIT_API_KEY", "")
        env_secret_raw = os.getenv("BYBIT_API_SECRET", "")
        env_key_present = bool(env_key_raw)
        env_secret_present = bool(env_secret_raw)
        config_key_present = bool(config.exchange.api_key)
        config_secret_present = bool(config.exchange.api_secret)
        
        # Show preview without exposing full keys
        env_key_preview = f"{env_key_raw[:3]}...{env_key_raw[-3:]}" if len(env_key_raw) >= 6 else "MISSING"
        config_key_preview = f"{config.exchange.api_key[:3]}...{config.exchange.api_key[-3:]}" if config.exchange.api_key and len(config.exchange.api_key) >= 6 else "MISSING"
        
        logger.info(
            f"Credential check - ENV key: {env_key_present} (preview: {env_key_preview}, len: {len(env_key_raw)}), "
            f"ENV secret: {env_secret_present} (len: {len(env_secret_raw)}), "
            f"Config key: {config_key_present} (preview: {config_key_preview}, len: {len(config.exchange.api_key) if config.exchange.api_key else 0}), "
            f"Config secret: {config_secret_present} (len: {len(config.exchange.api_secret) if config.exchange.api_secret else 0})"
        )
        
        # Critical check: if env vars exist but config doesn't have them, something is wrong
        if env_key_present and not config_key_present:
            logger.error(
                "CRITICAL: BYBIT_API_KEY found in environment but NOT loaded into config! "
                "This indicates a bug in config loading."
            )
        if env_secret_present and not config_secret_present:
            logger.error(
                "CRITICAL: BYBIT_API_SECRET found in environment but NOT loaded into config! "
                "This indicates a bug in config loading."
            )
        
        if not config.exchange.api_key or not config.exchange.api_secret:
            logger.error(
                "API credentials are missing!\n"
                "Please set BYBIT_API_KEY and BYBIT_API_SECRET in:\n"
                "  1. .env file in the project root (recommended), or\n"
                "  2. config.yaml under exchange.api_key and exchange.api_secret\n"
                "\n"
                "If running via systemd, ensure:\n"
                "  - The .env file exists at the project root\n"
                "  - The systemd service has: EnvironmentFile=/path/to/.env\n"
                "  - The .env file contains (no quotes, no spaces around =):\n"
                "    BYBIT_API_KEY=your_api_key_here\n"
                "    BYBIT_API_SECRET=your_api_secret_here\n"
                f"\n"
                f"Current status: API key from config: {'SET' if config_key_present else 'MISSING'}, "
                f"API secret from config: {'SET' if config_secret_present else 'MISSING'}"
            )
            sys.exit(1)
        
        # Log key length for debugging (without exposing actual key)
        key_len = len(config.exchange.api_key) if config.exchange.api_key else 0
        secret_len = len(config.exchange.api_secret) if config.exchange.api_secret else 0
        logger.info(
            f"API credentials loaded - Key length: {key_len} chars, Secret length: {secret_len} chars, "
            f"Testnet: {config.exchange.testnet}, Mode: {config.exchange.mode}"
        )
    
    # Initialize exchange client
    logger.info("Initializing Bybit exchange client...")
    try:
        exchange = BybitClient(config.exchange)
        logger.info("Bybit client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Bybit client: {e}", exc_info=True)
        sys.exit(1)
    
    # Test connection and validate credentials before proceeding
    logger.info("Testing connection to Bybit exchange...")
    try:
        exchange.test_connection()
        logger.info("Connection test successful - API credentials are valid")
    except Exception as e:
        logger.error(
            f"Failed to connect to Bybit exchange: {e}\n"
            "The bot cannot start without valid API credentials.\n"
            "Please fix your API credentials and try again.",
            exc_info=True
        )
        sys.exit(1)
    
    # Initialize remaining components
    logger.info("Initializing data stores and strategy components...")
    try:
        store = OHLCVStore(config.data.db_path)
        downloader = DataDownloader(exchange, store)
        portfolio = PortfolioState(exchange)
        trend_gen = TrendSignalGenerator(config.strategy.trend)
        cross_sectional_gen = CrossSectionalSignalGenerator(config.strategy.cross_sectional)
        funding_bias = FundingBiasGenerator(config.strategy.funding_bias, exchange)
        position_sizer = PositionSizer(config.risk, exchange)
        portfolio_limits = PortfolioLimits(config.risk, exchange)
        trades_store = TradesStore(config.data.db_path)
        orders_store = OrdersStore(config.data.db_path)
        logger.info("Data stores initialized successfully")
        
        executor = OrderExecutor(
            exchange,
            trades_store=trades_store,
            orders_store=orders_store,
            risk_config=config.risk,
            strategy_config=config.strategy.trend,
        )
        logger.info("Order executor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loaded config version: {config.config_version}")
    logger.info("All components initialized successfully")

    # Update portfolio state and recover any existing positions
    logger.info("Checking for existing open positions...")
    try:
        portfolio.update()
        logger.info(f"Portfolio state updated - Equity: ${portfolio.equity:,.2f}, Positions: {len(portfolio.positions)}")
        portfolio_limits.update_daily_start(portfolio.equity)
    except Exception as e:
        logger.error(f"Failed to update portfolio state: {e}", exc_info=True)
        sys.exit(1)
    
    # Log recovered positions
    if portfolio.positions:
        logger.info(f"Found {len(portfolio.positions)} existing open position(s) on exchange:")
        total_notional = 0.0
        for symbol, pos in portfolio.positions.items():
            side = pos.get('side', 'long')
            contracts = pos.get('contracts', 0)
            entry_price = pos.get('entry_price', 0)
            mark_price = pos.get('mark_price', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            notional = abs(pos.get('notional', 0))
            total_notional += notional
            
            logger.info(
                f"  {symbol}: {side.upper()} {abs(contracts):.4f} contracts | "
                f"Entry: ${entry_price:,.2f} | Mark: ${mark_price:,.2f} | "
                f"Notional: ${notional:,.2f} | PnL: ${unrealized_pnl:,.2f}"
            )
        
        logger.info(
            f"Total position notional: ${total_notional:,.2f} | "
            f"Effective leverage: {portfolio.get_leverage():.2f}x | "
            f"Equity: ${portfolio.equity:,.2f}"
        )
        logger.info("Bot will manage these positions according to strategy rules.")
    else:
        logger.info("No existing open positions found. Bot will start fresh.")
    
    # Track last rebalance time
    last_rebalance_time = None
    rebalance_frequency_hours = config.strategy.cross_sectional.rebalance_frequency_hours
    
    # Initialize universe selector if using dynamic universe
    universe_selector = None
    if config.universe.rebalance_frequency_hours > 0:
        universe_store = UniverseStore(config.data.db_path)
        universe_selector = UniverseSelector(
            config.universe,
            exchange,
            store,
            universe_store
        )
        logger.info("Dynamic universe selection enabled")
        
        # Pre-download data for initial universe to ensure data exists at startup
        logger.info("Pre-downloading data for initial universe...")
        try:
            initial_universe = list(universe_selector.get_universe())
            if not initial_universe:
                # Build universe if empty
                logger.info("Universe is empty, building initial universe...")
                initial_universe, _ = universe_selector.build_universe(config.exchange.timeframe)
            
            if initial_universe:
                # Download data for all symbols in initial universe
                downloader.update_all_symbols(
                    initial_universe,
                    config.exchange.timeframe,
                    lookback_days=30
                )
                logger.info(f"Pre-downloaded data for {len(initial_universe)} symbols in initial universe")
            else:
                logger.warning("Initial universe is empty after build - no symbols to download")
        except Exception as e:
            logger.warning(f"Error pre-downloading universe data: {e}. Continuing anyway.")
            # Continue - data will be downloaded in main loop
    
    logger.info(f"Bot initialized. Mode: {config.exchange.mode}, Equity: ${portfolio.equity:,.2f}")
    logger.info("Starting main trading loop...")
    logger.info(f"Rebalance frequency: every {rebalance_frequency_hours} hours")
    logger.info("Entering main loop - bot will run until interrupted or error occurs")
    
    loop_iteration = 0
    try:
        while True:
            loop_iteration += 1
            logger.info(f"="*60)
            logger.info(f"Loop iteration #{loop_iteration} - {datetime.now(timezone.utc).isoformat()}")
            logger.debug("Starting loop iteration processing...")
            
            try:
                # Update portfolio state
                logger.info("Updating portfolio state...")
                portfolio.update()
                logger.info(f"Portfolio updated - Equity: ${portfolio.equity:,.2f}, Positions: {len(portfolio.positions)}")
                
                # Check time-based exits and trailing stops for existing positions
                now_utc = datetime.now(timezone.utc)
                logger.debug(f"Current UTC time: {now_utc.isoformat()}")
                
                # Check time-based exits for existing positions
                if config.strategy.trend.max_holding_hours:
                    positions_to_close_time = []
                    for symbol, pos in portfolio.positions.items():
                        entry_time = pos.get('entry_time')
                        if entry_time:
                            if isinstance(entry_time, str):
                                try:
                                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                                except Exception:
                                    continue
                            hours_held = (now_utc - entry_time).total_seconds() / 3600
                            if hours_held >= config.strategy.trend.max_holding_hours:
                                positions_to_close_time.append(symbol)
                                logger.info(
                                    f"Closing {symbol}: max holding period exceeded "
                                    f"({hours_held:.1f}h >= {config.strategy.trend.max_holding_hours}h)"
                                )
                    
                    if positions_to_close_time:
                        for symbol in positions_to_close_time:
                            executor._cancel_stop_orders(symbol, portfolio)
                            result = executor.close_position(symbol)
                            if result.get('status') == 'closed':
                                logger.info(f"Closed {symbol} due to max holding period")
                
                # Check and update trailing stops for existing positions
                if config.strategy.trend.use_trailing_stop:
                    for symbol, pos in list(portfolio.positions.items()):
                        try:
                            mark_price = pos.get('mark_price', 0)
                            entry_price = pos.get('entry_price', 0)
                            side = pos.get('side', 'long')
                            current_stop = pos.get('stop_loss_price')
                            highest = pos.get('highest_price', mark_price)
                            lowest = pos.get('lowest_price', mark_price)
                            
                            if not mark_price or not entry_price or not current_stop:
                                continue
                            
                            # Update highest/lowest price
                            if side == 'long':
                                if mark_price > highest:
                                    portfolio.set_position_metadata(symbol, stop_loss_price=current_stop)
                                    pos = portfolio.positions[symbol]  # Refresh
                                    pos['highest_price'] = mark_price
                                    highest = mark_price
                            else:  # short
                                if mark_price < lowest:
                                    portfolio.set_position_metadata(symbol, stop_loss_price=current_stop)
                                    pos = portfolio.positions[symbol]  # Refresh
                                    pos['lowest_price'] = mark_price
                                    lowest = mark_price
                            
                            # Check if we should update trailing stop
                            if side == 'long':
                                # Calculate profit in ATR terms
                                price_above_entry = mark_price - entry_price
                                atr = config.strategy.trend.atr_period  # Approximate: we'd need actual ATR
                                # For now, use percentage-based trailing
                                # Trail stop up if price has moved favorably
                                if mark_price > highest * 0.95:  # Near highest
                                    # Calculate trailing stop distance
                                    atr_estimate = mark_price * 0.01  # Rough 1% ATR approximation
                                    new_stop = mark_price - (atr_estimate * config.strategy.trend.trailing_stop_atr_multiplier)
                                    # Only move stop up, never down
                                    if new_stop > current_stop:
                                        # Check if profit is enough to activate trailing
                                        profit_pct = (mark_price - entry_price) / entry_price
                                        stop_distance_pct = (entry_price - current_stop) / entry_price
                                        if profit_pct >= (stop_distance_pct * config.strategy.trend.trailing_stop_activation_rr):
                                            logger.info(
                                                f"Updating trailing stop for {symbol}: {current_stop:.2f} -> {new_stop:.2f}"
                                            )
                                            # Cancel old stop order and place new one
                                            executor._cancel_stop_orders(symbol, portfolio)
                                            contracts = abs(pos.get('contracts', 0))
                                            new_stop_id = executor._place_stop_loss_order(
                                                symbol,
                                                side,
                                                contracts,
                                                new_stop,
                                                portfolio,
                                            )
                                            if new_stop_id:
                                                logger.info(f"Trailing stop updated for {symbol}: order_id={new_stop_id}")
                            # Similar logic for short positions...
                            # (implementation similar, trailing stop moves down)
                            
                        except Exception as e:
                            logger.warning(f"Error updating trailing stop for {symbol}: {e}")
                
                # Check daily loss limits (MUST be inside try block)
                logger.debug("Checking daily loss limits...")
                can_trade, loss_reason = portfolio_limits.check_daily_loss_limits(
                    portfolio.equity,
                    realized_pnl=0.0  # TODO: Calculate from closed trades
                )
                logger.debug(f"Daily loss check result: can_trade={can_trade}, loss_reason={loss_reason}")
                
                if not can_trade:
                    logger.warning(f"Daily HARD loss limit breached: {loss_reason}")
                    
                    # Flatten all positions on hard loss cap
                    if portfolio.positions:
                        logger.critical(
                            f"Flattening all {len(portfolio.positions)} position(s) due to hard loss cap"
                        )
                        for symbol in list(portfolio.positions.keys()):
                            executor._cancel_stop_orders(symbol, portfolio)
                            result = executor.close_position(symbol)
                            if result.get('status') == 'closed':
                                logger.info(f"Flattened {symbol} due to hard loss cap")
                    
                    logger.error(
                        "Trading disabled until next UTC day. Bot will sleep and check again."
                    )
                    # Wait until next day
                    time.sleep(3600)  # Check again in 1 hour
                    continue
                elif loss_reason:  # Soft loss cap reached
                    logger.warning(f"Daily soft loss cap reached: {loss_reason}. Continuing with reduced risk.")
                
                # Update data (only if we can trade)
                logger.info("Updating market data...")
                
                # Determine which symbols to download data for
                if universe_selector is not None:
                    # Dynamic universe mode: download for current universe + potential new symbols
                    current_universe = list(universe_selector.get_universe())
                    # Also fetch all symbols from exchange to check for new listings
                    # Limit to top 100 for performance
                    try:
                        all_exchange_symbols = universe_selector.fetch_all_symbols()
                        # Combine current universe with exchange symbols (limit to top 100 to avoid excessive API calls)
                        symbols_to_download = list(set(current_universe + all_exchange_symbols[:100]))
                        logger.debug(f"Downloading data for {len(symbols_to_download)} symbols (universe: {len(current_universe)}, exchange: {len(all_exchange_symbols)})")
                    except Exception as e:
                        logger.warning(f"Error fetching all exchange symbols, using current universe only: {e}")
                        symbols_to_download = current_universe
                else:
                    # Fixed list mode
                    symbols_to_download = config.exchange.symbols
                
                downloader.update_all_symbols(
                    symbols_to_download,
                    config.exchange.timeframe,
                    lookback_days=30
                )
                logger.info(f"Market data updated successfully for {len(symbols_to_download)} symbols")
                
                # Determine if we should rebalance
                now = datetime.now(timezone.utc)
                should_rebalance = False
                
                if last_rebalance_time is None:
                    logger.info("First iteration - will rebalance now")
                    should_rebalance = True
                else:
                    hours_diff = (now - last_rebalance_time).total_seconds() / 3600
                    logger.info(f"Last rebalance: {last_rebalance_time.isoformat()} ({hours_diff:.1f} hours ago)")
                    if hours_diff >= rebalance_frequency_hours:
                        logger.info(f"Rebalance threshold reached ({hours_diff:.1f}h >= {rebalance_frequency_hours}h) - will rebalance")
                        should_rebalance = True
                    else:
                        logger.info(f"Rebalance threshold not reached ({hours_diff:.1f}h < {rebalance_frequency_hours}h) - skipping rebalance")
                    
                if should_rebalance:
                    logger.info("="*60)
                    logger.info("REBALANCING PORTFOLIO")
                    logger.info("="*60)
                    logger.info("Rebalancing portfolio...")
                
                # Get current universe (dynamic or fixed)
                if universe_selector is not None:
                    # Get current universe
                    trading_symbols = list(universe_selector.get_universe())
                    
                    # Rebuild universe if needed (based on rebalance frequency)
                    universe_rebalance_hours = config.universe.rebalance_frequency_hours
                    hours_since_last_universe_rebuild = (
                        (now - last_rebalance_time).total_seconds() / 3600
                        if last_rebalance_time else float('inf')
                    )
                    
                    if hours_since_last_universe_rebuild >= universe_rebalance_hours:
                        logger.info("Rebuilding universe...")
                        trading_symbols, universe_changes = universe_selector.build_universe(
                            config.exchange.timeframe
                        )
                        logger.info(
                            f"Universe updated: {len(trading_symbols)} symbols "
                            f"({len([c for c in universe_changes.values() if c['action'] == 'added'])} added, "
                            f"{len([c for c in universe_changes.values() if c['action'] == 'removed'])} removed)"
                        )
                else:
                    # Use fixed symbol list from config
                    trading_symbols = config.exchange.symbols
                
                # Generate signals for all symbols in universe
                symbol_signals = {}
                symbol_data = {}
                
                for symbol in trading_symbols:
                    try:
                        df = store.get_ohlcv(
                            symbol,
                            config.exchange.timeframe,
                            limit=config.data.lookback_bars
                        )
                        
                        if df.empty:
                            logger.warning(
                                f"Symbol {symbol} is in trading universe but has no data in database. "
                                f"Ensure downloader has fetched data for all universe symbols. "
                                f"Skipping signal generation for this symbol."
                            )
                            continue
                        
                        if len(df) < config.strategy.trend.ma_long:
                            logger.debug(
                                f"Symbol {symbol} has insufficient data: {len(df)} bars "
                                f"(need {config.strategy.trend.ma_long})"
                            )
                            continue
                        
                        symbol_data[symbol] = df
                        
                        # Generate trend signal
                        trend_signal = trend_gen.generate_signal(df)
                        symbol_signals[symbol] = trend_signal
                        
                    except Exception as e:
                        logger.warning(f"Error generating signal for {symbol}: {e}")
                        continue
                
                # Cross-sectional selection
                selected_symbols = cross_sectional_gen.select_top_symbols(
                    symbol_data,
                    symbol_signals,
                    config.strategy.cross_sectional.require_trend_alignment
                )
                
                logger.info(f"Selected symbols: {selected_symbols}")
                
                # Generate target positions
                target_positions = {}
                
                for symbol in selected_symbols:
                    signal_dict = symbol_signals.get(symbol)
                    if not signal_dict or signal_dict['signal'] == 'flat':
                        continue
                    
                    signal = signal_dict['signal']
                    entry_price = signal_dict['entry_price']
                    stop_loss = signal_dict['stop_loss']
                    
                    if not stop_loss:
                        continue
                    
                    # Calculate position size
                    size, error = position_sizer.calculate_position_size(
                        symbol,
                        portfolio.equity,
                        entry_price,
                        stop_loss,
                        signal
                    )
                    
                    if size <= 0:
                        logger.warning(f"Skipping {symbol}: {error}")
                        continue
                    
                    # Apply funding bias adjustment
                    adjusted_size = funding_bias.calculate_size_adjustment(
                        symbol,
                        signal,
                        size
                    )
                    
                    # Check portfolio limits
                    market_info = exchange.get_market_info(symbol)
                    contract_size = market_info.get('contractSize', 1.0)
                    notional = adjusted_size * entry_price * contract_size
                    
                    within_limits, limit_error = portfolio_limits.check_leverage_limit(
                        portfolio,
                        notional,
                        signal
                    )
                    
                    if not within_limits:
                        logger.warning(f"Position for {symbol} exceeds leverage: {limit_error}")
                        # Scale down
                        adjusted_size, scale_reason = portfolio_limits.scale_position_for_limits(
                            portfolio,
                            symbol,
                            adjusted_size,
                            entry_price,
                            signal
                        )
                        if adjusted_size <= 0:
                            logger.warning(f"Cannot scale {symbol} to fit limits: {scale_reason}")
                            continue
                    
                    # Check symbol concentration
                    within_concentration, conc_error = portfolio_limits.check_symbol_concentration(
                        portfolio,
                        symbol,
                        notional
                    )
                    
                    if not within_concentration:
                        logger.warning(f"Position for {symbol} exceeds concentration: {conc_error}")
                        adjusted_size, scale_reason = portfolio_limits.scale_position_for_limits(
                            portfolio,
                            symbol,
                            adjusted_size,
                            entry_price,
                            signal
                        )
                        if adjusted_size <= 0:
                            continue
                    
                    # Check max positions
                    within_max, max_error = portfolio_limits.check_max_positions(
                        portfolio,
                        symbol
                    )
                    
                    if not within_max:
                        logger.warning(f"Max positions reached: {max_error}")
                        continue
                    
                    # Convert to signed size (positive for long, negative for short)
                    signed_size = adjusted_size if signal == 'long' else -adjusted_size
                    
                    target_positions[symbol] = {
                        'size': signed_size,
                        'signal': signal,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss
                    }
                
                # Execute position changes
                # Note: reconcile_positions handles both:
                # 1. Positions to open/adjust (in target_positions)
                # 2. Positions to close (existing but not in target_positions)
                # This ensures that positions opened before restart are properly managed
                if target_positions or portfolio.positions:
                    existing_count = len(portfolio.positions)
                    target_count = len(target_positions)
                    logger.info(
                        f"Reconciling positions: {existing_count} existing position(s), "
                        f"{target_count} target position(s)"
                    )
                    results = executor.reconcile_positions(portfolio, target_positions)
                    
                    for result in results:
                        if result.get('status') == 'filled':
                            logger.info(f"Position updated: {result.get('symbol')}")
                        elif result.get('status') == 'closed':
                            logger.info(f"Position closed: {result.get('symbol')}")
                        elif result.get('status') == 'error':
                            logger.error(f"Error executing position: {result.get('error')}")
                else:
                    logger.info("No position changes needed (no targets and no existing positions)")
                
                last_rebalance_time = now
                
                # Heartbeat / health summary
                logger.info(
                    "Heartbeat | mode=%s | config_version=%s | equity=%.2f | positions=%d | open_symbols=%s",
                    config.exchange.mode,
                    config.config_version,
                    portfolio.equity,
                    len(portfolio.positions),
                    ", ".join(portfolio.positions.keys()) if portfolio.positions else "none"
                )

                # Sleep until next iteration (check every hour for rebalancing)
                sleep_seconds = 3600  # 1 hour
                logger.info(f"Loop iteration #{loop_iteration} complete. Sleeping for {sleep_seconds}s ({sleep_seconds/60:.0f} minutes) until next check...")
                logger.info("="*60)
                time.sleep(sleep_seconds)
            
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                # CRITICAL: Log the full exception with stack trace
                logger.error("=" * 60)
                logger.error(f"EXCEPTION in trading loop iteration #{loop_iteration}: {type(e).__name__}: {e}")
                logger.error(f"Exception location: {e.__class__.__module__}.{e.__class__.__name__}")
                logger.error("Full traceback:", exc_info=True)
                logger.error("=" * 60)
                # Wait 60 seconds before retrying to avoid rapid error loops
                logger.warning(f"Sleeping 60 seconds before retrying loop iteration...")
                time.sleep(60)
                logger.info(f"Resuming loop after exception recovery")
                continue
    
    except Exception as e:
        logger.error(f"Fatal error in live bot: {e}", exc_info=True)
        sys.exit(1)


def run_backtest(config_path: str, symbols: list = None, output_file: str = None):
    """Run backtest."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="backtest",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    logger.info("Starting backtest")
    
    # Initialize components
    store = OHLCVStore(config.data.db_path)
    backtester = Backtester(config)
    
    # Load data
    test_symbols = symbols or config.exchange.symbols
    symbol_data = {}
    
    for symbol in test_symbols:
        try:
            df = store.get_ohlcv(symbol, config.exchange.timeframe, limit=config.data.lookback_bars)
            if not df.empty:
                symbol_data[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}")
    
    if not symbol_data:
        logger.error("No data available for backtest")
        sys.exit(1)
    
    # Run backtest
    # Pass slippage parameters to backtest
    stop_slippage = config.risk.stop_slippage_bps if config.risk else 10.0
    tp_slippage = config.risk.tp_slippage_bps if config.risk else 5.0
    result = backtester.backtest(
        symbol_data,
        stop_slippage_bps=stop_slippage,
        tp_slippage_bps=tp_slippage,
    )
    
    if 'error' in result:
        logger.error(f"Backtest error: {result['error']}")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital: ${result['initial_capital']:,.2f}")
    print(f"Final Equity: ${result['final_equity']:,.2f}")
    print(f"Total Return: {result['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']*100:.1f}%")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print("="*60)
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")


def run_health(config_path: str):
    """Basic healthcheck: DB + exchange connectivity."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="health",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)

    ok = True

    # DB check
    try:
        conn = sqlite3.connect(config.data.db_path)
        conn.execute("SELECT 1")
        conn.close()
        logger.info("Healthcheck: DB OK (%s)", config.data.db_path)
    except Exception as e:
        logger.error("Healthcheck: DB FAILED (%s): %s", config.data.db_path, e)
        ok = False

    # Exchange check (simple balance fetch in paper/testnet)
    try:
        exchange = BybitClient(config.exchange)
        _ = exchange.fetch_balance()
        logger.info("Healthcheck: Exchange OK (mode=%s, testnet=%s)", config.exchange.mode, config.exchange.testnet)
    except Exception as e:
        logger.error("Healthcheck: Exchange FAILED: %s", e)
        ok = False

    if not ok:
        sys.exit(1)


def run_optimize(config_path: str):
    """Run parameter optimization."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="optimizer",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    logger.info("Starting optimization")
    
    # Initialize components
    store = OHLCVStore(config.data.db_path)
    optimizer = Optimizer(config, store)
    
    # Run optimization
    result = optimizer.optimize(config.exchange.symbols, config.exchange.timeframe)
    
    if 'error' in result:
        logger.error(f"Optimization error: {result['error']}")
        sys.exit(1)
    
    # Compare with current
    comparison = optimizer.compare_with_current(result['best_params'])
    result['comparison'] = comparison
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Parameters: {result['best_params']}")
    print(f"Metrics:")
    metrics = result['best_metrics']
    print(f"  Avg Return: {metrics['avg_return_pct']:+.2f}%")
    print(f"  Avg Sharpe: {metrics['avg_sharpe']:.2f}")
    print(f"  Avg Drawdown: {metrics['avg_drawdown_pct']:.2f}%")
    print(f"  Avg Trades: {metrics['avg_trades']:.0f}")
    print(f"\nComparison: {comparison['recommendation'] if comparison.get('should_update') else 'No update needed'}")
    print("="*60)
    
    # Save result
    optimizer.save_optimization_result(result, config.data.db_path)


def run_optimize_universe(
    config_path: str,
    start_date: str,
    end_date: str,
    n_combinations: int = 200,
    method: str = "random",
    output_file: Optional[str] = None
):
    """Run universe parameter optimization."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="universe-optimizer",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    logger.info("Starting universe parameter optimization")
    
    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Initialize components
    ohlcv_store = OHLCVStore(config.data.db_path)
    optimizer = UniverseOptimizer(config, ohlcv_store, config.data.db_path)
    
    # Load historical data for all available symbols
    logger.info("Loading historical data...")
    all_symbols = ohlcv_store.db_path  # Get all symbols from database
    # For now, use symbols from config or fetch all from DB
    # In production, would query DB for all symbols with data in date range
    
    # Load data for symbols in config (or all if empty)
    symbols_to_load = config.exchange.symbols if config.exchange.symbols else []
    if not symbols_to_load:
        # Load all symbols from database
        conn = sqlite3.connect(config.data.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ?", (config.exchange.timeframe,))
        symbols_to_load = [row[0] for row in cursor.fetchall()]
        conn.close()
    
    symbol_data = {}
    since = int(start.timestamp() * 1000) - (180 * 24 * 3600 * 1000)  # Start earlier for history
    
    for symbol in symbols_to_load:
        try:
            df = ohlcv_store.get_ohlcv(symbol, config.exchange.timeframe, since=since)
            if not df.empty and len(df) > 100:  # Need sufficient data
                # Filter to date range
                df_filtered = df[(df.index.date >= start) & (df.index.date <= end)]
                if not df_filtered.empty:
                    symbol_data[symbol] = df_filtered
                    logger.debug(f"Loaded {len(df_filtered)} candles for {symbol}")
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}")
            continue
    
    if not symbol_data:
        logger.error("No data available for universe optimization")
        sys.exit(1)
    
    logger.info(f"Loaded data for {len(symbol_data)} symbols")
    
    # Run optimization
    results = optimizer.optimize(
        symbol_data,
        start,
        end,
        config.exchange.timeframe,
        n_combinations=n_combinations,
        method=method
    )
    
    if not results:
        logger.error("No valid parameter sets found")
        sys.exit(1)
    
    # Select best configs
    best_configs = optimizer.select_best_configs(results, n_top=5)
    
    # Print results
    print("\n" + "="*80)
    print("UNIVERSE PARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Backtest Period: {start_date} to {end_date}")
    print(f"Total Parameter Sets Tested: {len(results)} passed constraints")
    print(f"\nTop 5 Recommended Configurations:")
    print("="*80)
    
    for i, result in enumerate(best_configs, 1):
        label = result.params.get('_label', f'Config #{i}')
        print(f"\n{label}:")
        print("-" * 80)
        print(f"Composite Score: {result.composite_score:.4f}")
        print(f"\nStrategy Performance:")
        print(f"  Annualized Return: {result.annualized_return_pct:+.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate*100:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"\nUniverse Quality:")
        print(f"  Avg Universe Size: {result.avg_universe_size:.1f} symbols")
        print(f"  Avg 24h Volume: ${result.avg_volume_24h:,.0f} USDT")
        print(f"  Universe Turnover Rate: {result.universe_turnover_rate:.2f}%")
        print(f"  Avg Time in Universe: {result.avg_time_in_universe_days:.1f} days")
        print(f"\nKey Parameters:")
        print(f"  min_24h_volume_entry: ${result.params['min_24h_volume_entry']:,.0f}")
        print(f"  min_24h_volume_exit: ${result.params['min_24h_volume_exit']:,.0f}")
        print(f"  min_history_days: {result.params['min_history_days']}")
        print(f"  warmup_days: {result.params['warmup_days']}")
        print(f"  min_time_in_universe_days: {result.params['min_time_in_universe_days']}")
        
        # Print config YAML
        print(f"\nConfig YAML (copy to config.yaml):")
        print("-" * 80)
        yaml_config = optimizer.results_to_config_yaml(result)
        print(yaml_config)
    
    print("\n" + "="*80)
    
    # Save to file if requested
    if output_file:
        import json
        output_data = {
            'backtest_period': {
                'start': start_date,
                'end': end_date
            },
            'total_tested': len(results),
            'top_configs': [
                {
                    'label': r.params.get('_label', f'Config #{i}'),
                    'composite_score': r.composite_score,
                    'params': r.params,
                    'strategy_metrics': {
                        'annualized_return_pct': r.annualized_return_pct,
                        'sharpe_ratio': r.sharpe_ratio,
                        'max_drawdown_pct': r.max_drawdown_pct,
                        'total_trades': r.total_trades,
                        'win_rate': r.win_rate,
                        'profit_factor': r.profit_factor,
                    },
                    'universe_metrics': {
                        'avg_universe_size': r.avg_universe_size,
                        'avg_volume_24h': r.avg_volume_24h,
                        'universe_turnover_rate': r.universe_turnover_rate,
                    },
                    'config_yaml': optimizer.results_to_config_yaml(r)
                }
                for i, r in enumerate(best_configs, 1)
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")


def run_compare_timeframes(
    config_path: str,
    timeframes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    is_oos_split: float = 0.7
):
    """Compare strategy performance across different timeframes."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="timeframe-compare",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    logger.info("Starting timeframe comparison")
    
    # Parse dates
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Use symbols from config if not provided
    if symbols is None:
        symbols = config.exchange.symbols
    
    # Default output file
    if output_file is None:
        output_file = "results/timeframe_comparison.json"
    
    # Initialize components
    store = OHLCVStore(config.data.db_path)
    analyzer = TimeframeAnalyzer(config, store)
    
    # Run comparison
    results = analyzer.compare_timeframes(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_dt,
        end_date=end_dt,
        is_oos_split=is_oos_split,
        classify_regimes=False  # Simplified for initial version
    )
    
    if not results:
        logger.error("No results from timeframe comparison")
        sys.exit(1)
    
    # Print results table
    print("\n" + "="*100)
    print("TIMEFRAME COMPARISON RESULTS")
    print("="*100)
    print(f"{'TF':<6} {'Return%':<10} {'AnnRet%':<10} {'Sharpe':<8} {'Sortino':<8} {'MaxDD%':<8} {'PF':<6} {'WR%':<6} {'Trades':<8} {'Trades/Day':<10} {'AvgHoldH':<10}")
    print("-"*100)
    
    for r in results:
        print(
            f"{r.timeframe:<6} "
            f"{r.total_return_pct:>9.2f}% "
            f"{r.annualized_return_pct:>9.2f}% "
            f"{r.sharpe_ratio:>7.2f} "
            f"{r.sortino_ratio:>7.2f} "
            f"{r.max_drawdown_pct:>7.2f}% "
            f"{r.profit_factor:>5.2f} "
            f"{r.win_rate*100:>5.1f}% "
            f"{r.total_trades:>7d} "
            f"{r.trades_per_day:>9.2f} "
            f"{r.avg_holding_hours:>9.1f}h"
        )
    
    print("="*100)
    
    # Print best timeframe
    best = results[0]  # Already sorted by Sharpe
    print(f"\nBest Timeframe (by Sharpe): {best.timeframe}")
    print(f"  Annualized Return: {best.annualized_return_pct:.2f}%")
    print(f"  Sharpe Ratio: {best.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {best.max_drawdown_pct:.2f}%")
    print(f"  Total Trades: {best.total_trades}")
    print(f"  Trades/Day: {best.trades_per_day:.2f}")
    
    if best.oos_return_pct is not None:
        print(f"\nOut-of-Sample Performance:")
        print(f"  OOS Return: {best.oos_return_pct:.2f}%")
        print(f"  OOS Sharpe: {best.oos_sharpe:.2f}")
        print(f"  OOS Drawdown: {best.oos_drawdown_pct:.2f}%")
    
    print("="*100 + "\n")
    
    # Save results
    analyzer.save_results(results, output_file)
    logger.info(f"Results saved to {output_file}")


def run_report(config_path: str):
    """Send daily Discord report."""
    # Load config FIRST
    config = BotConfig.from_yaml(config_path)
    
    # Setup logging with config
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_log_size_mb=config.logging.max_log_size_mb,
        backup_count=config.logging.backup_count,
        service_name="report",
        force=True,
    )
    
    # Now get logger
    logger = get_logger(__name__)
    logger.info("Sending daily Discord report")
    
    # Initialize components
    exchange = BybitClient(config.exchange)
    portfolio = PortfolioState(exchange)
    reporter = DiscordReporter(config.reporting, exchange)
    
    # Update portfolio state
    portfolio.update()
    
    # Send report
    success = reporter.send_daily_report(portfolio, config.data.db_path, config)
    
    if success:
        logger.info("Report sent successfully")
    else:
        logger.error("Failed to send report")
        sys.exit(1)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Bybit Trading Bot")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Run live trading bot')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--symbols', nargs='+', help='Symbols for backtest (optional)')
    backtest_parser.add_argument('--output', help='Output file for backtest results (optional)')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run strategy parameter optimization')
    
    # Optimize universe command
    optimize_universe_parser = subparsers.add_parser('optimize-universe', help='Run universe parameter optimization')
    optimize_universe_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    optimize_universe_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    optimize_universe_parser.add_argument('--n-combinations', type=int, default=200, help='Number of parameter combinations to test')
    optimize_universe_parser.add_argument('--method', choices=['random', 'grid'], default='random', help='Search method')
    optimize_universe_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Compare timeframes command
    compare_tf_parser = subparsers.add_parser('compare-timeframes', help='Compare strategy performance across different timeframes')
    compare_tf_parser.add_argument('--timeframes', nargs='+', default=['15m', '30m', '1h', '2h', '4h', '6h', '1d'], help='Timeframes to compare (default: 15m 30m 1h 2h 4h 6h 1d)')
    compare_tf_parser.add_argument('--start', help='Start date (YYYY-MM-DD, default: 1 year ago)')
    compare_tf_parser.add_argument('--end', help='End date (YYYY-MM-DD, default: now)')
    compare_tf_parser.add_argument('--symbols', nargs='+', help='Symbols to test (default: from config)')
    compare_tf_parser.add_argument('--output', help='Output file for results (JSON, default: results/timeframe_comparison.json)')
    compare_tf_parser.add_argument('--is-oos-split', type=float, default=0.7, help='In-sample split ratio (default: 0.7)')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Send daily Discord report')
    
    # Universe commands
    universe_parser = subparsers.add_parser('universe', help='Universe management')
    universe_subparsers = universe_parser.add_subparsers(dest='universe_action', help='Universe action')
    
    universe_build_parser = universe_subparsers.add_parser('build', help='Build/update universe')
    universe_show_parser = universe_subparsers.add_parser('show', help='Show current universe')
    universe_history_parser = universe_subparsers.add_parser('history', help='Show universe history')
    universe_history_parser.add_argument('--symbol', help='Symbol to show history for (optional)')

    # Healthcheck command
    health_parser = subparsers.add_parser('health', help='Run basic healthcheck')
    
    # Common argument (available to all commands)
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    config_path = getattr(args, 'config', 'config.yaml')
    
    if args.command == 'live':
        run_live(config_path)
    elif args.command == 'backtest':
        run_backtest(config_path, getattr(args, 'symbols', None), getattr(args, 'output', None))
    elif args.command == 'optimize':
        run_optimize(config_path)
    elif args.command == 'optimize-universe':
        run_optimize_universe(
            config_path,
            getattr(args, 'start'),
            getattr(args, 'end'),
            getattr(args, 'n_combinations', 200),
            getattr(args, 'method', 'random'),
            getattr(args, 'output', None)
        )
    elif args.command == 'compare-timeframes':
        run_compare_timeframes(
            config_path,
            getattr(args, 'timeframes', ['15m', '30m', '1h', '2h', '4h', '6h', '1d']),
            getattr(args, 'start', None),
            getattr(args, 'end', None),
            getattr(args, 'symbols', None),
            getattr(args, 'output', None),
            getattr(args, 'is_oos_split', 0.7)
        )
    elif args.command == 'report':
        run_report(config_path)
    elif args.command == 'universe':
        if args.universe_action == 'build':
            run_universe_build(config_path)
        elif args.universe_action == 'show':
            run_universe_show(config_path)
        elif args.universe_action == 'history':
            run_universe_history(config_path, getattr(args, 'symbol', None))
        else:
            universe_parser.print_help()
            sys.exit(1)
    elif args.command == 'health':
        run_health(config_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

