"""Main CLI entrypoint."""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from typing import Optional
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
from ..reporting.discord_reporter import DiscordReporter
from ..universe.selector import UniverseSelector
from ..universe.store import UniverseStore
import sqlite3


def run_universe_build(config_path: str):
    """Build/update the universe."""
    logger = get_logger(__name__)
    logger.info("Building universe")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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
    logger = get_logger(__name__)
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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
    logger = get_logger(__name__)
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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
    logger = get_logger(__name__)
    logger.info("Starting live trading bot")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    errors = config.validate()
    if errors:
        logger.error(f"Config validation errors: {errors}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
    # Validate API credentials early
    if config.exchange.mode != "paper":
        if not config.exchange.api_key or not config.exchange.api_secret:
            logger.error(
                "API credentials are missing!\n"
                "Please set BYBIT_API_KEY and BYBIT_API_SECRET in:\n"
                "  1. .env file in the project root (recommended), or\n"
                "  2. config.yaml under exchange.api_key and exchange.api_secret\n"
                "\n"
                "If running via systemd, ensure the .env file exists and contains:\n"
                "  BYBIT_API_KEY=your_api_key_here\n"
                "  BYBIT_API_SECRET=your_api_secret_here"
            )
            sys.exit(1)
    
    # Initialize exchange client
    exchange = BybitClient(config.exchange)
    
    # Test connection and validate credentials before proceeding
    try:
        exchange.test_connection()
    except Exception as e:
        logger.error(
            f"Failed to connect to Bybit exchange: {e}\n"
            "The bot cannot start without valid API credentials.\n"
            "Please fix your API credentials and try again."
        )
        sys.exit(1)
    
    # Initialize remaining components
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
    executor = OrderExecutor(exchange, trades_store=trades_store, orders_store=orders_store)

    logger.info(f"Loaded config version: {config.config_version}")

    # Update portfolio state
    portfolio.update()
    portfolio_limits.update_daily_start(portfolio.equity)
    
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
    
    logger.info(f"Bot initialized. Mode: {config.exchange.mode}, Equity: ${portfolio.equity:,.2f}")
    
    try:
        while True:
            try:
                # Update portfolio state
                portfolio.update()
            except ccxt.AuthenticationError as e:
                # Authentication errors are fatal - stop the bot
                logger.critical(
                    f"Authentication error during trading loop: {e}\n"
                    "The bot will stop. Please check your API credentials and restart."
                )
                sys.exit(1)
                
                # Check daily loss limits
                can_trade, loss_reason = portfolio_limits.check_daily_loss_limits(
                    portfolio.equity,
                    realized_pnl=0.0  # TODO: Calculate from closed trades
                )
                
                if not can_trade:
                    logger.warning(f"Daily loss limit breached: {loss_reason}")
                    # Wait until next day
                    time.sleep(3600)  # Check again in 1 hour
                    continue
                
                # Update data
                logger.info("Updating market data...")
                downloader.update_all_symbols(
                    config.exchange.symbols,
                    config.exchange.timeframe,
                    lookback_days=30
                )
                
                # Determine if we should rebalance
                now = datetime.now(timezone.utc)
                should_rebalance = False
                
                if last_rebalance_time is None:
                    should_rebalance = True
                else:
                    hours_diff = (now - last_rebalance_time).total_seconds() / 3600
                    if hours_diff >= rebalance_frequency_hours:
                        should_rebalance = True
                
                if should_rebalance:
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
                            
                            if df.empty or len(df) < config.strategy.trend.ma_long:
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
                    if target_positions:
                        logger.info(f"Executing {len(target_positions)} position changes...")
                        results = executor.reconcile_positions(portfolio, target_positions)
                        
                        for result in results:
                            if result.get('status') == 'filled':
                                logger.info(f"Position updated: {result.get('symbol')}")
                            elif result.get('status') == 'error':
                                logger.error(f"Error executing position: {result.get('error')}")
                    
                    last_rebalance_time = now

                # Heartbeat / health summary
                logger.info(
                    "Heartbeat | mode=%s | config_version=%s | equity=%.2f | positions=%d",
                    config.exchange.mode,
                    config.config_version,
                    portfolio.equity,
                    len(portfolio.positions),
                )

                # Sleep until next iteration (check every hour for rebalancing)
                sleep_seconds = 3600  # 1 hour
                logger.info(f"Sleeping for {sleep_seconds}s until next check...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)  # Wait before retrying
    
    except Exception as e:
        logger.error(f"Fatal error in live bot: {e}", exc_info=True)
        sys.exit(1)


def run_backtest(config_path: str, symbols: list = None, output_file: str = None):
    """Run backtest."""
    logger = get_logger(__name__)
    logger.info("Starting backtest")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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
    result = backtester.backtest(symbol_data)
    
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
    logger = get_logger(__name__)

    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count,
    )

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
    logger = get_logger(__name__)
    logger.info("Starting optimization")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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
    logger = get_logger(__name__)
    logger.info("Starting universe parameter optimization")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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


def run_report(config_path: str):
    """Send daily Discord report."""
    logger = get_logger(__name__)
    logger.info("Sending daily Discord report")
    
    # Load config
    config = BotConfig.from_yaml(config_path)
    setup_logging(
        config.logging.log_dir,
        config.logging.level,
        config.logging.max_log_size_mb,
        config.logging.backup_count
    )
    
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

