# Bybit Trading Bot

A fully-automated, self-improving crypto trading bot for Bybit perpetual futures (USDT-margined). This bot implements systematic trading strategies with comprehensive risk management and automated parameter optimization.

## Features

- **Multi-Strategy System**: Implements time-series trend-following, cross-sectional momentum, and funding-rate bias overlays
- **Risk Management**: Comprehensive risk controls including position sizing, leverage limits, daily loss caps, and portfolio-level constraints
- **Self-Improvement**: Daily parameter optimization using walk-forward analysis
- **Daily Reporting**: Automated Discord webhook reports with performance metrics and risk analysis
- **Paper/Testnet Support**: Safe testing with paper trading and Bybit testnet integration
- **CLI-Based**: Runs headless on Ubuntu servers
- **Single-Script Installation**: Easy setup with `install.sh`

## ⚠️ Important Warnings

- **No Guarantee of Profitability**: Trading bots can lose money. This bot is for educational and research purposes. Past performance does not guarantee future results.
- **Use at Your Own Risk**: Cryptocurrency trading involves substantial risk. Only trade with capital you can afford to lose.
- **Test First**: Always test in paper/testnet mode before using real capital.
- **Monitor Performance**: Regularly review performance and risk metrics. Markets change, and strategies that worked in the past may stop working.
- **API Key Security**: Keep your API keys secure. Use API keys with trading permissions only, not withdrawal permissions.

## Architecture

### Strategy Sleeves

1. **Sleeve A: Time-Series Trend-Following**
   - Dual moving average crossover system
   - Momentum confirmation
   - ATR-based stop losses
   - Volatility filtering

2. **Sleeve B: Cross-Sectional Momentum**
   - Ranks symbols by recent performance
   - Selects top K performers
   - Rebalances periodically (e.g., every 4 hours)

3. **Sleeve C: Funding-Rate Bias Overlay**
   - Adjusts position sizes based on funding rates
   - Reduces size when paying high funding
   - Increases size when receiving funding (within limits)

### Risk Management

- **Position Sizing**: Volatility-scaled with ATR-based stops
- **Per-Trade Risk**: Configurable % of equity at risk per trade (default: 0.5%)
- **Portfolio Limits**: Max leverage, max symbol concentration, max positions
- **Daily Loss Caps**: Soft and hard daily loss limits (default: -2% / -4%)
- **Exchange Constraints**: Respects Bybit precision, min sizes, fees

## Logging

The bot uses a centralized logging system that writes to both files and systemd/journald.

### Viewing Logs

**File Logs:**
```bash
# View live bot logs
tail -f logs/bot-live.log

# View all logs
tail -f logs/*.log
```

**Systemd/Journald:**
```bash
# View live bot service logs
sudo journalctl -u bybit-bot.service -f

# View all bot services
sudo journalctl -u bybit-bot* -f
```

See [LOGGING_SYSTEM.md](LOGGING_SYSTEM.md) for detailed logging documentation.

## Installation

### Prerequisites

- Ubuntu 20.04+ or Debian 11+
- Python 3.10+
- sudo/root access for systemd services
- Bybit API keys (from [Bybit API Management](https://www.bybit.com/app/user/api-management))
- Discord webhook URL (optional, for daily reports)

### Quick Install

```bash
# Clone or download the repository
cd trading-bot

# Run installation script
./install.sh
```

The installation script will:
1. Install system dependencies (Python, git, curl, etc.)
2. Create a Python virtual environment
3. Install Python packages
4. Prompt for API keys and configuration
5. Set up systemd services and timers
6. Configure daily optimizer and report schedules

### Manual Installation

If you prefer manual installation:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git curl jq sqlite3

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Copy example config
cp config.example.yaml config.yaml

# Create .env file
cat > .env <<EOF
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
DISCORD_WEBHOOK_URL=your_webhook_url_here
EOF
```

## Universe Selection

The bot includes a **dynamic universe selection system** that automatically discovers and maintains a set of tradable Bybit USDT perpetual contracts.

### How It Works

1. **Dynamic Discovery**: Fetches all available USDT perpetuals from Bybit API
2. **Liquidity Filtering**: Only includes symbols with sufficient 24h volume (default: $10M entry, $7M exit for hysteresis)
3. **Data Quality Checks**: Requires minimum 30 days of historical data with <5% gaps
4. **Warm-Up Period**: New listings must pass filters for 14 days before entering universe
5. **Hysteresis**: Separate thresholds for entry vs. exit to reduce churn
6. **Volatility Filters**: Rejects symbols with excessive volatility or dust prices

### Configuration

Universe selection is configured in the `universe` section of `config.yaml`:

```yaml
universe:
  # Liquidity filters
  min_24h_volume_entry: 10_000_000    # $10M USDT entry threshold
  min_24h_volume_exit: 7_000_000      # $7M USDT exit threshold (hysteresis)
  volume_check_days: 7                 # Consecutive days for entry/exit
  
  # Historical data
  min_history_days: 30                 # Minimum days of data required
  warmup_days: 14                      # Warm-up period for new listings
  
  # Rebalancing
  rebalance_frequency_hours: 24        # Recompute universe daily
  
  # Overrides
  include_list: [BTCUSDT, ETHUSDT]     # Always included
  exclude_list: []                     # Always excluded
```

### CLI Commands

```bash
# Build/update universe now
python -m src.main universe build --config config.yaml

# Show current universe
python -m src.main universe show --config config.yaml

# Show universe history (last 30 days)
python -m src.main universe history --config config.yaml

# Show history for specific symbol
python -m src.main universe history --symbol BTCUSDT --config config.yaml
```

### Disabling Dynamic Universe

To use a fixed symbol list instead of dynamic universe, set `universe.rebalance_frequency_hours: 0` and configure symbols in `exchange.symbols`.

### Universe Parameter Optimization

The bot includes a comprehensive optimization framework for universe-selection parameters. This allows you to systematically test different liquidity thresholds, history requirements, warm-up periods, and hysteresis settings to find robust configurations.

#### Running Universe Optimization

```bash
# Optimize universe parameters over a historical period
python -m src.main optimize-universe \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --n-combinations 200 \
  --method random \
  --output results/universe_optimization.json

# Use grid search instead (smaller, more exhaustive)
python -m src.main optimize-universe \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --n-combinations 100 \
  --method grid \
  --output results/universe_grid_search.json
```

#### What It Does

1. **Tests Parameter Combinations**: Randomly or via grid search, tests different combinations of:
   - Volume thresholds (entry/exit with hysteresis)
   - History requirements (min days, warm-up periods)
   - Stability parameters (min time in universe, max turnover)
   - Volatility filters

2. **Time-Respecting Backtests**: For each parameter set:
   - Builds historical universe at each date (only using data available up to that date)
   - Respects warm-up periods for new listings
   - Handles delistings properly
   - Runs strategy backtest using that dynamic universe

3. **Evaluates Performance**: Calculates:
   - Strategy performance (return, Sharpe, drawdown, win rate, profit factor)
   - Universe quality (size, liquidity, turnover, longevity)
   - Robustness metrics (consistency across periods)

4. **Selects Best Configs**: Ranks parameter sets by composite score (balanced Sharpe, return, drawdown) and applies constraints:
   - Minimum universe size (10 symbols)
   - Maximum universe size (100 symbols)
   - Maximum turnover rate (50%)
   - Maximum drawdown (-30%)
   - Minimum trades (50)
   - Minimum win rate (35%)

5. **Outputs Recommendations**: Provides top 5 configurations with:
   - Performance metrics
   - Universe quality metrics
   - Ready-to-use `config.yaml` blocks

#### Interpreting Results

The optimizer outputs top 5 configurations with labels:
- **Primary Recommended**: Best composite score passing all constraints
- **Conservative Alternative**: Higher volume thresholds, longer warm-up, lower risk
- **Aggressive Alternative**: Lower volume thresholds, shorter warm-up, higher expected returns

For each configuration, review:
- **Composite Score**: Higher is better (balanced Sharpe, return, drawdown)
- **Strategy Performance**: Return, Sharpe, max drawdown, win rate
- **Universe Quality**: Avg size, liquidity, turnover rate
- **Parameter Values**: Volume thresholds, history requirements, etc.

#### Using Optimized Parameters

Copy the recommended `config.yaml` block from the optimizer output into your `config.yaml` under the `universe:` section. Always:

1. **Review parameters** before applying
2. **Test in backtest/paper mode** with new parameters
3. **Monitor performance** closely after applying
4. **Keep conservative** if uncertain (higher volume thresholds, longer warm-up)

#### Important Notes

- **No Guarantee of Future Performance**: Optimized parameters are based on historical data and may not work in future market conditions
- **Avoid Overfitting**: The optimizer applies constraints to reduce overfitting, but always review results critically
- **Data Requirements**: Need sufficient historical data (6-12+ months) for meaningful optimization
- **Computational Cost**: Can take hours for 200+ combinations depending on data size

## Configuration

### Config File (`config.yaml`)

Edit `config.yaml` to customize:
- Exchange settings (symbols, timeframe)
- Strategy parameters (MA periods, momentum lookback, etc.)
- Risk limits (per-trade risk, max leverage, daily loss caps)
- Optimizer settings
- Reporting preferences

See `config.example.yaml` for detailed documentation of each field.

### Environment Variables (`.env`)

Store secrets in `.env`:
- `BYBIT_API_KEY`: Your Bybit API key
- `BYBIT_API_SECRET`: Your Bybit API secret
- `DISCORD_WEBHOOK_URL`: Discord webhook URL for daily reports

**Important**: The `.env` file contains sensitive information. Never commit it to version control.

### Trading Modes

Set `exchange.mode` in `config.yaml`:
- **`paper`**: Simulates trading without real orders (uses live market data)
- **`testnet`**: Uses Bybit testnet (requires testnet API keys)
- **`live`**: Real trading on Bybit mainnet (USE WITH CAUTION)

**Default is `testnet`** - explicitly set to `live` only when ready.

## Usage

### Command Line Interface

The bot provides several CLI commands:

#### Live Trading
```bash
# Start live trading bot
python -m src.main live --config config.yaml

# Or using systemd service (after installation)
sudo systemctl start bybit-bot
sudo systemctl status bybit-bot
sudo journalctl -u bybit-bot -f  # Follow logs
```

#### Backtesting
```bash
# Run backtest on all configured symbols
python -m src.main backtest --config config.yaml

# Backtest specific symbols
python -m src.main backtest --config config.yaml --symbols BTCUSDT ETHUSDT

# Save results to file
python -m src.main backtest --config config.yaml --output backtest_results.json
```

#### Strategy Parameter Optimization
```bash
# Run optimization to find best strategy parameters
python -m src.main optimize --config config.yaml
```

- The optimizer uses a walk-forward procedure with **in-sample (IS)** and **out-of-sample (OOS)** windows.
- Parameter sets must satisfy trade/Sharpe/drawdown constraints in both IS and OOS (with slightly looser thresholds for OOS).
- Ranking emphasizes OOS performance to reduce overfitting.

#### Universe Parameter Optimization
```bash
# Optimize universe selection parameters
python -m src.main optimize-universe \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --n-combinations 200 \
  --method random \
  --output results/universe_opt.json
```

- For each universe parameter set, a **time-respecting universe history** is built.
- If there is enough data, the backtest period is split into IS and OOS segments; OOS Sharpe/drawdown/trades are the primary metrics.
- The optimizer filters out universes that are too small/large, too unstable, or fail OOS performance constraints, and ranks the rest by a composite score that emphasizes OOS Sharpe and drawdown.

#### Daily Report
```bash
# Send daily Discord report manually
python -m src.main report --config config.yaml
```

### Systemd Services

After installation, the bot runs as systemd services:

```bash
# Main trading bot service
sudo systemctl enable bybit-bot
sudo systemctl start bybit-bot
sudo systemctl status bybit-bot

# Daily optimizer (runs once per day at 8 AM UTC)
sudo systemctl enable bybit-bot-optimizer.timer
sudo systemctl start bybit-bot-optimizer.timer

# Daily Discord report (runs once per day at 9 AM UTC)
sudo systemctl enable bybit-bot-report.timer
sudo systemctl start bybit-bot-report.timer

# View logs
sudo journalctl -u bybit-bot -f
sudo journalctl -u bybit-bot-optimizer -f
sudo journalctl -u bybit-bot-report -f
```

## Deployment Phases

### Phase 1: Backtesting

Test strategies on historical data:

```bash
python -m src.main backtest --config config.yaml
```

Review metrics:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

**Do not proceed to live trading until you're satisfied with backtest results.**

### Phase 2: Paper Trading / Testnet

Test with simulated or testnet trading:

1. Set `exchange.mode: paper` or `exchange.mode: testnet` in `config.yaml`
2. Start the bot: `python -m src.main live --config config.yaml`
3. Monitor logs and behavior
4. Review performance for at least 1-2 weeks

**Only proceed to live trading after successful paper/testnet testing.**

### Phase 3: Live Trading (Production)

1. Review all configuration carefully
2. Set conservative risk limits (low per-trade risk, low max leverage)
3. Set `exchange.mode: live` in `config.yaml` (requires explicit change)
4. Start with small capital allocation
5. Monitor closely for first days/weeks
6. Gradually increase allocation if performance is satisfactory

**Always monitor performance and be ready to stop trading if necessary.**

## Daily Discord Report

The bot sends a daily Discord report including:
- Current equity and PnL (daily, weekly, monthly)
- Open positions with unrealized PnL
- Key risk metrics (leverage, concentration)
- Strategy performance breakdown
- Recent parameter changes from optimizer
- Critical errors or incidents

Configure the Discord webhook URL in `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

## Self-Improvement Loop

The bot includes a daily optimization routine:

1. **Data Collection**: Gathers recent historical data (default: last 6 months)
2. **Parameter Search**: Tests parameter combinations using random/grid search
3. **Walk-Forward Analysis**: Validates on multiple time windows to reduce overfitting
4. **Evaluation**: Checks metrics (Sharpe, drawdown, trade count)
5. **Parameter Update**: Updates config if new parameters outperform current ones

The optimizer runs daily via systemd timer. Review optimization results regularly.

## Risk Management Details

### Position Sizing

- **Base Size**: Calculated from per-trade risk % and stop-loss distance
- **ATR Adjustment**: Scales down in high volatility, up in low volatility (capped)
- **Funding Bias**: Adjusts size based on funding rates
- **Fractional Kelly**: Applies fractional Kelly criterion (default: 0.5)

### Portfolio Limits

- **Max Leverage**: Total notional / equity (default: 3x)
- **Max Symbol Concentration**: % of equity in single symbol (default: 30%)
- **Max Positions**: Number of concurrent positions (default: 5)

### Daily Loss Limits

- **Soft Limit**: Reduces risk when daily loss reaches threshold (default: -2%)
- **Hard Limit**: Stops all trading when daily loss exceeds threshold (default: -4%)

## Troubleshooting

### Bot won't start

- Check logs: `sudo journalctl -u bybit-bot -n 100`
- Verify API keys in `.env`
- Check config file: `python -m src.main live --config config.yaml --dry-run` (if implemented)
- Verify exchange.mode is set correctly

### No trades being placed

- Check if daily loss limits are reached
- Verify position sizing calculations
- Check if symbols have sufficient data
- Review risk limit logs

### API errors

- Verify API keys are correct
- Check API key permissions (needs trading, not withdrawal)
- Check rate limits (bot includes rate limiting)
- Verify network connectivity

### Optimizer not updating parameters

- Check optimizer logs: `sudo journalctl -u bybit-bot-optimizer -n 100`
- Verify sufficient historical data in database
- Check optimizer criteria (min Sharpe, max drawdown, etc.)

## File Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main entrypoint
│   ├── config.py                # Configuration management
│   ├── logging_utils.py         # Logging setup
│   ├── exchange/
│   │   ├── __init__.py
│   │   └── bybit_client.py      # Bybit API wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ohlcv_store.py       # SQLite data storage
│   │   └── downloader.py        # Data downloader
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── trend.py             # Trend-following signals
│   │   ├── cross_sectional.py   # Cross-sectional momentum
│   │   └── funding_carry.py     # Funding-rate bias
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizing.py   # Position sizing logic
│   │   └── portfolio_limits.py  # Portfolio risk limits
│   ├── execution/
│   │   ├── __init__.py
│   │   └── executor.py          # Order execution
│   ├── state/
│   │   ├── __init__.py
│   │   └── portfolio.py         # Portfolio state tracking
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py        # Vectorized backtester
│   ├── optimizer/
│   │   ├── __init__.py
│   │   └── optimizer.py         # Parameter optimization
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── discord_reporter.py  # Discord webhook reporter
│   └── cli/
│       ├── __init__.py
│       └── main.py              # CLI commands
├── config.example.yaml          # Example configuration
├── requirements.txt             # Python dependencies
├── install.sh                   # Installation script
└── README.md                    # This file
```

## Dependencies

- `ccxt`: Cryptocurrency exchange library (Bybit support)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `pyyaml`: YAML configuration parsing
- `requests`: HTTP requests (Discord webhook)
- `python-dotenv`: Environment variable management

## Contributing

This is a research/educational project. Contributions are welcome but please:
- Test thoroughly before submitting
- Document changes clearly
- Maintain backward compatibility where possible
- Follow existing code style

## License

This project is provided as-is for educational and research purposes. Use at your own risk.

## Disclaimer

**This software is provided "as is" without warranty of any kind.** Trading cryptocurrencies carries substantial risk. The authors and contributors are not responsible for any financial losses incurred from using this software. Always:

- Test thoroughly in paper/testnet mode
- Start with small capital allocations
- Monitor performance closely
- Understand the strategies and risks involved
- Only trade with capital you can afford to lose

Past performance does not guarantee future results. Markets change, and strategies that worked in backtests or historical periods may not work in the future.

## Support

For issues and questions:
1. Check logs: `sudo journalctl -u bybit-bot -n 100`
2. Review configuration files
3. Test in backtest/paper mode first
4. Consult Bybit API documentation for exchange-specific issues

## Changelog

### Version 1.1.0 (Universe Optimization)

- Dynamic universe selection system
- Universe parameter optimization framework
- Time-respecting backtesting for universe selection
- Comprehensive universe quality metrics
- CLI commands for universe management (`universe build`, `universe show`, `universe history`)

### Version 1.0.0 (Initial Release)

- Multi-strategy system (trend, cross-sectional, funding bias)
- Comprehensive risk management
- Daily parameter optimization
- Discord reporting
- Paper/testnet/live modes
- CLI interface
- Systemd integration

