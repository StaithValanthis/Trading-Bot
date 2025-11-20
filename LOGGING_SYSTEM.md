# Logging System Documentation

## Overview

The bot now uses a centralized, production-grade logging system that:
- Configures the **root logger** so all loggers inherit handlers automatically
- Writes to both **files** (rotating) and **stderr** (for systemd/journald)
- Uses **UTC timestamps** consistently
- **Redacts secrets** (API keys, webhooks) from logs
- Ensures **immediate log flushing** for systemd

## Root Causes of Previous Issues

### 1. Root Logger Not Configured
**Problem**: Only a named logger "trading_bot" was configured, but modules use `get_logger(__name__)` which creates loggers with different names (e.g., "src.cli.main"). These loggers didn't inherit handlers.

**Fix**: Now configures the **root logger** so all child loggers automatically inherit handlers.

### 2. Logging Setup Order
**Problem**: `setup_logging()` was called AFTER `get_logger(__name__)`, so early log messages were lost or logged to a logger without handlers.

**Fix**: All CLI functions now:
1. Load config first
2. Call `setup_logging(config)` to configure root logger
3. Then get logger (which inherits from root)

### 3. No Propagation
**Problem**: Logger propagation wasn't explicitly controlled, causing inconsistent behavior.

**Fix**: Root logger has `propagate=False` and handles all logs directly. Child loggers propagate to root.

### 4. UTC Timestamps Missing
**Problem**: Logs used local time, making it hard to correlate events across servers.

**Fix**: Custom `UTCFormatter` class uses `time.gmtime` for UTC timestamps.

### 5. Systemd Buffering
**Problem**: stdout buffering caused logs to appear delayed in journald.

**Fix**: 
- Use **stderr** instead of stdout (unbuffered by default)
- Explicitly set `line_buffering=True`
- Force flush after critical log statements

## Logging Architecture

### File Structure

```
src/logging_utils.py          # Centralized logging configuration
  ├── setup_logging()         # Main configuration function
  ├── get_logger()            # Logger factory
  ├── UTCFormatter            # UTC timestamp formatter
  ├── SecretFilter            # Redacts secrets from logs
  └── flush_logs()            # Force flush all handlers
```

### Log Files

Logs are written to the `logs/` directory (configurable via `config.yaml`):
- `logs/bot-live.log` - Live trading bot
- `logs/bot-optimizer.log` - Parameter optimizer
- `logs/bot-report.log` - Daily Discord reports
- `logs/bot-universe.log` - Universe builder
- `logs/bot-backtest.log` - Backtests
- `logs/bot-health.log` - Health checks

Each log file:
- Rotates when it reaches `max_log_size_mb` (default: 10MB)
- Keeps `backup_count` backups (default: 5)
- Uses UTC timestamps
- Contains DEBUG level and above

### Console Logs (journald)

Console logs go to **stderr** and appear in `journalctl`:
- Level: INFO and above (DEBUG only in files)
- Format: `%(asctime)s [%(levelname)8s] %(name)s - %(message)s`
- UTC timestamps
- Secrets automatically redacted

## Configuration

### Config File (`config.yaml`)

```yaml
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_dir: logs            # Directory for log files
  max_log_size_mb: 10      # Rotate when file reaches this size
  backup_count: 5          # Number of backup files to keep
```

### Environment Variables

No environment variables needed for logging (all configured via YAML).

## Usage

### In Code

```python
from src.logging_utils import get_logger

logger = get_logger(__name__)  # Automatically inherits from root logger

logger.debug("Detailed debugging info")
logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)  # Include stack trace
logger.critical("Critical error")
```

### Viewing Logs

#### File Logs

```bash
# View live bot logs
tail -f logs/bot-live.log

# View all logs
tail -f logs/*.log

# Search logs
grep "ERROR" logs/bot-live.log
```

#### Systemd/Journald

```bash
# View live bot service logs
sudo journalctl -u bybit-bot.service -f

# View last 100 lines
sudo journalctl -u bybit-bot.service -n 100

# View logs since boot
sudo journalctl -u bybit-bot.service -b

# View logs with timestamps
sudo journalctl -u bybit-bot.service --since "2024-01-01 00:00:00"

# View logs at ERROR level and above
sudo journalctl -u bybit-bot.service -p err
```

#### All Services

```bash
# View all bot service logs
sudo journalctl -u bybit-bot* -f

# View optimizer logs
sudo journalctl -u bybit-bot-optimizer.service -f

# View report logs
sudo journalctl -u bybit-bot-report.service -f
```

## Log Format

### Console/Journald Format

```
2024-01-01 12:00:00 UTC [    INFO] src.cli.main - Starting live trading bot
2024-01-01 12:00:01 UTC [    INFO] src.exchange.bybit_client - Initializing Bybit client
2024-01-01 12:00:02 UTC [   ERROR] src.exchange.bybit_client - Connection failed: ...
```

### File Format

```
2024-01-01 12:00:00 UTC [    INFO] src.cli.main:157 - run_live() - Starting live trading bot
2024-01-01 12:00:01 UTC [   DEBUG] src.exchange.bybit_client:45 - __init__() - Creating CCXT exchange instance
```

File format includes:
- Timestamp (UTC)
- Level
- Logger name (module path)
- Line number
- Function name
- Message

## Secret Redaction

The `SecretFilter` automatically redacts:
- API keys (patterns like `api_key="..."`, `BYBIT_API_KEY=...`)
- API secrets (patterns like `api_secret="..."`, `secret="..."`)
- Webhook URLs (patterns like `webhook_url="https://..."`)
- Long alphanumeric strings (likely keys/secrets)

**Example:**
```python
logger.info(f"API key: {api_key}")  # Logs: "API key: REDACTED"
```

## Instrumentation Points

The bot logs at key points:

### Live Trading Loop
- ✅ Bot startup/shutdown
- ✅ Config loading and validation
- ✅ Exchange connection and credentials
- ✅ Portfolio state updates
- ✅ Rebalance decisions
- ✅ Signal generation
- ✅ Order execution
- ✅ Risk limit checks
- ✅ Daily loss cap triggers
- ✅ Position reconciliation

### Exchange Wrapper
- ✅ Connection attempts
- ✅ API errors (rate limits, network errors)
- ✅ Retries and circuit breakers
- ✅ Order placement and fills

### Risk Management
- ✅ Trade blocks (leverage, concentration, etc.)
- ✅ Daily loss cap breaches
- ✅ Position sizing calculations

### Optimization
- ✅ Parameter search progress
- ✅ Backtest results
- ✅ Best parameter selection

### Reporting
- ✅ Discord report generation
- ✅ Report send status

## Troubleshooting

### Logs Not Appearing

1. **Check log directory exists**:
   ```bash
   ls -la logs/
   ```

2. **Check file permissions**:
   ```bash
   ls -la logs/*.log
   ```
   The bot user should have write access.

3. **Check systemd service status**:
   ```bash
   sudo systemctl status bybit-bot.service
   ```

4. **Check journald logs**:
   ```bash
   sudo journalctl -u bybit-bot.service -n 50
   ```

5. **Verify config file**:
   ```bash
   cat config.yaml | grep -A 5 logging
   ```

### Too Many/Few Logs

Adjust `config.logging.level` in `config.yaml`:
- `DEBUG` - Very verbose (all logs)
- `INFO` - Normal operation (recommended)
- `WARNING` - Warnings and errors only
- `ERROR` - Errors only

### Log Files Growing Too Large

Reduce `config.logging.max_log_size_mb` or `backup_count` in `config.yaml`.

### Logs Missing in Journald

Ensure systemd services have:
```ini
StandardOutput=journal
StandardError=journal
```

## Best Practices

1. **Use appropriate log levels**:
   - `DEBUG` - Detailed debugging info
   - `INFO` - Normal operation events
   - `WARNING` - Recoverable issues
   - `ERROR` - Errors requiring attention
   - `CRITICAL` - System failures

2. **Include context**:
   ```python
   logger.info(f"Order placed: {symbol} {side} {size} @ {price}")
   ```

3. **Use exc_info for exceptions**:
   ```python
   try:
       do_something()
   except Exception as e:
       logger.error(f"Failed to do something: {e}", exc_info=True)
   ```

4. **Don't log secrets**:
   - Use `SecretFilter` (automatic)
   - Don't include API keys in log messages

5. **Flush critical logs**:
   ```python
   logger.critical("System shutting down")
   from src.logging_utils import flush_logs
   flush_logs()
   ```

## Testing Logging

```bash
# Test logging setup
python -c "from src.logging_utils import setup_logging, get_logger; setup_logging(); logger = get_logger('test'); logger.info('Test log message'); logger.error('Test error')"

# Verify logs appear
tail -f logs/bot.log
```

## Migration Notes

If you're upgrading from the old logging system:
- All existing log files will continue to work
- New logs will use the improved format
- No code changes needed (all modules use `get_logger(__name__)`)

