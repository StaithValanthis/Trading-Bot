# Symbol Selection Analysis: Are BTC/ETH/SOL/BNB/ADA the Only Tradable Symbols?

**Date:** 2025-11-20  
**Question:** Does the bot only trade BTC/ETH/SOL/BNB/ADA, or is there a dynamic universe selector that's being underused?

---

## Section A – How Symbols Are Currently Selected (All Mechanisms)

### 1. Configuration: `config.yaml` / `config.example.yaml`

**Location:** `config.example.yaml` lines 17-22  
**Type:** Static list (default)

```yaml
exchange:
  symbols:
    - BTCUSDT
    - ETHUSDT
    - SOLUSDT
    - BNBUSDT
    - ADAUSDT
```

**Behavior:**
- Default symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, ADAUSDT (5 symbols)
- Used as fallback if dynamic universe is disabled
- Also used by downloader ALWAYS (see below)

**Code Reference:**
- `src/config.py:35`: `symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])`
- Default in code is only 2 symbols, but config.example.yaml has 5

---

### 2. Dynamic Universe Selector: `UniverseSelector`

**Location:** `src/universe/selector.py`  
**Type:** Dynamic (based on volume, history, filters)

**How It Works:**
1. `fetch_all_symbols()`: Fetches ALL USDT perpetuals from Bybit exchange (line 46-85)
   - Can return 100+ symbols depending on exchange listings
   - Filters for: `type == 'swap'`, `settle == 'USDT'`, `active == True`

2. `build_universe()`: Applies filters to select tradable symbols (line ~200+)
   - **Liquidity filters:** Min 24h volume (entry/exit thresholds with hysteresis)
   - **History filters:** Min history days, warm-up period
   - **Volatility filters:** Max realized volatility
   - **Override lists:** `include_list` (always included), `exclude_list` (always excluded)
   - Returns filtered set of symbols (could be 10-50+ symbols depending on filters)

3. `get_universe()`: Returns current universe from `UniverseStore` (database)

**When Used:**
- Live trading: **ONLY if `config.universe.rebalance_frequency_hours > 0`** (line 354 in `src/cli/main.py`)
- Universe commands: `python -m src.main universe build/show/history`

**Code Reference:**
- `src/cli/main.py:354`: Checks if dynamic universe is enabled
- `src/cli/main.py:356-362`: Initializes `UniverseSelector` if enabled
- `src/cli/main.py:544`: Gets symbols from `universe_selector.get_universe()`

---

### 3. Data Downloader: `DataDownloader.update_all_symbols()`

**Location:** `src/data/downloader.py:83-107`  
**Type:** Parameter-based (accepts symbols list)

**How It Works:**
```python
def update_all_symbols(
    self,
    symbols: List[str],  # Symbols passed in
    timeframe: str,
    lookback_days: int = 30
):
```

**Current Call Site (Live Loop):**
```python
# src/cli/main.py:512-516
downloader.update_all_symbols(
    config.exchange.symbols,  # <-- ALWAYS uses config.exchange.symbols (5 symbols)
    config.exchange.timeframe,
    lookback_days=30
)
```

**Key Finding:**
- **NO hardcoded symbols in downloader itself** - it's flexible
- **BUT always called with `config.exchange.symbols`** in live loop
- This means **downloader only fetches data for the 5 default symbols**, regardless of dynamic universe!

**Code Reference:**
- `src/data/downloader.py:83-107`: Method definition
- `src/cli/main.py:512`: Call site - **BUG HERE!**

---

### 4. Live Trading Loop: `run_live()`

**Location:** `src/cli/main.py:512-565`  
**Type:** Conditional (dynamic if enabled, else static)

**How It Chooses Symbols:**

1. **Data Downloading (line 512-516):**
   ```python
   downloader.update_all_symbols(
       config.exchange.symbols,  # ALWAYS the 5 symbols
       config.exchange.timeframe,
       lookback_days=30
   )
   ```
   - **Always** downloads for `config.exchange.symbols` (5 symbols)
   - **Regardless** of whether dynamic universe is enabled

2. **Signal Generation (line 541-565):**
   ```python
   if universe_selector is not None:
       # Dynamic universe enabled
       trading_symbols = list(universe_selector.get_universe())  # Could be 10-50+ symbols
       # OR rebuild if needed:
       trading_symbols, changes = universe_selector.build_universe(timeframe)
   else:
       # Fixed list
       trading_symbols = config.exchange.symbols  # 5 symbols
   ```

**The Problem:**
- If dynamic universe is enabled, `trading_symbols` could be 20+ symbols
- But downloader only fetched data for 5 symbols
- Signals for symbols NOT in `config.exchange.symbols` will fail because no data!

**Code Reference:**
- `src/cli/main.py:512`: Downloader call (always uses config.exchange.symbols)
- `src/cli/main.py:541-565`: Symbol selection for trading (dynamic or static)

---

### 5. Backtester: `Backtester.backtest()`

**Location:** `src/backtest/backtester.py:81`  
**Type:** Parameter-based (accepts symbol_data dict)

**How It Works:**
```python
def backtest(
    self,
    symbol_data: Dict[str, pd.DataFrame],  # Symbols passed in via data
    ...
)
```

**Call Site:**
```python
# src/cli/main.py:787
test_symbols = symbols or config.exchange.symbols  # Uses CLI arg or config defaults
# Then loads data for test_symbols and passes to backtester
```

**Behavior:**
- Uses symbols provided via CLI `--symbols` argument OR `config.exchange.symbols`
- Does NOT use dynamic universe
- Limited to symbols that have data in the database

**Code Reference:**
- `src/cli/main.py:787`: Symbol selection for backtest
- `src/cli/main.py:790`: Loads data for selected symbols

---

### 6. Optimizer: `Optimizer.optimize()`

**Location:** `src/optimizer/optimizer.py:34`  
**Type:** Parameter-based (accepts symbols list)

**How It Works:**
```python
def optimize(
    self,
    symbols: List[str],  # Symbols passed in
    timeframe: str
):
```

**Call Site:**
```python
# src/cli/main.py:906
result = optimizer.optimize(config.exchange.symbols, config.exchange.timeframe)
```

**Behavior:**
- **Always** uses `config.exchange.symbols` (5 symbols)
- Does NOT use dynamic universe
- Limited to symbols with data in database

**Code Reference:**
- `src/cli/main.py:906`: Optimizer call (always uses config.exchange.symbols)

---

### 7. Universe Optimizer: `UniverseOptimizer`

**Location:** `src/optimizer/universe_optimizer.py`  
**Type:** Parameter-based (but can load all symbols from DB)

**How It Works:**
```python
# src/cli/main.py:979-985
symbols_to_load = config.exchange.symbols if config.exchange.symbols else []
if not symbols_to_load:
    # Load all symbols from database
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ?", (timeframe,))
    symbols_to_load = [row[0] for row in cursor.fetchall()]
```

**Behavior:**
- If `config.exchange.symbols` is empty, loads ALL symbols from database
- Otherwise uses `config.exchange.symbols`
- This is the ONLY component that can use all available symbols (if data exists)

**Code Reference:**
- `src/cli/main.py:979-985`: Symbol selection for universe optimizer

---

## Section B – Are BTC/ETH/SOL/BNB/ADA the Only Tradable Symbols?

### **ANSWER: It Depends on Configuration**

### Scenario 1: Dynamic Universe DISABLED (`config.universe.rebalance_frequency_hours = 0`)

**Answer: YES** - Only the 5 symbols (or whatever is in `config.exchange.symbols`) will be traded.

**Code Path:**
1. Line 354: `if config.universe.rebalance_frequency_hours > 0:` → **FALSE**
2. Line 565: `trading_symbols = config.exchange.symbols` → Uses 5 symbols
3. Line 512: Downloader also uses `config.exchange.symbols` → Downloads 5 symbols
4. **Result:** Everything is consistent, bot trades exactly those 5 symbols

### Scenario 2: Dynamic Universe ENABLED (`config.universe.rebalance_frequency_hours > 0`)

**Answer: PARTIALLY** - Bot **attempts** to trade symbols from dynamic universe, but **fails silently** for symbols not in `config.exchange.symbols` because:

1. **Downloader** (line 512) **always** downloads data for `config.exchange.symbols` (5 symbols)
2. **Trading loop** (line 544/555) gets symbols from `universe_selector.get_universe()` (could be 20+ symbols)
3. **Signal generation** (line 571) tries to generate signals for all `trading_symbols`
4. **Problem:** For symbols NOT in `config.exchange.symbols`, `store.get_ohlcv()` returns empty DataFrame
5. **Result:** Signals fail silently (line 580: `if df.empty or len(df) < ...: continue`)

**Code Path:**
```python
# Line 512: Downloader - ALWAYS uses config.exchange.symbols
downloader.update_all_symbols(config.exchange.symbols, ...)  # Downloads 5 symbols

# Line 544: Trading symbols - Gets from dynamic universe
trading_symbols = list(universe_selector.get_universe())  # Could be 20+ symbols

# Line 571-590: Signal generation
for symbol in trading_symbols:  # Iterates over 20+ symbols
    df = store.get_ohlcv(symbol, ...)  # For 15 symbols, this returns EMPTY!
    if df.empty:  # Line 580: Skip silently
        continue
```

**Conclusion:** If dynamic universe is enabled, the bot will:
- ✅ Generate signals for symbols in `config.exchange.symbols` (5 symbols)
- ❌ Skip signals for symbols NOT in `config.exchange.symbols` (missing data)
- ❌ **Never trade symbols that aren't in the default 5**

### **Final Answer:**

**"Given the current code, these five (BTC/ETH/SOL/BNB/ADA) are / are not the only symbols that will be traded by the live bot."**

**YES** - Even with dynamic universe enabled, the bot will effectively only trade the 5 symbols because:
1. Downloader only fetches data for `config.exchange.symbols` (5 symbols)
2. Symbols not in the downloader list have no data
3. Signal generation fails silently for symbols without data
4. No trades are executed for symbols without signals

**The dynamic universe selector exists and works, but it's useless because the downloader doesn't respect it!**

---

## Section C – Inconsistencies or Design Issues

### Critical Bug #1: Downloader Doesn't Use Dynamic Universe

**Location:** `src/cli/main.py:512`  
**Issue:** Downloader always uses `config.exchange.symbols`, even when dynamic universe is enabled

**Impact:**
- Dynamic universe can select 20+ symbols
- But downloader only fetches data for 5 symbols
- 15+ symbols are selected for trading but have no data
- Signals fail silently, no trades executed

**Code:**
```python
# Line 512: ALWAYS uses config.exchange.symbols
downloader.update_all_symbols(config.exchange.symbols, ...)

# Line 544: Could return 20+ symbols
trading_symbols = list(universe_selector.get_universe())
```

### Inconsistency #2: Backtester Doesn't Use Dynamic Universe

**Location:** `src/cli/main.py:787`  
**Issue:** Backtester uses `config.exchange.symbols` or CLI args, never dynamic universe

**Impact:**
- Backtests are limited to 5 symbols (or CLI-provided list)
- Cannot backtest with dynamic universe
- Results don't reflect live trading behavior if dynamic universe is enabled

### Inconsistency #3: Optimizer Doesn't Use Dynamic Universe

**Location:** `src/cli/main.py:906`  
**Issue:** Optimizer always uses `config.exchange.symbols`

**Impact:**
- Parameter optimization limited to 5 symbols
- Cannot optimize across broader universe
- Results may be suboptimal

### Inconsistency #4: Universe Optimizer Has Special Logic

**Location:** `src/cli/main.py:979-985`  
**Issue:** Universe optimizer can load all symbols from DB, but other components cannot

**Impact:**
- Inconsistent behavior between components
- Universe optimizer works differently than live trading

### Design Issue #5: Silent Failures

**Location:** `src/cli/main.py:580`  
**Issue:** Signal generation silently skips symbols without data

**Impact:**
- No warning when symbols from dynamic universe lack data
- Bot appears to work but isn't trading the full universe
- Difficult to debug why certain symbols aren't traded

---

## Section D – Recommended Changes

### Fix #1: Make Downloader Use Dynamic Universe (CRITICAL)

**File:** `src/cli/main.py`  
**Lines:** 512-516

**Current Code:**
```python
downloader.update_all_symbols(
    config.exchange.symbols,  # Always uses 5 symbols
    config.exchange.timeframe,
    lookback_days=30
)
```

**Fixed Code:**
```python
# Determine which symbols to download
if universe_selector is not None:
    # Download for current universe + potential new symbols
    # Get current universe
    current_universe = list(universe_selector.get_universe())
    # Also fetch all symbols from exchange to check for new listings
    all_exchange_symbols = universe_selector.fetch_all_symbols()
    # Download for all symbols that might enter universe
    symbols_to_download = list(set(current_universe + all_exchange_symbols[:50]))  # Limit to top 50 to avoid excessive API calls
else:
    # Fixed list mode
    symbols_to_download = config.exchange.symbols

downloader.update_all_symbols(
    symbols_to_download,
    config.exchange.timeframe,
    lookback_days=30
)
```

**Impact:** Downloader now respects dynamic universe, ensures data exists for all tradable symbols.

---

### Fix #2: Add Warning for Missing Data

**File:** `src/cli/main.py`  
**Lines:** 571-590

**Current Code:**
```python
for symbol in trading_symbols:
    try:
        df = store.get_ohlcv(symbol, ...)
        if df.empty or len(df) < config.strategy.trend.ma_long:
            continue  # Silent skip
```

**Fixed Code:**
```python
for symbol in trading_symbols:
    try:
        df = store.get_ohlcv(symbol, ...)
        if df.empty:
            logger.warning(
                f"Symbol {symbol} in trading universe but no data available. "
                f"Ensure downloader has fetched data for all universe symbols."
            )
            continue
        if len(df) < config.strategy.trend.ma_long:
            logger.debug(f"Symbol {symbol} has insufficient data: {len(df)} bars (need {config.strategy.trend.ma_long})")
            continue
```

**Impact:** Makes it obvious when symbols are skipped due to missing data.

---

### Fix #3: Unify Symbol Selection Across Components

**Create a helper function:**

**File:** `src/cli/main.py`  
**Location:** Before `run_live()` function

**New Function:**
```python
def get_symbols_for_download(
    config: BotConfig,
    universe_selector: Optional[UniverseSelector] = None
) -> List[str]:
    """
    Get list of symbols that should have data downloaded.
    
    If dynamic universe is enabled, returns current universe + potential new symbols.
    Otherwise returns config.exchange.symbols.
    
    Args:
        config: Bot configuration
        universe_selector: Optional universe selector (if dynamic universe enabled)
    
    Returns:
        List of symbols to download data for
    """
    if universe_selector is not None:
        # Dynamic universe mode
        current_universe = list(universe_selector.get_universe())
        # Also include all exchange symbols (limited to top 100 for performance)
        all_symbols = universe_selector.fetch_all_symbols()
        # Combine and deduplicate
        symbols = list(set(current_universe + all_symbols[:100]))
        return symbols
    else:
        # Fixed list mode
        return config.exchange.symbols
```

**Update Downloader Call:**
```python
# Line 512: Use unified function
symbols_to_download = get_symbols_for_download(config, universe_selector)
downloader.update_all_symbols(
    symbols_to_download,
    config.exchange.timeframe,
    lookback_days=30
)
```

**Impact:** Centralizes symbol selection logic, ensures consistency.

---

### Fix #4: Add Universe Pre-Download Step

**File:** `src/cli/main.py`  
**Location:** After universe selector initialization (around line 362)

**New Code:**
```python
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
    
    # Pre-download data for initial universe
    logger.info("Pre-downloading data for initial universe...")
    initial_universe = list(universe_selector.get_universe())
    if not initial_universe:
        # Build universe if empty
        logger.info("Universe is empty, building initial universe...")
        initial_universe, _ = universe_selector.build_universe(config.exchange.timeframe)
    
    # Download data for all symbols in universe
    downloader.update_all_symbols(
        initial_universe,
        config.exchange.timeframe,
        lookback_days=config.data.lookback_days if hasattr(config.data, 'lookback_days') else 30
    )
    logger.info(f"Pre-downloaded data for {len(initial_universe)} symbols in universe")
```

**Impact:** Ensures data exists for initial universe before trading starts.

---

### Fix #5: Update Backtester to Support Dynamic Universe

**File:** `src/cli/main.py`  
**Function:** `run_backtest()` (around line 763)

**Current Code:**
```python
test_symbols = symbols or config.exchange.symbols
```

**Fixed Code:**
```python
# If dynamic universe is enabled, use it (with historical reconstruction if needed)
if config.universe.rebalance_frequency_hours > 0:
    # For backtest, we'd need historical universe construction
    # For now, fall back to config symbols or use universe from latest date
    universe_store = UniverseStore(config.data.db_path)
    latest_universe = universe_store.get_current_universe()
    test_symbols = symbols or list(latest_universe) or config.exchange.symbols
else:
    test_symbols = symbols or config.exchange.symbols
```

**Impact:** Backtests can use dynamic universe if configured.

---

### Fix #6: Update Optimizer to Support Dynamic Universe

**File:** `src/cli/main.py`  
**Function:** `run_optimize()` (around line 881)

**Current Code:**
```python
result = optimizer.optimize(config.exchange.symbols, config.exchange.timeframe)
```

**Fixed Code:**
```python
# Determine symbols for optimization
if config.universe.rebalance_frequency_hours > 0:
    universe_store = UniverseStore(config.data.db_path)
    latest_universe = universe_store.get_current_universe()
    optimize_symbols = list(latest_universe) or config.exchange.symbols
else:
    optimize_symbols = config.exchange.symbols

result = optimizer.optimize(optimize_symbols, config.exchange.timeframe)
```

**Impact:** Optimizer can use dynamic universe if configured.

---

## Summary of Changes

### Priority 1 (Critical):
1. ✅ **Fix downloader to use dynamic universe** (Fix #1) - Without this, dynamic universe is broken
2. ✅ **Add warning for missing data** (Fix #2) - Helps debug issues

### Priority 2 (Important):
3. ✅ **Unify symbol selection** (Fix #3) - Prevents future inconsistencies
4. ✅ **Pre-download universe data** (Fix #4) - Ensures data exists at startup

### Priority 3 (Nice to Have):
5. ✅ **Update backtester for dynamic universe** (Fix #5) - Makes backtests match live
6. ✅ **Update optimizer for dynamic universe** (Fix #6) - Makes optimization more realistic

---

## Implementation Checklist

- [ ] Update `src/cli/main.py` downloader call to use dynamic universe
- [ ] Add warning logging for missing data
- [ ] Create `get_symbols_for_download()` helper function
- [ ] Add universe pre-download step at startup
- [ ] Update backtester symbol selection
- [ ] Update optimizer symbol selection
- [ ] Test with dynamic universe enabled
- [ ] Test with dynamic universe disabled (should still work)
- [ ] Update documentation

---

**Generated:** 2025-11-20  
**Status:** Analysis complete, fixes proposed

