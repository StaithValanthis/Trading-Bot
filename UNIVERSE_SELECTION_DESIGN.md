# Universe Selection Design Document

## 1. Research Summary: Symbol/Universe Selection Practices

### Key Findings

#### A. Common Practices in Crypto Futures & Quantitative Trading

1. **Liquidity Filters (Universal)**
   - **Minimum 24h Volume**: Most systematic strategies require minimum notional volume (e.g., $5M-50M USDT per day) to ensure:
     - Sufficient market depth for entry/exit
     - Low slippage costs
     - Stable price discovery
   - **Open Interest**: Some strategies require minimum OI as a proxy for market depth and stability
   - **Bid-Ask Spread**: Maximum spread thresholds (e.g., ≤10-50 bps) to control transaction costs

2. **Data Quality Requirements**
   - **Minimum History**: Require N days/weeks of continuous OHLCV data (e.g., 30-90 days) before a symbol becomes tradable
   - **Data Gaps**: Reject symbols with large gaps (>threshold % missing) or unstable price series
   - **Recent Listings**: Implement "warm-up periods" (e.g., 14-30 days) to avoid survivorship bias and allow markets to stabilize

3. **Volatility & Risk Filters**
   - **Price Sanity**: Reject extremely low-priced symbols (dust) that cause rounding/min-notional issues
   - **Realized Volatility Caps**: Avoid symbols with pathological volatility (>threshold over a lookback period)
   - **Limit-Up/Down Frequency**: Filter out symbols with frequent limit moves (indicates illiquidity or manipulation)

4. **Universe Stability Techniques (Critical)**
   - **Hysteresis**: Separate thresholds for entering vs. staying in the universe to reduce churn
     - Example: Enter if volume ≥ 100M for 7 days; stay in if volume ≥ 70M; exit if < 70M for 7 days
   - **Minimum Time in Universe**: Once added, symbol must stay for minimum period (e.g., 7-14 days) before it can be removed (unless hard delisted)
   - **Maximum Turnover**: Limit how many symbols can enter/exit per rebalance (e.g., max 20% of universe size)

5. **Bybit-Specific Considerations**
   - **Category Filtering**: Only `category='linear'` for USDT-margined perpetuals
   - **Status Filtering**: Only `status='Trading'` (exclude Delisting, PreLaunch, etc.)
   - **Quote Coin**: Only `quoteCoin='USDT'` to avoid USDC or other quote currencies
   - **Contract Size**: Some contracts have unusual sizes; validate min/max limits per symbol

#### B. Design Patterns

1. **Conservative vs. Aggressive Filtering**
   - **Conservative** (Recommended for retail):
     - Higher volume thresholds (e.g., $10M+ 24h volume)
     - Longer warm-up periods (30 days)
     - Stricter hysteresis (70% of entry threshold for exit)
     - Smaller universe (20-50 symbols) for easier monitoring
   - **Aggressive**:
     - Lower volume thresholds (e.g., $1M+ 24h volume)
     - Shorter warm-up periods (7-14 days)
     - Tighter hysteresis (80-90% of entry threshold)
     - Larger universe (100+ symbols) for more diversification

2. **Pitfalls to Avoid**
   - **Survivorship Bias**: Include delisted symbols in backtests by using historical universe snapshots
   - **Over-Trading from Churn**: Aggressive thresholds cause constant universe turnover → excessive trading costs
   - **Missing New Winners**: Too-conservative warm-up periods miss early momentum in new listings
   - **Correlation Blindness**: Not grouping/capping similar assets (e.g., multiple L2s, meme coins)

## 2. Universe Definition: Eligible Symbol Framework

### Core Filters

#### 1. Instrument Type & Metadata
- **Category**: Only `linear` (USDT-margined perpetuals)
- **Status**: Only `Trading` (exclude `Delisting`, `PreLaunch`, `Inverse`, etc.)
- **Quote Coin**: Only `USDT`
- **Symbol Format**: Standard format (e.g., `BTCUSDT`, `ETHUSDT`; validate consistency)

**Default**: Enforced by API filtering

#### 2. Liquidity & Market Quality

**A. Minimum 24h Notional Volume**
- **Entry Threshold**: 24h volume ≥ $10,000,000 USDT (for 7 consecutive days)
- **Exit Threshold**: 24h volume < $7,000,000 USDT (for 7 consecutive days) ← Hysteresis
- **Rationale**: Ensures sufficient liquidity for retail-sized positions without excessive slippage

**B. Optional Open Interest Filter**
- **Entry Threshold**: Open Interest ≥ $5,000,000 USDT (if available)
- **Rationale**: Proxy for market depth and stability

**C. Optional Spread Filter** (if orderbook data available)
- **Max Average Spread**: ≤ 20 bps (0.20%) over 24h window
- **Rationale**: Control transaction costs

**Default Values**:
```yaml
min_24h_volume_entry: 10_000_000  # $10M USDT
min_24h_volume_exit: 7_000_000    # $7M USDT (hysteresis)
volume_check_days: 7               # Consecutive days
min_open_interest: 5_000_000       # Optional, $5M USDT
max_spread_bps: 20                 # Optional, 20 basis points
```

#### 3. Historical Data Availability

**A. Minimum History Period**
- **Requirement**: At least 30 days of continuous OHLCV data at trading timeframe (e.g., 1h)
- **Rationale**: Enough data for indicators (MA100 needs ~100 bars = 100 hours = ~4 days, but 30 days provides robustness)

**B. Data Quality Checks**
- **Maximum Gap**: ≤ 5% missing candles over lookback period
- **Broken History**: Reject if > 10% gaps or no recent data (last update > 7 days ago)

**C. Warm-Up Period for New Listings**
- **Requirement**: Symbol must exist for at least 30 days AND pass liquidity filters for 14 consecutive days before entering universe
- **Rationale**: Avoid survivorship bias; allow markets to stabilize

**Default Values**:
```yaml
min_history_days: 30
warmup_days: 14                    # Days of passing filters before entry
max_data_gap_pct: 5.0              # Max % missing candles
max_days_since_last_update: 7
```

#### 4. Volatility & Price Sanity

**A. Price Floor**
- **Minimum Price**: ≥ $0.01 USDT (reject dust coins)
- **Rationale**: Avoid rounding/min-notional issues

**B. Volatility Caps**
- **Max Realized Volatility**: ≤ 200% annualized over 30-day lookback
- **Rationale**: Avoid pathological volatility that breaks position sizing

**C. Limit Move Frequency**
- **Max Limit-Up/Down Days**: ≤ 5% of trading days in last 30 days
- **Rationale**: Indicates manipulation or extreme illiquidity

**Default Values**:
```yaml
min_price_usdt: 0.01
max_realized_vol_pct: 200.0        # Annualized %
limit_move_frequency_pct: 5.0      # Max % of days
```

#### 5. Risk & Correlation (Simple Version)

**A. Symbol Buckets** (Optional Grouping)
- Group symbols into categories:
  - `majors`: BTCUSDT, ETHUSDT
  - `large_caps`: Top 20 by market cap
  - `mid_caps`: Rank 21-50
  - `small_caps`: Rank 51+
  - `meme`: Known meme coins (manual list)
  - `defi`: DeFi-related (manual list)
- **Soft Caps**: Max K symbols per bucket (e.g., max 10 small_caps, max 5 meme)

**B. Correlation Deduplication** (Advanced, Optional)
- If two symbols have >0.95 correlation over 90 days and similar volume, prefer the one with higher volume
- Not implemented initially; can be added later

**Default Values**:
```yaml
max_symbols_per_bucket: {}  # Optional, e.g. {meme: 5, small_caps: 10}
correlation_dedup_threshold: null  # Disabled by default
```

#### 6. Override Controls

**A. Include List**
- Always include symbols in this list (bypass filters)
- Use for core holdings (e.g., BTCUSDT, ETHUSDT) that should always be considered

**B. Exclude List (Ban List)**
- Never include symbols in this list (even if they pass all filters)
- Use for known problem symbols or temporary bans

**Default Values**:
```yaml
include_list: [BTCUSDT, ETHUSDT]  # Always included
exclude_list: []                   # Never included
```

### Recommended Default Configuration (Conservative)

```yaml
universe:
  # Liquidity filters
  min_24h_volume_entry: 10_000_000    # $10M USDT
  min_24h_volume_exit: 7_000_000      # $7M USDT (hysteresis)
  volume_check_days: 7                 # Consecutive days for entry/exit
  min_open_interest: null              # Disabled by default
  max_spread_bps: null                 # Disabled by default (requires orderbook)
  
  # Historical data
  min_history_days: 30
  warmup_days: 14                      # Warm-up period for new listings
  max_data_gap_pct: 5.0
  
  # Volatility & price
  min_price_usdt: 0.01
  max_realized_vol_pct: 200.0
  limit_move_frequency_pct: 5.0
  
  # Universe stability
  min_time_in_universe_days: 7         # Once added, stay for 7 days min
  max_turnover_per_rebalance_pct: 20.0 # Max 20% of universe can change
  
  # Rebalancing
  rebalance_frequency_hours: 24        # Recompute universe daily
  
  # Overrides
  include_list: [BTCUSDT, ETHUSDT]
  exclude_list: []
  
  # Buckets (optional)
  max_symbols_per_bucket: {}
```

## 3. Dynamic Maintenance Logic

### A. Rebalance Frequency
- **Default**: Once per day (24h interval)
- **Rationale**: Balance between staying current and avoiding excessive churn
- **Alternative**: Every 4-6 hours for more frequent updates (higher API usage)

### B. Entry & Exit Rules (Hysteresis)

**Entry Conditions** (all must be met for 7 consecutive days):
1. 24h volume ≥ entry threshold ($10M)
2. Historical data ≥ min_history_days (30 days)
3. Symbol age ≥ warmup_days (14 days of passing filters)
4. Pass all volatility/price filters
5. Not in exclude_list

**Exit Conditions** (any condition met for 7 consecutive days):
1. 24h volume < exit threshold ($7M) ← Hysteresis
2. Symbol delisted (status != 'Trading')
3. Fails volatility/price filters
4. Added to exclude_list
5. Exception: Symbol stays in universe if added < min_time_in_universe_days ago (unless hard delisted)

### C. New Listings / Warm-Up

**Process**:
1. Detect new listing (symbol appears in API but not in historical data)
2. Mark symbol as "warming up" (not eligible yet)
3. Start collecting OHLCV data
4. After `warmup_days` (14 days) AND symbol age ≥ `min_history_days` (30 days):
   - Check if symbol passes all filters for `volume_check_days` (7 days)
   - If yes, add to universe on next rebalance

**Backtest Handling**:
- Store universe membership history by date
- In backtest, use historical universe at each timestamp (not current universe)
- For new listings in historical data, respect warm-up period based on listing date

### D. Delistings / Hard Removals

**Detection**:
- Symbol status changes to `Delisting` or `Settling`
- Symbol disappears from API entirely

**Action**:
- Immediately remove from universe (bypass min_time_in_universe_days)
- Log reason as "delisted"
- In live bot: close any open positions ASAP (exit immediately)
- Alert in daily Discord report

### E. Turnover Controls

**Max Turnover Per Rebalance**:
- Limit number of symbols that can enter/exit in single rebalance
- Formula: `max_changes = max(1, int(universe_size * max_turnover_pct / 100))`
- If more symbols want to enter/exit:
  - Prioritize by volume (highest volume for entry, lowest for exit)
  - Apply changes on next rebalance

### F. Audit Trail

**Store in SQLite** (`universe_history` table):
- `date`: Date of universe snapshot
- `symbol`: Symbol name
- `action`: 'added', 'removed', or 'kept'
- `reason`: Reason for change (e.g., 'volume_above_threshold', 'delisted', 'exclude_list')
- `volume_24h`: 24h volume at time of decision
- `metadata`: JSON with additional info (OI, spread, etc.)

**Query Interface**:
- `get_universe(as_of_date)`: Get universe at specific date
- `get_history(symbol)`: Get history of symbol additions/removals
- `get_changes(start_date, end_date)`: Get all changes in date range

## 4. Backtesting & Evaluation

### Methodology

1. **Historical Universe Construction**:
   - Load broad set of symbols from historical data (including delisted ones)
   - For each date in backtest period:
     - Simulate universe construction using filters + hysteresis
     - Store universe membership
   - Use historical universe at each timestamp (not current universe)

2. **Reference Strategy Comparison**:
   - Run existing trend/cross-sectional strategy on:
     - Fixed symbol list (e.g., BTCUSDT, ETHUSDT, SOLUSDT)
     - Dynamic universe (filtered)
   - Compare metrics: Sharpe, return, max DD, trade count, win rate

3. **Parameter Sensitivity**:
   - Test different liquidity thresholds ($5M, $10M, $20M, $50M)
   - Test different warm-up periods (7, 14, 30 days)
   - Test different hysteresis ratios (70%, 80%, 90%)

### Metrics to Track

1. **PnL Metrics**:
   - Total return, Sharpe ratio, max drawdown, win rate, profit factor

2. **Universe Metrics**:
   - Universe size over time
   - Turnover (entries/exits per period)
   - Average 24h volume of universe
   - Number of new listings captured
   - Number of delistings avoided

3. **Operational Metrics**:
   - Average liquidity of traded symbols (actual 24h volume when traded)
   - Worst-case liquidity (minimum volume encountered)
   - Concentration (top 5 symbols by PnL contribution)

### Robustness Checks

- **Threshold Tightening**: Show performance vs. threshold level
- **Warm-Up Period Variation**: Show impact of different warm-up periods
- **Hysteresis Variation**: Show impact of hysteresis on turnover vs. performance

**Target Settings** (Good Compromise):
- Volume threshold: $10M entry, $7M exit (70% hysteresis)
- Warm-up: 14 days
- Min history: 30 days
- Expected universe size: 20-50 symbols (varies by market conditions)

## 5. Implementation Notes

### Data Dependencies
- **Exchange API**: Fetch instruments via Bybit v5 API (`/v5/market/instruments-info`)
- **OHLCV Cache**: Reuse existing `data/ohlcv_store.py` for historical data checks
- **24h Volume/Ticker Data**: Use CCXT `fetch_ticker()` or Bybit ticker endpoint (cached)

### Performance Optimizations
- **Cache Metadata**: Cache instrument metadata (refresh daily)
- **Cache 24h Stats**: Cache 24h volume/OI (refresh every 4-6 hours)
- **Batch Queries**: Fetch tickers for all symbols in one API call if possible
- **Lazy Evaluation**: Only compute volatility/filters for symbols that pass basic checks

### Error Handling
- **API Failures**: Fall back to last known universe if API fails
- **Data Gaps**: Warn but don't fail if some symbols missing data
- **Validation**: Validate all thresholds are positive, exit < entry, etc.

