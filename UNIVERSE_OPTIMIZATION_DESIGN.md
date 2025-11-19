# Universe Selection Optimization Design Document

## 1. Parameter Space Definition

### Universe Selection Parameters for Optimization

#### A. Liquidity Thresholds

1. **Volume Entry Threshold** (`min_24h_volume_entry`)
   - **Range**: $5M - $200M USDT
   - **Recommended Grid**: [5M, 10M, 20M, 30M, 50M, 75M, 100M, 150M, 200M]
   - **Rationale**: Lower bounds avoid illiquid coins, upper bounds ensure enough symbols in universe
   - **Default**: $10M (conservative retail)

2. **Volume Exit Threshold** (`min_24h_volume_exit`)
   - **Range**: 50% - 90% of entry threshold (hysteresis ratio)
   - **Recommended Ratios**: [0.5, 0.6, 0.7, 0.8, 0.9]
   - **Rationale**: Prevents churn; symbols must fall significantly before removal
   - **Default**: 70% ($7M for $10M entry)

3. **Volume Check Days** (`volume_check_days`)
   - **Range**: 3 - 14 days
   - **Recommended Grid**: [3, 5, 7, 10, 14]
   - **Rationale**: Require sustained liquidity before entry/exit
   - **Default**: 7 days

4. **Open Interest Filter** (optional)
   - **Range**: $0 (disabled) - $20M USDT
   - **Recommended**: [null, 5M, 10M, 20M]
   - **Note**: Only if OI data available

#### B. Historical Data Requirements

1. **Minimum History Days** (`min_history_days`)
   - **Range**: 14 - 180 days
   - **Recommended Grid**: [14, 21, 30, 45, 60, 90, 120, 180]
   - **Rationale**: Need sufficient data for indicators, but not too restrictive
   - **Default**: 30 days

2. **Warm-Up Period** (`warmup_days`)
   - **Range**: 0 - 45 days
   - **Recommended Grid**: [0, 7, 14, 21, 30, 45]
   - **Rationale**: Allow new listings to stabilize before trading
   - **Default**: 14 days

3. **Maximum Data Gap** (`max_data_gap_pct`)
   - **Range**: 1% - 10%
   - **Recommended**: [1.0, 2.5, 5.0, 7.5, 10.0]
   - **Rationale**: Allow minor gaps but reject broken data
   - **Default**: 5.0%

#### C. Hysteresis & Stability Parameters

1. **Minimum Time in Universe** (`min_time_in_universe_days`)
   - **Range**: 3 - 21 days
   - **Recommended**: [3, 5, 7, 10, 14, 21]
   - **Rationale**: Once added, stay for minimum period (unless delisted)
   - **Default**: 7 days

2. **Maximum Turnover Per Rebalance** (`max_turnover_per_rebalance_pct`)
   - **Range**: 10% - 50%
   - **Recommended**: [10, 15, 20, 25, 30, 40, 50]
   - **Rationale**: Limit churn per rebalance cycle
   - **Default**: 20%

3. **Rebalance Frequency** (`rebalance_frequency_hours`)
   - **Range**: 6 - 168 hours (1 week)
   - **Recommended**: [6, 12, 24, 48, 72, 168]
   - **Rationale**: Balance responsiveness vs stability
   - **Default**: 24 hours (daily)

#### D. Universe Size & Composition

1. **Maximum Universe Size** (optional)
   - **Range**: 10 - 100 symbols
   - **Recommended**: [10, 20, 30, 50, 75, 100, null (unlimited)]
   - **Rationale**: Limit to most liquid/quality symbols
   - **Default**: null (unlimited)

2. **Bucket Caps** (optional, advanced)
   - Per-bucket limits (majors, L1s, meme, DeFi, etc.)
   - **Note**: Not included in initial optimization due to complexity

#### E. Volatility & Price Filters

1. **Maximum Realized Volatility** (`max_realized_vol_pct`)
   - **Range**: 100% - 500% annualized
   - **Recommended**: [100, 150, 200, 250, 300, 400, 500]
   - **Rationale**: Avoid pathological volatility
   - **Default**: 200%

2. **Minimum Price** (`min_price_usdt`)
   - **Range**: $0.001 - $0.10
   - **Recommended**: [0.001, 0.01, 0.05, 0.10]
   - **Rationale**: Avoid dust coins with rounding issues
   - **Default**: $0.01

### Parameter Space Summary

**Core Parameters to Optimize** (priority order):
1. `min_24h_volume_entry` (5M-200M)
2. `min_24h_volume_exit` ratio (0.5-0.9 of entry)
3. `min_history_days` (14-180)
4. `warmup_days` (0-45)
5. `min_time_in_universe_days` (3-21)
6. `max_turnover_per_rebalance_pct` (10-50)
7. `max_realized_vol_pct` (100-500)

**Fixed Parameters** (not optimized):
- `include_list`, `exclude_list` (manual overrides)
- `max_data_gap_pct` (data quality threshold)
- `min_price_usdt` (dust filter)

**Total Parameter Combinations**: ~50,000-500,000+ (need sampling)

**Recommended Initial Sweep**: Random sample of 200-500 combinations from above ranges

## 2. Survivorship-Bias-Safe Backtesting Methodology

### A. Data Requirements

1. **Historical OHLCV Data**
   - For all Bybit USDT perpetuals with any historical data
   - Include delisted symbols (don't exclude future knowledge)
   - Minimum: Last 12-24 months of 1h bars

2. **24h Volume Time Series**
   - Daily snapshots of 24h volume per symbol
   - Approximate from OHLCV if needed: rolling 24h sum of volume * close price

3. **Listing Dates**
   - Use first timestamp in OHLCV data as proxy for listing date
   - Note: This is approximate but acceptable for backtesting

4. **Delisting Dates**
   - Use last timestamp in OHLCV data before gap > 30 days
   - Or explicit delisting flag if available

### B. Time-Respecting Universe Construction

For each test date `t` in backtest period:

1. **Available Symbols** (as of date `t`):
   - Only symbols with data up to date `t`
   - Symbols that haven't been delisted yet (last data >= `t`)

2. **Volume Calculation** (as of date `t`):
   - Use 24h volume ending at date `t`
   - Or rolling average over `volume_check_days` ending at `t`

3. **History Check** (as of date `t`):
   - Require `min_history_days` of data before date `t`
   - Check for gaps in data up to date `t`

4. **Warm-Up Check** (as of date `t`):
   - If symbol listed after `t - warmup_days - min_history_days`, exclude
   - Track when symbol first passed filters, only add after warmup period

5. **Hysteresis** (as of date `t`):
   - For entry: Check if volume >= entry threshold for `volume_check_days` consecutive days ending at `t`
   - For exit: Check if volume < exit threshold for `volume_check_days` consecutive days ending at `t`
   - Track current membership to apply entry vs exit logic

6. **Delisting Check**:
   - If symbol's last data < `t`, immediately remove (hard delisting)

### C. Integration with Strategy Backtest

1. **Universe-Aware Backtest Flow**:
   ```
   For each date t in backtest period:
     a. Build universe at date t using time-respecting filters
     b. Record universe membership
     c. For each symbol in universe:
        - Generate trading signals (trend, cross-sectional, etc.)
        - Apply strategy logic
        - Execute trades only if symbol in universe
     d. Track positions, PnL, etc.
   ```

2. **Position Handling**:
   - If symbol removed from universe, close position at removal date
   - Don't enter new positions in symbols outside universe
   - Respect universe membership at every rebalance

3. **Delisting Handling**:
   - When symbol delisted (no data), close position immediately
   - Mark reason as "delisted" in trade log

### D. Survivorship-Bias Prevention

1. **Include All Symbols with Data**:
   - Don't filter out symbols that later died
   - Don't use future information about delistings

2. **Warm-Up Periods**:
   - New listings must exist + pass filters for `warmup_days` before tradable
   - This avoids survivorship bias (only including coins that survived)

3. **Historical Universe Snapshots**:
   - Store universe membership at each date
   - Use historical snapshots when backtesting (not current universe)

## 3. Metrics for Universe-Quality Evaluation

### A. Strategy Performance Metrics

For each universe parameter set, calculate:

1. **Return Metrics**:
   - Total return (%)
   - Annualized return (%)
   - Cumulative return curve

2. **Risk-Adjusted Metrics**:
   - Sharpe ratio (annualized)
   - Sortino ratio (downside deviation only)
   - Calmar ratio (return / max drawdown)

3. **Drawdown Metrics**:
   - Maximum drawdown (%)
   - Average drawdown duration (days)
   - Recovery time after max DD (days)

4. **Trade Statistics**:
   - Total number of trades
   - Win rate (%)
   - Average win / average loss
   - Profit factor (gross profit / gross loss)
   - Expected value per trade

5. **Turnover & Costs**:
   - Estimated total fees (assuming 0.055% taker fee)
   - Trades per day/week
   - Position turnover rate

### B. Universe Quality Metrics

1. **Size Characteristics**:
   - Average universe size (symbols)
   - Median universe size
   - Min/max universe size
   - Standard deviation of size (stability)

2. **Liquidity Characteristics**:
   - Average 24h volume of universe members (USDT)
   - Median 24h volume
   - 25th/75th percentile volumes
   - Minimum volume encountered while trading

3. **Turnover Metrics**:
   - Average symbols added per rebalance
   - Average symbols removed per rebalance
   - Total symbol additions over period
   - Total symbol removals over period
   - Universe turnover rate (symbols changed / universe size)

4. **Longevity Metrics**:
   - Average time symbol stays in universe (days)
   - Median time in universe
   - Symbols that stayed entire period (%)
   - Symbols that entered/left multiple times

5. **Composition Metrics**:
   - % of PnL from top 5 symbols
   - % of PnL from top 10 symbols
   - Herfindahl concentration index
   - Number of unique symbols traded

### C. Stability & Robustness Metrics

1. **Regime Performance**:
   - Performance in bull periods (positive trend)
   - Performance in bear periods (negative trend)
   - Performance in choppy/sideways periods
   - Consistency across subperiods (coefficient of variation)

2. **Parameter Sensitivity**:
   - Performance change for ±20% volume threshold
   - Performance change for ±5 days history requirement
   - Stability rank (low sensitivity = more robust)

3. **Data Quality Indicators**:
   - Average data gaps per symbol
   - Symbols with data quality issues
   - Symbols excluded due to gaps

### D. Results Schema

```python
@dataclass
class UniverseOptimizationResult:
    """Results for a single universe parameter set."""
    # Parameters tested
    params: Dict[str, Any]
    
    # Strategy performance
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    estimated_fees: float
    
    # Universe quality
    avg_universe_size: float
    median_universe_size: float
    min_universe_size: int
    max_universe_size: int
    universe_size_std: float
    
    avg_volume_24h: float
    median_volume_24h: float
    min_volume_24h: float
    p25_volume_24h: float
    p75_volume_24h: float
    
    avg_additions_per_rebalance: float
    avg_removals_per_rebalance: float
    total_additions: int
    total_removals: int
    universe_turnover_rate: float
    
    avg_time_in_universe_days: float
    pct_symbols_stayed_entire_period: float
    
    pct_pnl_from_top5: float
    pct_pnl_from_top10: float
    unique_symbols_traded: int
    
    # Robustness
    performance_by_regime: Dict[str, float]  # bull, bear, chop
    regime_consistency_score: float  # lower is better (more consistent)
    sensitivity_score: float  # lower is better (more robust)
    
    # Metadata
    backtest_start_date: str
    backtest_end_date: str
    timestamp: str
```

## 4. Optimization Approach

### A. Search Strategy

**Phase 1: Coarse Random Search** (200-500 combinations)
- Randomly sample from parameter ranges
- Goal: Identify promising regions of parameter space
- Fast, broad coverage

**Phase 2: Local Refinement** (optional, 50-100 combinations)
- Around top 10 performers from Phase 1
- Fine-tune parameters (±10-20% adjustments)
- Goal: Find local optima near best coarse results

**Phase 3: Robustness Check** (top 20-30 combinations)
- Test across multiple subperiods
- Test parameter sensitivity
- Select final recommendations

### B. Objective Function

**Primary Ranking**:
1. **Composite Score** = `0.4 * normalized_sharpe + 0.3 * normalized_return - 0.3 * normalized_drawdown`
   - All metrics normalized to [0, 1] scale
   - Penalize drawdown, reward Sharpe and return

2. **Constraints** (must pass):
   - `avg_universe_size >= 10` (need enough symbols)
   - `avg_universe_size <= 100` (avoid too many)
   - `universe_turnover_rate <= 0.5` (avoid excessive churn)
   - `max_drawdown_pct >= -30%` (avoid catastrophic losses)
   - `total_trades >= 50` (need sufficient sample)
   - `win_rate >= 0.35` (basic profitability filter)

3. **Secondary Ranking** (for ties):
   - Higher Sharpe ratio
   - Lower max drawdown
   - Lower turnover rate

### C. Robustness Filters

**Must Pass All**:
1. Minimum trades: `total_trades >= 50`
2. Universe size bounds: `10 <= avg_universe_size <= 100`
3. Drawdown cap: `max_drawdown_pct >= -30%`
4. Regime consistency: Performance positive in at least 2 of 3 regimes (bull/bear/chop)
5. Parameter sensitivity: `sensitivity_score <= threshold` (low sensitivity to small changes)

### D. Recommended Parameter Sets

From optimization results, select:

1. **Primary Recommended**:
   - Best composite score passing all constraints
   - Strong robustness metrics
   - Reasonable universe size and turnover

2. **Conservative Alternative**:
   - Higher volume thresholds
   - Longer warm-up periods
   - Lower turnover
   - Lower drawdown risk

3. **Aggressive Alternative**:
   - Lower volume thresholds
   - Shorter warm-up periods
   - Larger universe
   - Higher expected returns (with higher risk)

**Output Format**: Ready-to-paste `config.yaml` blocks for each recommendation

## 5. Implementation Notes

### Key Challenges

1. **Data Availability**:
   - May not have exact listing dates → use first data timestamp
   - May not have daily volume → approximate from OHLCV
   - Missing data gaps → interpolate or flag

2. **Computational Cost**:
   - 200-500 backtests × full historical period can be slow
   - Optimize by caching universe construction
   - Parallelize where possible

3. **Overfitting Risk**:
   - Don't optimize on same data you'll trade
   - Use walk-forward or out-of-sample validation
   - Prefer robust, stable parameters over peak performers

### Performance Optimization

1. **Caching**:
   - Cache 24h volume calculations per symbol per date
   - Cache universe membership snapshots
   - Reuse strategy backtest results for same universe

2. **Vectorization**:
   - Vectorize volume calculations where possible
   - Batch universe construction checks

3. **Early Termination**:
   - Stop backtest early if universe becomes empty
   - Stop if drawdown exceeds threshold

