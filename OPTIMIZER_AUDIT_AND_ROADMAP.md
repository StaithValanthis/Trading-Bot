# Optimizer Architecture Audit & Improvement Roadmap

## Section A – Current Optimizer Architecture (as implemented here)

### Overview
The optimizer (`src/optimizer/optimizer.py`) implements a walk-forward parameter optimization system for a cross-sectional momentum trading bot. It tests discrete parameter combinations from `config.optimizer.param_ranges` using historical backtests.

### Parameter Search Methods

**Supported Methods:**
- **Random search** (default): Uniform or Latin Hypercube sampling from discrete parameter ranges
- **Grid search**: Exhaustive enumeration of all combinations (rarely used due to combinatorial explosion)

**Implementation Details:**
- Parameter sets generated via `_generate_random_params()` or `_generate_grid_params()`
- Coverage warning when `n_trials / total_combinations < coverage_warning_threshold` (default 0.0001)
- Always includes current config as baseline (test #0)
- Hall-of-fame seeding: Top 5 historical performers from `OptimizerStore.get_top_historical_parameters()` added as seeds
- Reproducibility: Optional `random_seed` for deterministic runs

**Current Limitations:**
- No Bayesian optimization despite `search_method: "bayesian"` being mentioned in config (not implemented)
- Latin Hypercube sampling exists but is basic (just shuffles values, not true stratified sampling)
- No adaptive search (doesn't learn from previous trials to focus on promising regions)

### Walk-Forward & OOS Design

**Current Implementation:**
- **Window creation**: Non-overlapping windows of `walk_forward_window_days` (default 60 days)
- **IS/OOS split**: Fixed 70/30 split of windows (first 70% = IS, last 30% = OOS)
- **Embargo**: Minimal – skips first OOS window (`oos_eval = oos_results[1:]`) to avoid look-ahead, but no proper purge period
- **Multiple folds**: `walk_forward_folds` config exists (default 1) but **not implemented** – only single pass through windows

**Window Processing:**
```python
# Lines 770-780: Creates non-overlapping windows
while current_date < end_date:
    window_end = current_date + timedelta(days=window_days)
    windows.append((current_date, window_end))
    current_date = window_end  # No overlap, no gap
```

**IS/OOS Aggregation:**
- Metrics computed per window, then averaged across IS and OOS separately
- No rolling retraining – each window is independent backtest
- No consideration of parameter stability across windows

**Gaps:**
1. **No purged cross-validation**: Overlapping lookback periods can leak information (e.g., MA(100) on window N uses data that overlaps with window N+1's training)
2. **No embargo period**: Should skip a gap between IS and OOS to prevent data leakage from rebalancing logic
3. **Single fold only**: `walk_forward_folds` config is ignored
4. **Fixed split**: 70/30 split doesn't adapt to data availability

### Metrics & Objective Function

**Computed Metrics:**
- **Per window**: `total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `total_trades`
- **Aggregated IS**: `avg_sharpe_is`, `avg_dd_is`, `min_trades_is`
- **Aggregated OOS**: `avg_sharpe_oos`, `avg_dd_oos`, `min_trades_oos`
- **Robustness**: `positive_window_ratio`, `oos_sharpe_std`, `worst_oos_dd` (computed but not always used in ranking)

**Acceptance Criteria:**
```python
# Lines 397-407: Three-tier criteria
passes_is = (
    min_trades_is >= min_trades_threshold
    and avg_sharpe_is >= min_sharpe_threshold
    and avg_dd_is >= max_dd_threshold
)
passes_oos = (
    min_trades_oos >= min_trades_threshold
    and avg_sharpe_oos >= min_sharpe_threshold * 0.7  # 30% relaxation
    and avg_dd_oos >= max_dd_threshold * 1.2  # 20% more lenient
)
```

**Ranking Logic:**
```python
# Lines 550-559: Multi-criteria sorting
results.sort(
    key=lambda x: (
        0 if x.get('passes_all', False) else 1,  # Passes all first
        x.get('avg_sharpe_oos', -999),          # Then OOS Sharpe
        x.get('avg_dd_oos', -999),              # Then OOS DD
    ),
    reverse=True,
)
```

**Robustness Filters (Partially Implemented):**
- `min_positive_windows`: Computed (`positive_window_ratio`) but **not enforced** in acceptance criteria
- `max_oos_sharpe_std`: Computed but **not enforced** (only logged if exceeds threshold)
- `max_worst_oos_dd`: Computed but **not enforced** (only logged if exceeds threshold)

**Objective Function:**
- **Effective objective**: Maximize OOS Sharpe, subject to hard constraints (trades, Sharpe, DD thresholds)
- **Issue**: Robustness metrics are computed but don't affect ranking unless they cause `passes_oos = False` (which they currently don't)

### Overfitting Controls

**Existing Controls:**
1. **IS/OOS split**: Prevents direct overfitting to full dataset
2. **Multiple windows**: Tests across different time periods
3. **Relaxed OOS criteria**: 30% lower Sharpe threshold, 20% more lenient DD for OOS
4. **Minimum trades**: Ensures statistical significance
5. **Hall-of-fame continuity**: Seeds from historical runs (prevents complete parameter drift)

**Missing Controls:**
1. **No purged cross-validation**: Overlapping lookbacks leak information
2. **No embargo period**: Rebalancing logic can see future data
3. **Robustness metrics not enforced**: `min_positive_windows`, `max_oos_sharpe_std`, `max_worst_oos_dd` are computed but ignored
4. **No multiple testing correction**: Testing 100+ parameter sets increases false discovery rate
5. **No stability checks**: Doesn't verify parameters work across multiple regimes

### Universe, Funding, and Execution Realism

**Universe History:**
- **Supported**: `universe_history: Optional[Dict[date, Set[str]]]` parameter exists
- **Usage**: Passed to `Backtester.backtest()` which filters symbols by date
- **Config**: `use_universe_history: bool` flag exists but must be set manually in CLI
- **Gap**: Not automatically loaded from `UniverseStore` – requires manual construction

**Funding Modeling:**
- **Constant rate**: `funding_rate_per_8h` (default 0.0) applied per bar
- **Per-symbol**: `symbol_funding_rates` dict supported in config but **not used in optimizer**
- **Realism**: Uses constant approximation, not historical funding rates

**Execution Costs:**
- **Fees**: `taker_fee` (default 0.00055) applied to all trades
- **Slippage**: `stop_slippage_bps` (10 bps) and `tp_slippage_bps` (5 bps) modeled
- **Realism**: Reasonable for crypto futures, but no spread modeling or market impact

**Risk Constraints:**
- **Parity**: Backtester uses same `PositionSizer` and `PortfolioLimits` as live trading
- **Enforcement**: Leverage, concentration, max positions all enforced identically
- **Gap**: No modeling of exchange-specific constraints (min order size, precision)

### Result Persistence & Usability

**SQLite Schema (`OptimizerStore`):**
- `optimizer_runs`: Run metadata (timeframe, symbols, dates, status)
- `optimizer_param_results`: Per-parameter-set metrics (IS/OOS Sharpe, DD, trades, accepted flag)
- `optimizer_best_parameters`: Best-known params per timeframe/universe (updated on each run)

**Hall-of-Fame Logic:**
- `get_top_historical_parameters()` aggregates across runs, ranks by composite score
- Composite score: `2.0 * sharpe_oos - 0.1 * |dd_oos| + consistency_bonus + robustness_bonus`
- Seeds top 5 performers into new runs (prevents parameter drift)

**Reproducibility:**
- `random_seed` supported for deterministic sampling
- Run IDs (UUID) track each optimization
- Config version stored for tracking changes

**Usability Issues:**
- No CLI command to query historical results
- No visualization of parameter space exploration
- No automatic parameter update (must manually copy best params to config)
- Limited logging of why parameters were rejected

### Technical Debt & Complexity

**Code Smells:**
1. **Dead code**: Lines 633-652 have unreachable `return` statements after error handling
2. **Incomplete feature**: `walk_forward_folds` config exists but unused
3. **Inconsistent enforcement**: Robustness metrics computed but not applied to ranking
4. **Manual universe history**: Must construct `Dict[date, Set[str]]` manually instead of auto-loading
5. **Mixed concerns**: Optimizer handles both search logic and walk-forward orchestration

**Complexity:**
- Single 986-line file with multiple responsibilities
- Walk-forward logic embedded in optimizer (not reusable)
- Hard to test individual components in isolation

---

## Section B – Best-Practice Optimizer Design for a Bot Like This

### 1. Search Methods

**Random Search:**
- **When appropriate**: Large parameter spaces (>10k combinations), limited compute budget (<200 trials)
- **Trade-offs**: Simple, parallelizable, but inefficient (no learning from previous trials)
- **Best practice**: Use Latin Hypercube Sampling (LHS) for better coverage than uniform random

**Grid Search:**
- **When appropriate**: Small parameter spaces (<1k combinations), need exhaustive coverage
- **Trade-offs**: Guarantees finding global optimum in discrete space, but exponential growth
- **Best practice**: Only for 2-3 parameters with <10 values each

**Bayesian Optimization (SMBO):**
- **When appropriate**: Expensive evaluations (slow backtests), limited trials (<100), continuous or large discrete spaces
- **Trade-offs**: Learns from previous trials to focus on promising regions, but requires surrogate model (GP, RF)
- **Best practice**: Use libraries like `scikit-optimize` or `optuna` for production systems
- **For crypto futures**: Highly recommended due to slow backtests and need for efficient exploration

**Evolutionary/Genetic Algorithms:**
- **When appropriate**: Very large spaces, multi-objective optimization, need to escape local optima
- **Trade-offs**: Can find diverse solutions, but slower convergence and harder to tune
- **Best practice**: Use for final refinement after initial Bayesian/random search

**Recommendation for this bot**: Start with improved random/LHS, then add Bayesian optimization as Phase 2 upgrade.

### 2. Overfitting Control

**Walk-Forward Optimization (WFO):**
- **Rolling windows**: Train on window N, test on window N+1, then retrain on N+1, test on N+2, etc.
- **Fixed origin**: Train on first K windows, test on remaining windows (current approach)
- **Best practice**: Use rolling windows for more robust evaluation, but requires more compute

**Purged Cross-Validation:**
- **Problem**: Overlapping lookback periods leak information (e.g., MA(100) on training window uses data that overlaps with test window)
- **Solution**: "Purge" training data that would be visible in test window's lookback period
- **Implementation**: If test window starts at T and strategy uses 100-bar lookback, purge training data from [T-100, T+embargo]
- **Best practice**: Essential for strategies with long lookbacks (MA, momentum windows)

**Embargo Period:**
- **Problem**: Rebalancing logic may use data from end of training period that's too close to test period
- **Solution**: Add gap between training and test (e.g., skip 1-2 rebalance periods)
- **Best practice**: Embargo = max(rebalance_frequency, signal_lookback) to prevent leakage

**Multiple OOS Regimes:**
- **Problem**: Single OOS period may be favorable/unfavorable regime
- **Solution**: Test across multiple time periods (bull, bear, sideways markets)
- **Best practice**: Require positive performance in at least 50-60% of OOS windows (`min_positive_windows`)

**Robustness Metrics:**
- **OOS Sharpe dispersion**: Low std dev indicates stability across regimes
- **Worst-case OOS drawdown**: Ensures strategy doesn't fail catastrophically in any regime
- **Best practice**: Reject parameter sets with `oos_sharpe_std > 1.0` or `worst_oos_dd < -30%`

**Multiple Testing Correction:**
- **Problem**: Testing 100+ parameter sets increases false discovery (lucky parameter set)
- **Solution**: Bonferroni correction, or require higher Sharpe threshold when testing many sets
- **Best practice**: Adjust `min_sharpe_ratio` upward based on `n_trials` (e.g., `min_sharpe = 1.0 + 0.1 * log10(n_trials/10)`)

### 3. Realism and Parity

**Universe Membership:**
- **Problem**: Using current universe in historical backtests creates look-ahead bias
- **Solution**: Use historical universe snapshots (what was actually tradable at each date)
- **Best practice**: Load from `UniverseStore` automatically, filter symbols by date in backtester

**Funding Rates:**
- **Problem**: Constant funding rate doesn't reflect reality (varies by symbol and time)
- **Solution**: Use historical funding rates per symbol (if available) or realistic estimates
- **Best practice**: Per-symbol funding from exchange API or historical database

**Execution Costs:**
- **Fees**: Model taker/maker fees accurately (current: 0.055% is reasonable for Bybit)
- **Slippage**: Different for market vs limit orders (current: 10 bps stop, 5 bps TP is reasonable)
- **Spread**: Model bid-ask spread (currently missing – add 2-5 bps for market orders)
- **Market impact**: For large orders, model price impact (currently missing)

**Risk Constraints:**
- **Parity**: Backtester must enforce same limits as live (leverage, concentration, max positions)
- **Current status**: ✅ Good – uses same `PositionSizer` and `PortfolioLimits`
- **Gap**: Exchange-specific constraints (min order size, precision) not modeled

### 4. Practical Constraints

**Limited Compute:**
- **Problem**: Can only run 50-200 backtests due to time constraints
- **Solutions**:
  1. Narrow parameter ranges (reduce combinations to <10k)
  2. Use Bayesian optimization (more efficient than random)
  3. Parallelize backtests (current: sequential)
  4. Early stopping (reject obviously bad parameters after 1-2 windows)

**Parameter Dimensionality:**
- **Problem**: Too many parameters → overfitting, too few → miss opportunities
- **Best practice**: Start with 5-8 core parameters, expand only if necessary
- **Current**: 9 parameters (ma_short, ma_long, momentum_lookback, atr_stop_multiplier, atr_period, top_k, ranking_window, exit_band, rebalance_frequency_hours) – reasonable

**Reproducibility:**
- **Seeds**: Use fixed seeds for random number generation
- **Logging**: Log all parameter sets tested, metrics, and decisions
- **Versioning**: Track config versions and data versions
- **Current status**: ✅ Good – `random_seed` and run tracking exist

---

## Section C – Gap Analysis (Current vs Best Practice)

| Topic | Current Implementation | Best Practice | Gap Severity | Consequences |
|-------|----------------------|---------------|--------------|--------------|
| **Search Strategy** | Random/Grid only | Bayesian optimization for efficiency | **Medium** | Wasted compute, may miss optimal regions |
| **Parameter Ranges** | Manual config, no validation | Auto-validate coverage vs trials | **Low** | User must manually balance ranges/trials |
| **Hall-of-Fame** | ✅ Implemented (top 5 seeds) | ✅ Good – prevents parameter drift | **None** | Working as intended |
| **Walk-Forward** | Fixed 70/30 split, single fold | Rolling windows, multiple folds | **High** | Less robust evaluation, may overfit to single split |
| **Purged CV** | ❌ Not implemented | Purge training data overlapping test lookback | **High** | Data leakage from overlapping lookbacks (MA, momentum) |
| **Embargo Period** | Minimal (skips first OOS window) | Proper gap = max(rebalance, lookback) | **Medium** | Rebalancing logic may see future data |
| **Robustness Metrics** | Computed but not enforced | Enforce min_positive_windows, max_oos_sharpe_std, max_worst_oos_dd | **High** | Unstable parameters may pass (high Sharpe in one window, fail in others) |
| **Multiple Testing** | ❌ No correction | Adjust Sharpe threshold by log(n_trials) | **Medium** | False discovery rate increases with more trials |
| **Universe History** | Manual construction required | Auto-load from UniverseStore | **Medium** | Easy to forget, creates look-ahead bias if omitted |
| **Funding Rates** | Constant approximation | Per-symbol historical funding | **Low** | Minor impact for most strategies, but not realistic |
| **Execution Costs** | Fees + slippage modeled | Missing spread modeling | **Low** | Minor impact (2-5 bps), but adds up over many trades |
| **Objective Function** | OOS Sharpe (with constraints) | Multi-objective or composite score | **Low** | Current approach is reasonable, but could be more nuanced |
| **Result Persistence** | ✅ SQLite with run tracking | ✅ Good | **None** | Working well |
| **Reproducibility** | ✅ Seeds and run IDs | ✅ Good | **None** | Working well |
| **Usability** | Manual parameter update | Auto-suggest or CLI to apply best params | **Low** | Minor inconvenience |

### Summary by Severity

**High Severity (Must Fix):**
1. **Purged cross-validation**: Data leakage from overlapping lookbacks
2. **Robustness metrics not enforced**: Unstable parameters can pass
3. **Single fold walk-forward**: Less robust than multiple folds

**Medium Severity (Should Fix):**
4. **No embargo period**: Rebalancing logic may leak information
5. **No multiple testing correction**: False discovery rate increases
6. **Universe history manual**: Easy to forget, causes look-ahead bias
7. **No Bayesian optimization**: Inefficient search for limited compute

**Low Severity (Nice to Have):**
8. **No spread modeling**: Minor cost underestimation
9. **Constant funding rates**: Not realistic but minor impact
10. **Manual parameter update**: Usability issue

---

## Section D – Phased Improvement Roadmap

### Phase 1 – Low-Risk Incremental Improvements

**Goals:**
- Fix critical robustness issues with minimal code changes
- Enforce already-computed metrics
- Improve usability and logging

**Key Changes:**

1. **Enforce Robustness Filters** (`src/optimizer/optimizer.py`)
   - Apply `min_positive_windows` to `passes_oos` (lines 403-407)
   - Apply `max_oos_sharpe_std` and `max_worst_oos_dd` as hard filters
   - Update ranking to penalize high dispersion
   ```python
   # After line 407, add:
   if passes_oos and positive_ratio < min_positive_windows:
       passes_oos = False
   if passes_oos and max_oos_sharpe_std and oos_sharpe_std > max_oos_sharpe_std:
       passes_oos = False
   if passes_oos and max_worst_oos_dd and worst_oos_dd < max_worst_oos_dd:
       passes_oos = False
   ```

2. **Auto-Load Universe History** (`src/optimizer/optimizer.py`, `src/cli/main.py`)
   - Add `_load_universe_history()` helper that queries `UniverseStore`
   - Auto-load when `use_universe_history=True` (no manual construction)
   - Files: `optimizer.py` (add method), `main.py` (call before optimize)

3. **Multiple Testing Correction** (`src/optimizer/optimizer.py`)
   - Adjust `min_sharpe_threshold` based on `n_trials`
   - Formula: `adjusted_sharpe = base_sharpe + 0.1 * log10(max(1, n_trials / 10))`
   - Apply to both IS and OOS thresholds

4. **Better Logging & Diagnostics** (`src/optimizer/optimizer.py`)
   - Log why parameters were rejected (which filter failed)
   - Add summary table of top 10 parameter sets with all metrics
   - Log parameter space coverage statistics

5. **Remove Dead Code** (`src/optimizer/optimizer.py`)
   - Delete unreachable code after line 631 (lines 633-652)

**Files to Modify:**
- `src/optimizer/optimizer.py` (~50 lines changed)
- `src/cli/main.py` (~10 lines for universe history loading)
- `src/config.py` (no changes, use existing config)

**Risks:**
- Low – mostly enforcing existing logic, no architectural changes
- May reject more parameter sets (good for robustness, but may find fewer "good" sets)

**Estimated Effort:** 4-6 hours

---

### Phase 2 – Structural Optimizer Upgrades

**Goals:**
- Implement proper purged cross-validation
- Add embargo periods
- Implement multiple walk-forward folds
- Improve search efficiency

**Key Changes:**

1. **Purged Cross-Validation** (`src/optimizer/optimizer.py`)
   - Add `_calculate_purge_period()` based on max lookback (MA, momentum, ranking_window)
   - Purge training data that overlaps with test window's lookback
   - Modify `_walk_forward_backtest()` to filter training data by purge period
   ```python
   def _calculate_purge_period(self, params: Dict, timeframe: str) -> int:
       """Calculate purge period in bars based on max lookback."""
       hours_per_bar = parse_timeframe_to_hours(timeframe)
       max_lookback_bars = max(
           params.get('ma_long', 100),
           params.get('momentum_lookback', 24),
           params.get('ranking_window', 18),
       )
       purge_bars = max_lookback_bars + 10  # Safety margin
       return purge_bars
   ```

2. **Embargo Period** (`src/optimizer/optimizer.py`)
   - Add `embargo_days` config (default: max(rebalance_frequency_hours/24, 1))
   - Skip embargo period between training and test windows
   - Modify window creation to include embargo gap

3. **Multiple Walk-Forward Folds** (`src/optimizer/optimizer.py`)
   - Implement `walk_forward_folds` (currently ignored)
   - Create multiple IS/OOS splits (e.g., fold 1: windows 1-7 train, 8-10 test; fold 2: 2-8 train, 9-10 test)
   - Average metrics across folds for more robust evaluation

4. **Improved Latin Hypercube Sampling** (`src/optimizer/optimizer.py`)
   - Replace basic shuffling with proper LHS (use `scipy.stats.qmc.LatinHypercube`)
   - Ensures better coverage of parameter space

5. **Early Stopping** (`src/optimizer/optimizer.py`)
   - Reject parameter sets after 1-2 windows if they fail hard (e.g., Sharpe < -1.0, DD < -50%)
   - Saves compute for obviously bad parameters

**Files to Modify:**
- `src/optimizer/optimizer.py` (~200 lines changed, new methods)
- `src/config.py` (add `embargo_days` to `OptimizerConfig`)
- `config.example.yaml` (add `embargo_days` example)

**Risks:**
- Medium – more complex logic, need to test purge/embargo correctness
- May significantly reduce number of valid windows (especially with long lookbacks)
- Need to handle edge cases (insufficient data after purging)

**Estimated Effort:** 16-24 hours

---

### Phase 3 – Advanced / Nice-to-Have

**Goals:**
- Add Bayesian optimization
- Multi-objective optimization
- Monte Carlo robustness checks
- Better visualization and tooling

**Key Changes:**

1. **Bayesian Optimization** (`src/optimizer/optimizer.py`, new `src/optimizer/bayesian_optimizer.py`)
   - Integrate `scikit-optimize` or `optuna` for SMBO
   - Use Gaussian Process or Random Forest surrogate model
   - Acquisition function: Expected Improvement (EI) or Upper Confidence Bound (UCB)
   - Convert discrete parameter ranges to continuous for GP, then round to nearest valid value
   ```python
   # Pseudo-code structure
   from skopt import gp_minimize
   from skopt.space import Integer, Real
   
   def objective(params):
       # Convert continuous to discrete
       discrete_params = _round_to_valid_values(params)
       # Run walk-forward backtest
       results = _walk_forward_backtest(...)
       # Return negative OOS Sharpe (minimize = maximize Sharpe)
       return -results['avg_sharpe_oos']
   
   space = [
       Integer(5, 20, name='ma_short'),
       Integer(50, 150, name='ma_long'),
       # ... etc
   ]
   result = gp_minimize(objective, space, n_calls=100)
   ```

2. **Multi-Objective Optimization** (`src/optimizer/optimizer.py`)
   - Use NSGA-II or similar for Pareto-optimal solutions
   - Objectives: Maximize OOS Sharpe, minimize OOS drawdown, minimize turnover
   - Return set of non-dominated solutions, let user choose based on risk preference

3. **Monte Carlo Robustness** (`src/optimizer/optimizer.py`)
   - Bootstrap sample returns from backtest to generate confidence intervals
   - Test parameter stability under random shocks (e.g., ±10% return perturbation)
   - Reject parameters that fail robustness tests

4. **Visualization & Tooling** (`src/cli/main.py`, new `src/optimizer/viz.py`)
   - CLI command: `python -m src.main optimizer-results --timeframe 1h`
   - Plot parameter space exploration (2D projections)
   - Plot equity curves for top parameter sets
   - Plot robustness metrics (Sharpe vs DD scatter, window performance heatmap)

5. **Auto-Parameter Update** (`src/cli/main.py`)
   - CLI command: `python -m src.main optimizer-apply --run-id <id>`
   - Automatically update `config.yaml` with best parameters (with confirmation)
   - Create backup of old config before updating

**Files to Modify:**
- `src/optimizer/optimizer.py` (add Bayesian wrapper)
- New: `src/optimizer/bayesian_optimizer.py`
- New: `src/optimizer/viz.py`
- `src/cli/main.py` (add CLI commands)
- `requirements.txt` (add `scikit-optimize` or `optuna`, `matplotlib`)

**Risks:**
- High – complex implementations, new dependencies
- Bayesian optimization requires tuning hyperparameters (GP kernel, acquisition function)
- May be overkill for current use case (100-200 trials is manageable with random search)

**Estimated Effort:** 40-60 hours

---

## Summary & Recommendations

### Immediate Actions (Phase 1)
1. **Enforce robustness filters** – Critical for preventing unstable parameters
2. **Auto-load universe history** – Prevents look-ahead bias
3. **Multiple testing correction** – Reduces false discovery

### Short-Term (Phase 2)
4. **Purged cross-validation** – Fixes data leakage
5. **Embargo periods** – Prevents rebalancing leakage
6. **Multiple folds** – More robust evaluation

### Long-Term (Phase 3)
7. **Bayesian optimization** – If compute becomes bottleneck
8. **Visualization** – Better understanding of parameter space

### Priority Order
1. **Phase 1** (enforce robustness) – Do immediately
2. **Phase 2** (purged CV, embargo) – Do within 1-2 months
3. **Phase 3** (Bayesian, viz) – Consider if/when needed

The current optimizer is **functional but has critical gaps** in overfitting control. Phase 1 fixes are low-risk and high-impact. Phase 2 requires more testing but significantly improves robustness. Phase 3 is optional enhancement for power users.

