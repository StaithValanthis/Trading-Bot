# Universe Parameter Optimization Examples

## Summary

This document provides examples and guidance for using the universe parameter optimization framework.

## Parameter Space

The optimizer tests combinations of the following parameters:

### Core Parameters

1. **Volume Entry Threshold** (`min_24h_volume_entry`)
   - Range: $5M - $200M USDT
   - Default: $10M USDT
   - Impact: Higher = fewer but more liquid symbols

2. **Volume Exit Threshold** (`min_24h_volume_exit`)
   - Range: 50% - 90% of entry threshold
   - Default: 70% ($7M for $10M entry)
   - Impact: Hysteresis prevents churn

3. **History Requirements** (`min_history_days`)
   - Range: 14 - 180 days
   - Default: 30 days
   - Impact: Filters out very new listings

4. **Warm-Up Period** (`warmup_days`)
   - Range: 0 - 45 days
   - Default: 14 days
   - Impact: Additional buffer before new listings become tradable

5. **Minimum Time in Universe** (`min_time_in_universe_days`)
   - Range: 3 - 21 days
   - Default: 7 days
   - Impact: Prevents rapid churn

6. **Maximum Turnover** (`max_turnover_per_rebalance_pct`)
   - Range: 10% - 50%
   - Default: 20%
   - Impact: Limits how many symbols can change per rebalance

7. **Volatility Cap** (`max_realized_vol_pct`)
   - Range: 100% - 500% annualized
   - Default: 200%
   - Impact: Filters out extremely volatile symbols

## Example Optimization Run

### Command

```bash
python -m src.main optimize-universe \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --n-combinations 200 \
  --method random \
  --output results/universe_optimization_2023.json
```

### Expected Output

```
================================================================================
UNIVERSE PARAMETER OPTIMIZATION RESULTS
================================================================================
Backtest Period: 2023-01-01 to 2024-01-01
Total Parameter Sets Tested: 87 passed constraints

Top 5 Recommended Configurations:
================================================================================

Primary Recommended:
--------------------------------------------------------------------------------
Composite Score: 0.7245

Strategy Performance:
  Annualized Return: +45.23%
  Sharpe Ratio: 1.85
  Max Drawdown: -12.34%
  Total Trades: 234
  Win Rate: 48.7%
  Profit Factor: 1.62

Universe Quality:
  Avg Universe Size: 28.5 symbols
  Avg 24h Volume: $45,230,000 USDT
  Universe Turnover Rate: 15.2%
  Avg Time in Universe: 45.3 days

Key Parameters:
  min_24h_volume_entry: $30,000,000
  min_24h_volume_exit: $21,000,000
  min_history_days: 45
  warmup_days: 14
  min_time_in_universe_days: 7

Config YAML (copy to config.yaml):
--------------------------------------------------------------------------------
universe:
  min_24h_volume_entry: 30000000
  min_24h_volume_exit: 21000000
  volume_check_days: 7
  min_open_interest: null
  max_spread_bps: null
  min_history_days: 45
  warmup_days: 14
  max_data_gap_pct: 5.0
  max_days_since_last_update: 7
  min_price_usdt: 0.01
  max_realized_vol_pct: 200.0
  limit_move_frequency_pct: 5.0
  min_time_in_universe_days: 7
  max_turnover_per_rebalance_pct: 20.0
  rebalance_frequency_hours: 24
  include_list: [BTCUSDT, ETHUSDT]
  exclude_list: []

Conservative Alternative:
--------------------------------------------------------------------------------
Composite Score: 0.6892

Strategy Performance:
  Annualized Return: +38.12%
  Sharpe Ratio: 1.72
  Max Drawdown: -9.87%
  Total Trades: 198
  Win Rate: 49.5%
  Profit Factor: 1.58

Universe Quality:
  Avg Universe Size: 22.3 symbols
  Avg 24h Volume: $52,100,000 USDT
  Universe Turnover Rate: 12.1%
  Avg Time in Universe: 52.1 days

Key Parameters:
  min_24h_volume_entry: $50,000,000
  min_24h_volume_exit: $35,000,000
  min_history_days: 60
  warmup_days: 21
  min_time_in_universe_days: 10

...
```

## Interpreting Results

### Composite Score

The composite score (0-1, higher is better) is calculated as:

```
composite = 0.4 * normalized_sharpe + 0.3 * normalized_return - 0.3 * normalized_drawdown
```

Where metrics are normalized to [0, 1] scale. This balances:
- **Sharpe Ratio** (40% weight): Risk-adjusted returns
- **Annualized Return** (30% weight): Absolute returns
- **Max Drawdown** (30% weight): Risk control (inverted, lower DD = higher score)

### Strategy Performance Metrics

- **Annualized Return**: Total return annualized over backtest period
- **Sharpe Ratio**: Risk-adjusted return metric (higher is better, >1.0 is good)
- **Max Drawdown**: Worst peak-to-trough decline (negative, closer to 0 is better)
- **Win Rate**: Percentage of profitable trades (typically 40-60% for trend strategies)
- **Profit Factor**: Gross profit / gross loss (>1.0 means profitable)

### Universe Quality Metrics

- **Avg Universe Size**: Average number of symbols in universe (aim for 15-50)
- **Avg 24h Volume**: Average liquidity of universe members (higher = more liquid)
- **Universe Turnover Rate**: How much the universe changes per rebalance (lower = more stable)
- **Avg Time in Universe**: How long symbols stay once added (higher = more stable)

### Key Parameter Insights

**Conservative Configs** typically have:
- Higher volume thresholds ($30M+ entry)
- Longer history requirements (45+ days)
- Longer warm-up periods (14-21 days)
- Lower turnover (<20%)

**Aggressive Configs** typically have:
- Lower volume thresholds ($10-20M entry)
- Shorter history requirements (30 days)
- Shorter warm-up periods (7-14 days)
- Higher turnover (20-30%)

## Recommendations

### For Retail Traders (Conservative)

Use parameters similar to:
- `min_24h_volume_entry: 30_000_000` ($30M)
- `min_24h_volume_exit: 21_000_000` ($21M, 70% hysteresis)
- `min_history_days: 45`
- `warmup_days: 14`
- `min_time_in_universe_days: 10`

**Rationale**: Focus on highly liquid, established symbols. Lower turnover reduces trading costs.

### For More Aggressive Strategies

Use parameters similar to:
- `min_24h_volume_entry: 10_000_000` ($10M)
- `min_24h_volume_exit: 7_000_000` ($7M, 70% hysteresis)
- `min_history_days: 30`
- `warmup_days: 7`
- `min_time_in_universe_days: 5`

**Rationale**: Capture more symbols and new listings earlier. Higher turnover but more opportunities.

### Robustness Checks

After running optimization, verify:

1. **Performance Across Subperiods**: Check if top configs perform well in both bull and bear markets
2. **Parameter Sensitivity**: Test ±20% changes to volume thresholds - does performance degrade gracefully?
3. **Sample Size**: Ensure enough trades (>50) for statistical significance
4. **Universe Stability**: Turnover rate should be <30% to avoid excessive churn

## Troubleshooting

### No Parameter Sets Pass Constraints

If optimizer returns no results:

1. **Relax constraints**: Lower `min_total_trades` or increase `max_drawdown_pct` in config
2. **Increase data**: Ensure you have sufficient historical data (6+ months)
3. **Check thresholds**: Volume thresholds may be too high for available symbols

### Universe Too Small/Large

If universe sizes are outside reasonable bounds:

1. **Adjust volume thresholds**: Lower entry threshold = larger universe
2. **Check data availability**: Ensure symbols exist in backtest period
3. **Review filters**: History/warm-up requirements may be too strict

### Optimization Takes Too Long

1. **Reduce combinations**: Use `--n-combinations 100` instead of 200
2. **Shorter period**: Reduce backtest window (e.g., 6 months instead of 12)
3. **Fewer symbols**: Test on subset of symbols first

## Best Practices

1. **Start with Defaults**: Test default config before optimizing
2. **Use Long Periods**: 12+ months of data for robust results
3. **Verify Out-of-Sample**: Test optimized params on data not used in optimization
4. **Monitor After Application**: Closely monitor performance when switching to optimized params
5. **Stay Conservative**: Prefer conservative configs unless you have strong evidence otherwise
6. **Regular Re-optimization**: Re-run optimization every 3-6 months as markets evolve

