"""Configuration management for the trading bot."""

import os
import yaml
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Module-level logger for debugging
_logger = logging.getLogger(__name__)

# Load .env file if it exists (look in current directory and parent directories)
# When running via systemd, EnvironmentFile directive should set env vars directly,
# but load_dotenv() is useful for manual runs.
# Note: This runs at import time, so it uses Python's current working directory.
# For systemd, EnvironmentFile should already set the vars, so this may be redundant.
env_result = load_dotenv(override=False)  # Don't override existing env vars
if env_result:
    _logger.debug("load_dotenv() found and loaded .env file at module import time")
else:
    _logger.debug("load_dotenv() did not find .env file at module import time (this is OK if using systemd EnvironmentFile)")


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str = "bybit"
    testnet: bool = True
    mode: str = "testnet"  # paper, testnet, or live
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframe: str = "4h"
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug: Check environment at __post_init__ time
        env_key_raw = os.getenv("BYBIT_API_KEY")
        env_secret_raw = os.getenv("BYBIT_API_SECRET")
        env_key_present = bool(env_key_raw)
        env_secret_present = bool(env_secret_raw)
        
        logger.debug(
            f"ExchangeConfig.__post_init__ - "
            f"Initial api_key: {bool(self.api_key)} ({len(self.api_key) if self.api_key else 0} chars), "
            f"Initial api_secret: {bool(self.api_secret)} ({len(self.api_secret) if self.api_secret else 0} chars), "
            f"ENV BYBIT_API_KEY present: {env_key_present}, "
            f"ENV BYBIT_API_SECRET present: {env_secret_present}"
        )
        
        # Try to reload .env if we're in a known working directory context
        # This is a fallback for cases where module-level load_dotenv() didn't work
        if not env_key_present or not env_secret_present:
            # Try loading from explicit paths
            cwd = os.getcwd()
            env_paths = [
                os.path.join(cwd, ".env"),
                os.path.join(os.path.dirname(cwd), ".env"),  # Parent directory
            ]
            for env_path in env_paths:
                if os.path.exists(env_path):
                    logger.debug(f"Attempting to reload .env from {env_path}")
                    result = load_dotenv(env_path, override=False)
                    if result:
                        logger.debug(f"Successfully loaded .env from {env_path}")
                        # Re-check environment after reload
                        env_key_raw = os.getenv("BYBIT_API_KEY")
                        env_secret_raw = os.getenv("BYBIT_API_SECRET")
                        env_key_present = bool(env_key_raw)
                        env_secret_present = bool(env_secret_raw)
                        break
        
        # Load from environment if None or empty string
        # Strip whitespace (common issue with .env files)
        if not self.api_key:
            if env_key_raw:
                self.api_key = env_key_raw.strip()
                logger.debug(
                    f"Loaded API key from environment ({len(self.api_key)} chars, "
                    f"first 3: {self.api_key[:3] if len(self.api_key) >= 3 else 'N/A'}, "
                    f"last 3: {self.api_key[-3:] if len(self.api_key) >= 3 else 'N/A'})"
                )
        if not self.api_secret:
            if env_secret_raw:
                self.api_secret = env_secret_raw.strip()
                logger.debug(
                    f"Loaded API secret from environment ({len(self.api_secret)} chars, "
                    f"first 3: {self.api_secret[:3] if len(self.api_secret) >= 3 else 'N/A'}, "
                    f"last 3: {self.api_secret[-3:] if len(self.api_secret) >= 3 else 'N/A'})"
                )
        
        # Final debug output
        logger.debug(
            f"ExchangeConfig.__post_init__ - Final api_key: {bool(self.api_key)} ({len(self.api_key) if self.api_key else 0} chars), "
            f"Final api_secret: {bool(self.api_secret)} ({len(self.api_secret) if self.api_secret else 0} chars)"
        )


@dataclass
class TrendStrategyConfig:
    """Time-series trend-following strategy parameters."""
    ma_short: int = 5  # Bars; interpretation depends on timeframe (5 bars @4h ≈ 20h)
    ma_long: int = 25  # Bars; 25 bars @4h ≈ 100h
    momentum_lookback: int = 6  # Bars; 6 bars @4h ≈ 24h
    atr_stop_multiplier: float = 2.5
    atr_period: int = 4  # Bars; 4 bars @4h ≈ 16h
    min_atr_threshold: float = 0.001
    # Position management
    max_holding_hours: Optional[int] = None  # Optional time-based exit (None = no limit)
    # Take-profit (optional, None = no TP, let winners run)
    take_profit_rr: Optional[float] = None  # Risk-reward multiple (e.g., 2.0 = TP at 2x SL distance)
    # Trailing stop (optional)
    use_trailing_stop: bool = False
    trailing_stop_atr_multiplier: float = 1.5  # Trailing stop distance (ATR multiplier)
    trailing_stop_activation_rr: float = 1.0  # Activate trailing after this RR profit (e.g., 1.0 = break-even)


@dataclass
class CrossSectionalStrategyConfig:
    """Cross-sectional momentum strategy parameters."""
    ranking_window: int = 18  # Bars; 18 bars @4h ≈ 3 days
    top_k: int = 3
    rebalance_frequency_hours: int = 8
    require_trend_alignment: bool = True


@dataclass
class FundingBiasConfig:
    """Funding-rate bias overlay parameters."""
    min_funding_threshold: float = 0.0005
    positive_funding_size_reduction: float = 0.5
    negative_funding_size_boost: float = 1.2
    max_boost_factor: float = 1.5


@dataclass
class StrategyConfig:
    """Overall strategy configuration."""
    trend: TrendStrategyConfig = field(default_factory=TrendStrategyConfig)
    cross_sectional: CrossSectionalStrategyConfig = field(default_factory=CrossSectionalStrategyConfig)
    funding_bias: FundingBiasConfig = field(default_factory=FundingBiasConfig)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    per_trade_risk_fraction: float = 0.005  # 0.5% per trade
    max_leverage: float = 3.0
    max_symbol_fraction: float = 0.30
    max_positions: int = 5
    daily_soft_loss_pct: float = -2.0
    daily_hard_loss_pct: float = -4.0
    kelly_fraction: float = 0.5
    # Advanced: approximate minimum liquidation distance as 1 / effective leverage.
    # Ensures effective leverage is capped such that 100 / leverage >= min_liquidation_distance_pct.
    min_liquidation_distance_pct: float = 5.0
    # Stop-loss and take-profit order management
    use_server_side_stops: bool = True  # Place real stop-loss orders on exchange (mandatory for production)
    stop_order_type: str = "stop_market"  # "stop_market" or "stop_limit"
    # Backtest slippage modeling
    stop_slippage_bps: float = 10.0  # Slippage in basis points when stop is hit (10 bps = 0.1%)
    tp_slippage_bps: float = 5.0  # Slippage for take-profit orders (usually tighter)


@dataclass
class DataConfig:
    """Data storage and retrieval configuration."""
    lookback_bars: int = 500
    db_path: str = "data/trading_bot.db"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lookback_months: int = 6
    walk_forward_window_days: int = 30
    min_trades: int = 20
    min_sharpe_ratio: float = 1.0
    max_drawdown_pct: float = -15.0
    search_method: str = "random"  # grid, random, bayesian
    n_trials: int = 50
    param_ranges: Dict[str, List[Any]] = field(default_factory=lambda: {
        "ma_short": [15, 25, 30],
        "ma_long": [80, 100, 120, 150],
        "momentum_lookback": [12, 24, 36, 48],
        "atr_stop_multiplier": [2.0, 2.5, 3.0],
        "top_k": [2, 3, 4, 5]
    })


@dataclass
class UniverseOptimizerConfig:
    """Universe parameter optimizer configuration."""
    n_combinations: int = 200
    search_method: str = "random"  # random or grid
    default_start_date: str = "2023-01-01"
    default_end_date: str = "2024-01-01"
    # Constraint thresholds
    min_avg_universe_size: int = 10
    max_avg_universe_size: int = 100
    max_universe_turnover_pct: float = 50.0
    max_drawdown_pct: float = -30.0
    min_total_trades: int = 50
    min_win_rate: float = 0.35


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    discord_webhook_url: Optional[str] = None
    report_time_utc: str = "09:00"
    include_strategy_breakdown: bool = True
    
    def __post_init__(self):
        """Load webhook URL from environment if not provided."""
        if self.discord_webhook_url is None:
            self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")


@dataclass
class UniverseConfig:
    """Universe selection configuration."""
    # Liquidity filters
    min_24h_volume_entry: float = 10_000_000.0  # $10M USDT
    min_24h_volume_exit: float = 7_000_000.0    # $7M USDT (hysteresis)
    volume_check_days: int = 7                   # Consecutive days for entry/exit
    min_open_interest: Optional[float] = None    # Optional, $5M USDT
    max_spread_bps: Optional[float] = None       # Optional, 20 basis points
    
    # Historical data
    min_history_days: int = 30
    warmup_days: int = 14                        # Warm-up period for new listings
    max_data_gap_pct: float = 5.0                # Max % missing candles
    max_days_since_last_update: int = 7          # Max days since last data update
    
    # Volatility & price
    min_price_usdt: float = 0.01                 # Minimum price
    max_realized_vol_pct: float = 200.0          # Annualized volatility %
    limit_move_frequency_pct: float = 5.0        # Max % of days with limit moves
    
    # Universe stability
    min_time_in_universe_days: int = 7           # Min days in universe before exit
    max_turnover_per_rebalance_pct: float = 20.0 # Max % of universe can change
    
    # Rebalancing
    rebalance_frequency_hours: int = 24          # Recompute universe daily
    
    # Overrides
    include_list: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    exclude_list: List[str] = field(default_factory=list)
    
    # Buckets (optional)
    max_symbols_per_bucket: Dict[str, int] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class BotConfig:
    """Main bot configuration."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    universe_optimizer: UniverseOptimizerConfig = field(default_factory=UniverseOptimizerConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Derived from YAML contents (short hash), for auditability.
    config_version: str = ""
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "BotConfig":
        """Load configuration from YAML file."""
        import logging
        logger = logging.getLogger(__name__)
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Debug: Log current working directory and environment state before loading
        import os
        cwd = os.getcwd()
        env_key_check = bool(os.getenv("BYBIT_API_KEY"))
        env_secret_check = bool(os.getenv("BYBIT_API_SECRET"))
        logger.debug(
            f"BotConfig.from_yaml - CWD: {cwd}, "
            f"Config path: {config_path}, "
            f"ENV BYBIT_API_KEY present: {env_key_check}, "
            f"ENV BYBIT_API_SECRET present: {env_secret_check}"
        )
        
        # Try to explicitly load .env from config file's directory or project root
        config_dir = config_path.parent.absolute()
        env_file_candidates = [
            config_dir / ".env",
            Path(cwd) / ".env",
            config_dir.parent / ".env",  # One level up from config
        ]
        for env_file in env_file_candidates:
            if env_file.exists():
                logger.debug(f"Found .env file at {env_file}, attempting to load...")
                result = load_dotenv(env_file, override=False)
                if result:
                    logger.debug(f"Successfully loaded .env from {env_file}")
                    # Re-check environment after reload
                    env_key_check = bool(os.getenv("BYBIT_API_KEY"))
                    env_secret_check = bool(os.getenv("BYBIT_API_SECRET"))
                    logger.debug(
                        f"After .env reload - ENV BYBIT_API_KEY present: {env_key_check}, "
                        f"ENV BYBIT_API_SECRET present: {env_secret_check}"
                    )
                    break
        
        with open(config_path, "r") as f:
            raw = f.read()
            config_dict = yaml.safe_load(raw)

        cfg = cls._from_dict(config_dict)

        # Compute stable, short config version hash for audit logs
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        cfg.config_version = sha[:10]

        return cfg
    
    @classmethod
    def _from_dict(cls, config_dict: dict) -> "BotConfig":
        """Create BotConfig from dictionary."""
        exchange_dict = config_dict.get("exchange", {})
        strategy_dict = config_dict.get("strategy", {})
        risk_dict = config_dict.get("risk", {})
        data_dict = config_dict.get("data", {})
        optimizer_dict = config_dict.get("optimizer", {})
        universe_dict = config_dict.get("universe", {})
        universe_optimizer_dict = config_dict.get("universe_optimizer", {})
        reporting_dict = config_dict.get("reporting", {})
        logging_dict = config_dict.get("logging", {})
        
        return cls(
            exchange=ExchangeConfig(**exchange_dict),
            strategy=StrategyConfig(
                trend=TrendStrategyConfig(**strategy_dict.get("trend", {})),
                cross_sectional=CrossSectionalStrategyConfig(**strategy_dict.get("cross_sectional", {})),
                funding_bias=FundingBiasConfig(**strategy_dict.get("funding_bias", {}))
            ),
            risk=RiskConfig(**risk_dict),
            data=DataConfig(**data_dict),
            optimizer=OptimizerConfig(**optimizer_dict),
            universe=UniverseConfig(**universe_dict),
            universe_optimizer=UniverseOptimizerConfig(**universe_optimizer_dict),
            reporting=ReportingConfig(**reporting_dict),
            logging=LoggingConfig(**logging_dict)
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Exchange validation
        if self.exchange.mode == "live" and not self.exchange.api_key:
            errors.append("API key required for live mode")
        if self.exchange.mode == "live" and not self.exchange.api_secret:
            errors.append("API secret required for live mode")
        if self.exchange.mode not in ["paper", "testnet", "live"]:
            errors.append(f"Invalid exchange mode: {self.exchange.mode}")
        
        # Risk validation
        if self.risk.per_trade_risk_fraction <= 0 or self.risk.per_trade_risk_fraction > 0.1:
            errors.append("per_trade_risk_fraction should be between 0 and 0.1 (0-10%)")
        if self.risk.max_leverage <= 0:
            errors.append("max_leverage must be positive")
        
        # Strategy validation
        if self.strategy.trend.ma_short >= self.strategy.trend.ma_long:
            errors.append("ma_short must be less than ma_long")
        
        # Universe validation
        if self.universe.min_24h_volume_exit >= self.universe.min_24h_volume_entry:
            errors.append("min_24h_volume_exit must be less than min_24h_volume_entry (hysteresis)")
        if self.universe.min_history_days <= 0:
            errors.append("min_history_days must be positive")
        
        return errors

