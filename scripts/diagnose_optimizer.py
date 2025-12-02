#!/usr/bin/env python3
"""Diagnostic script to identify optimizer startup issues."""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        print("  ✓ Importing config...")
        from src.config import BotConfig
        print("  ✓ Importing optimizer...")
        from src.optimizer.optimizer import Optimizer
        print("  ✓ Importing backtester...")
        from src.backtest.backtester import Backtester
        print("  ✓ Importing data store...")
        from src.data.ohlcv_store import OHLCVStore
        print("  ✓ Importing funding opportunity...")
        from src.signals.funding_opportunity import FundingOpportunityGenerator
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_load(config_path: str):
    """Test loading config file."""
    print("\n" + "=" * 60)
    print("Testing config load...")
    print("=" * 60)
    
    try:
        from src.config import BotConfig
        print(f"  Loading config from: {config_path}")
        config = BotConfig.from_yaml(config_path)
        print("  ✓ Config loaded successfully")
        
        # Check critical config sections
        print(f"  ✓ Exchange timeframe: {config.exchange.timeframe}")
        print(f"  ✓ Strategy funding_opportunity.enabled: {config.strategy.funding_opportunity.enabled}")
        print(f"  ✓ Optimizer lookback_months: {config.optimizer.lookback_months}")
        print(f"  ✓ Data DB path: {config.data.db_path}")
        
        return config
    except Exception as e:
        print(f"\n❌ Config load failed: {e}")
        traceback.print_exc()
        return None

def test_backtester_init(config):
    """Test backtester initialization."""
    print("\n" + "=" * 60)
    print("Testing Backtester initialization...")
    print("=" * 60)
    
    try:
        from src.backtest.backtester import Backtester
        print("  Creating Backtester instance...")
        backtester = Backtester(config)
        print("  ✓ Backtester initialized successfully")
        print(f"  ✓ Funding opportunity gen: {backtester.funding_opportunity_gen is not None}")
        return True
    except Exception as e:
        print(f"\n❌ Backtester init failed: {e}")
        traceback.print_exc()
        return False

def test_optimizer_init(config):
    """Test optimizer initialization."""
    print("\n" + "=" * 60)
    print("Testing Optimizer initialization...")
    print("=" * 60)
    
    try:
        from src.optimizer.optimizer import Optimizer
        from src.data.ohlcv_store import OHLCVStore
        print("  Creating OHLCVStore...")
        store = OHLCVStore(config.data.db_path)
        print("  Creating Optimizer instance...")
        optimizer = Optimizer(config, store)
        print("  ✓ Optimizer initialized successfully")
        return True
    except Exception as e:
        print(f"\n❌ Optimizer init failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostics."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.example.yaml"
    
    print("OPTIMIZER DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print()
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ CRITICAL: Import test failed. Check dependencies.")
        sys.exit(1)
    
    # Test 2: Config load
    config = test_config_load(config_path)
    if not config:
        print("\n❌ CRITICAL: Config load failed. Check config file.")
        sys.exit(1)
    
    # Test 3: Backtester init
    if not test_backtester_init(config):
        print("\n❌ CRITICAL: Backtester init failed.")
        sys.exit(1)
    
    # Test 4: Optimizer init
    if not test_optimizer_init(config):
        print("\n❌ CRITICAL: Optimizer init failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL DIAGNOSTICS PASSED")
    print("=" * 60)
    print("\nThe optimizer should be able to start. If it still fails,")
    print("check the systemd logs with:")
    print("  sudo journalctl -u bybit-bot-optimizer.service -n 100")

if __name__ == "__main__":
    main()

