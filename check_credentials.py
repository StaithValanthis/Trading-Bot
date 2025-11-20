#!/usr/bin/env python3
"""
Diagnostic script to check API credential configuration.
This helps debug authentication issues with Bybit.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_credentials():
    """Check API credentials configuration."""
    
    print("=" * 60)
    print("Bybit API Credential Diagnostic Tool")
    print("=" * 60)
    print()
    
    # Find .env file
    env_paths = [
        Path(".env"),
        Path("config/.env"),
        Path("/home/ubuntu/Trading-Bot/.env"),
        Path("/home/ubuntu/Trading-Bot/config/.env"),
    ]
    
    env_file = None
    for path in env_paths:
        if path.exists():
            env_file = path
            print(f"✓ Found .env file: {path.absolute()}")
            break
    
    if not env_file:
        print("✗ ERROR: .env file not found in any of these locations:")
        for path in env_paths:
            print(f"    - {path.absolute()}")
        print("\nPlease create a .env file with your Bybit API credentials.")
        return False
    
    # Load .env file
    print(f"\nLoading .env file: {env_file.absolute()}")
    load_dotenv(env_file, override=False)
    
    # Check API key
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    
    print("\n" + "=" * 60)
    print("Credential Status:")
    print("=" * 60)
    
    # API Key
    if api_key:
        api_key_clean = api_key.strip()
        print(f"✓ BYBIT_API_KEY: Found")
        print(f"  Length: {len(api_key_clean)} characters")
        if len(api_key_clean) >= 6:
            print(f"  Preview: {api_key_clean[:3]}...{api_key_clean[-3:]}")
        else:
            print(f"  ⚠️  WARNING: API key seems too short (expected ~18 chars)")
        
        # Check for whitespace issues
        if api_key != api_key_clean:
            print(f"  ⚠️  WARNING: API key has leading/trailing whitespace (will be stripped)")
    else:
        print(f"✗ BYBIT_API_KEY: NOT FOUND")
        print(f"  Add this to your .env file:")
        print(f"  BYBIT_API_KEY=your_api_key_here")
        return False
    
    # API Secret
    if api_secret:
        api_secret_clean = api_secret.strip()
        print(f"✓ BYBIT_API_SECRET: Found")
        print(f"  Length: {len(api_secret_clean)} characters")
        if len(api_secret_clean) >= 6:
            print(f"  Preview: {api_secret_clean[:3]}...{api_secret_clean[-3:]}")
        else:
            print(f"  ⚠️  WARNING: API secret seems too short (expected ~36 chars)")
        
        # Check for whitespace issues
        if api_secret != api_secret_clean:
            print(f"  ⚠️  WARNING: API secret has leading/trailing whitespace (will be stripped)")
    else:
        print(f"✗ BYBIT_API_SECRET: NOT FOUND")
        print(f"  Add this to your .env file:")
        print(f"  BYBIT_API_SECRET=your_api_secret_here")
        return False
    
    # Check .env file format
    print("\n" + "=" * 60)
    print(".env File Format Check:")
    print("=" * 60)
    
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        has_key_line = False
        has_secret_line = False
        problematic_lines = []
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            if 'BYBIT_API_KEY' in line:
                has_key_line = True
                if '=' not in line:
                    problematic_lines.append(f"Line {i}: Missing '=' in BYBIT_API_KEY")
                elif line.count('=') > 1:
                    problematic_lines.append(f"Line {i}: Multiple '=' in BYBIT_API_KEY (use quotes if value contains '=')")
            
            if 'BYBIT_API_SECRET' in line:
                has_secret_line = True
                if '=' not in line:
                    problematic_lines.append(f"Line {i}: Missing '=' in BYBIT_API_SECRET")
                elif line.count('=') > 1:
                    problematic_lines.append(f"Line {i}: Multiple '=' in BYBIT_API_SECRET (use quotes if value contains '=')")
            
            # Check for quotes
            if 'BYBIT_API_KEY' in line or 'BYBIT_API_SECRET' in line:
                if line.strip().startswith('"') or line.strip().startswith("'"):
                    print(f"  ⚠️  Line {i}: Keys are quoted (quotes will be included in the value)")
                    print(f"      Remove quotes from the .env file")
        
        if problematic_lines:
            print("✗ Format issues found:")
            for issue in problematic_lines:
                print(f"  - {issue}")
        else:
            print("✓ .env file format looks correct")
            
    except Exception as e:
        print(f"⚠️  Could not read .env file: {e}")
    
    # Check systemd environment file
    print("\n" + "=" * 60)
    print("Systemd Configuration Check:")
    print("=" * 60)
    
    systemd_service_file = Path("/etc/systemd/system/bybit-bot.service")
    if systemd_service_file.exists():
        print(f"✓ Found systemd service file: {systemd_service_file}")
        
        try:
            with open(systemd_service_file, 'r') as f:
                content = f.read()
            
            if f"EnvironmentFile={env_file.absolute()}" in content or f"EnvironmentFile={env_file}" in content:
                print(f"✓ Service file references .env file correctly")
            else:
                print(f"⚠️  Service file might not load .env file correctly")
                print(f"   Check that ExecStart includes: EnvironmentFile={env_file.absolute()}")
        except Exception as e:
            print(f"⚠️  Could not read service file: {e}")
    else:
        print(f"⚠️  Systemd service file not found (normal if not installed via install.sh)")
    
    # Test credential loading via config
    print("\n" + "=" * 60)
    print("Testing Config Loading:")
    print("=" * 60)
    
    try:
        # Try to load config (without validating API keys)
        sys.path.insert(0, str(Path.cwd()))
        from src.config import ExchangeConfig
        
        # Create a minimal config to test credential loading
        test_config = ExchangeConfig(
            name="bybit",
            mode="live",  # Will be overridden by actual config
            testnet=False,  # Will be overridden by actual config
            api_key="",  # Should be loaded from env
            api_secret=""  # Should be loaded from env
        )
        
        if test_config.api_key and test_config.api_secret:
            print("✓ Config successfully loaded credentials from environment")
            print(f"  API key loaded: {len(test_config.api_key)} chars")
            print(f"  API secret loaded: {len(test_config.api_secret)} chars")
        else:
            print("✗ Config did not load credentials from environment")
            print("  This suggests the credential loading logic needs debugging")
            
    except Exception as e:
        print(f"⚠️  Could not test config loading: {e}")
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    print("\n1. Verify your .env file has this format (NO quotes, NO spaces around =):")
    print("   BYBIT_API_KEY=your_actual_key_here")
    print("   BYBIT_API_SECRET=your_actual_secret_here")
    
    print("\n2. Verify your API keys are valid:")
    print("   - Check Bybit account: https://www.bybit.com/app/user/api-management (live)")
    print("   - Or testnet: https://testnet.bybit.com/app/user/api-management")
    
    print("\n3. Verify API key permissions:")
    print("   - Must have 'Read' permission")
    print("   - Must have 'Trade' permission (if trading)")
    
    print("\n4. Check testnet vs live mismatch:")
    print("   - Live keys only work with testnet: false")
    print("   - Testnet keys only work with testnet: true")
    
    print("\n5. If using systemd, ensure .env file is readable:")
    print(f"   sudo chmod 600 {env_file.absolute()}")
    print(f"   sudo chown $USER:$USER {env_file.absolute()}")
    
    print("\n6. Reload systemd and restart service:")
    print("   sudo systemctl daemon-reload")
    print("   sudo systemctl restart bybit-bot.service")
    
    return True


if __name__ == "__main__":
    success = check_credentials()
    sys.exit(0 if success else 1)

