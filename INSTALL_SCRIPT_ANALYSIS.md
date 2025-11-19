# Install Script Analysis & Improvements

## Section A: Analysis of Current install.sh vs Checklist

### ✅ Correctly Implemented

1. **Pre-flight & Environment Checks**
   - ✅ Checks for Ubuntu/Debian using `lsb_release`
   - ✅ Verifies Python 3.10+ availability
   - ✅ Warns about running as root
   - ✅ Updates package list
   - ✅ Installs system dependencies (python3, python3-venv, python3-pip, git, curl, jq, sqlite3, systemd)

2. **Project Setup**
   - ✅ Assumes project is already cloned (operates in current directory)
   - ✅ Creates virtual environment in `venv/` directory
   - ✅ Installs requirements.txt
   - ✅ Creates necessary directories (data/, logs/)

3. **Config & Secrets**
   - ✅ Copies config.example.yaml to config.yaml if not exists
   - ✅ Creates .env file with proper permissions (600)
   - ✅ Prompts for Bybit API key, secret, testnet setting, Discord webhook
   - ✅ Saves secrets to .env file (not config.yaml)
   - ✅ Prevents overwriting existing configs

4. **Systemd Services & Timers**
   - ✅ Creates bybit-bot.service for live trading
   - ✅ Creates bybit-bot-optimizer.service + timer for daily optimization
   - ✅ Creates bybit-bot-report.service + timer for daily Discord report
   - ✅ Calls systemctl daemon-reload

### ⚠️ Partially Implemented / Issues

1. **Systemd Service Issues**
   - ⚠️ Services don't load .env file (Environment variables not loaded)
   - ⚠️ Services don't start automatically after installation
   - ⚠️ Service user assumption (`$USER`) might not work correctly for systemd
   - ⚠️ Missing EnvironmentFile directive to load .env

2. **Error Handling**
   - ⚠️ Uses `set -e` but doesn't handle specific failures gracefully
   - ⚠️ No check if requirements.txt exists before trying to install
   - ⚠️ No validation that entry point actually works after installation

3. **Idempotency**
   - ⚠️ Good: Checks if venv exists before creating
   - ⚠️ Good: Checks if config.yaml exists before copying
   - ⚠️ Good: Checks if .env entries exist before prompting
   - ⚠️ Issue: Doesn't check if systemd units already exist before creating (will overwrite)

4. **Final Verification**
   - ⚠️ Missing: No verification that CLI works (e.g., `python -m src.main --help`)
   - ⚠️ Missing: No check of systemd service status after creation
   - ⚠️ Missing: No test of entry point validity

5. **Paths & Entrypoints**
   - ⚠️ Uses `src.main live` which is correct
   - ⚠️ Missing check that src/main.py exists
   - ⚠️ Missing check that venv/bin/python exists after creation

6. **Permissions & Ownership**
   - ⚠️ Sets permissions on data/ and logs/ directories (good)
   - ⚠️ Doesn't ensure proper ownership if run as different user
   - ⚠️ Service user is set to `$USER` but might not exist when running via sudo

### ❌ Missing

1. **Environment Variable Loading in Systemd**
   - ❌ Systemd services don't load .env file
   - ❌ Need `EnvironmentFile=$BOT_DIR/.env` directive
   - ❌ Need to handle .env file not existing gracefully

2. **Service Auto-Start**
   - ❌ Doesn't enable/start services automatically (manual step required)
   - ❌ Should at least enable services (user can start later)
   - ❌ Should verify systemd units are valid

3. **Entry Point Verification**
   - ❌ No check that `python -m src.main --help` works
   - ❌ No validation of Python entry point
   - ❌ No dry-run test

4. **Better Error Messages**
   - ❌ Generic error messages if something fails
   - ❌ No clear indication of which step failed

5. **Missing Checks**
   - ❌ Doesn't verify requirements.txt exists before installing
   - ❌ Doesn't verify config.example.yaml exists
   - ❌ Doesn't check if systemd is actually available

6. **Post-Install Verification**
   - ❌ No final sanity check
   - ❌ No verification that all services are created correctly
   - ❌ No check that timers are scheduled correctly

## Section B: Missing / Broken Parts and Recommended Fixes

### Critical Fixes Needed

1. **Load .env in Systemd Units**
   - Add `EnvironmentFile=$BOT_DIR/.env` to all service units
   - Handle case where .env doesn't exist (create empty one)
   - Ensure .env file has correct permissions

2. **Enable Services Automatically**
   - Call `systemctl enable` for main service and timers
   - Don't auto-start live bot (user should start manually)
   - Auto-start timers (they're one-shot and scheduled)

3. **Better Service User Handling**
   - Use `id -un` to get current user
   - If running as root, use a dedicated user (e.g., `bybit-bot`)
   - Create user if it doesn't exist (or document requirement)

4. **Entry Point Verification**
   - Test that `python -m src.main --help` works after installation
   - Verify all entry points are accessible

5. **Error Handling Improvements**
   - Check if requirements.txt exists before installing
   - Check if config.example.yaml exists
   - Add better error messages with step numbers
   - Use `set -euo pipefail` instead of just `set -e`

6. **Idempotency for Systemd Units**
   - Check if unit files exist before creating
   - Optionally back up existing units before overwriting

7. **Final Verification**
   - Run `python -m src.main --help` to verify CLI works
   - Check systemd unit syntax with `systemctl cat`
   - Verify timers are scheduled correctly

### Design Decisions

1. **Environment Variables**: Keep in .env file, load via EnvironmentFile in systemd
2. **Service User**: Use current user if not root, otherwise require dedicated user
3. **Auto-Start**: Enable services but don't auto-start live bot (let user decide)
4. **Config Location**: Use absolute paths in systemd units for reliability
5. **Idempotency**: Check for existence before creating (don't overwrite existing configs)

