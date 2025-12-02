#!/usr/bin/env bash
# Installation script for Bybit Trading Bot
# This script sets up the bot on Ubuntu with all dependencies and services
# Idempotent: safe to run multiple times

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BOT_DIR/venv"
PYTHON_VERSION="3.10"
SERVICE_NAME="bybit-bot"
CONFIG_FILE="$BOT_DIR/config.yaml"
ENV_FILE="$BOT_DIR/.env"

# Non-interactive mode flag (set NO_PROMPT=1 or pass --non-interactive)
NON_INTERACTIVE="${NO_PROMPT:-false}"
for arg in "$@"; do
    if [ "$arg" = "--non-interactive" ]; then
        NON_INTERACTIVE="true"
    fi
done

# Determine service user
if [ "$EUID" -eq 0 ]; then
    # Running as root - try to find a non-root user or create one
    if [ -n "${SUDO_USER:-}" ]; then
        SERVICE_USER="$SUDO_USER"
    else
        # Check if bybit-bot user exists
        if id "bybit-bot" &>/dev/null; then
            SERVICE_USER="bybit-bot"
        else
            SERVICE_USER="root"  # Fallback, but not ideal
            echo -e "${YELLOW}Warning: Running as root without SUDO_USER. Service will run as root (not recommended).${NC}"
        fi
    fi
else
    SERVICE_USER="$USER"
fi

# Helper functions
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

info() {
    echo -e "${GREEN}$1${NC}"
}

step() {
    echo -e "${BLUE}[$1] $2${NC}"
}

# Main installation
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Bybit Trading Bot Installation${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # ============================================================
    # Step 1: Pre-flight checks
    # ============================================================
    step "1/11" "Pre-flight checks..."
    
    # Check OS
    if ! command -v lsb_release &> /dev/null; then
        error_exit "lsb_release not found. This script is designed for Ubuntu/Debian."
    fi
    
    OS_NAME=$(lsb_release -si)
    OS_VERSION=$(lsb_release -rs)
    
    if [[ ! "$OS_NAME" =~ ^(Ubuntu|Debian)$ ]]; then
        warn "OS is $OS_NAME, not Ubuntu/Debian. Continuing anyway..."
    else
        info "Detected OS: $OS_NAME $OS_VERSION"
    fi
    
    # Check if running as root and warn
    if [ "$EUID" -eq 0 ] && [ -z "${SUDO_USER:-}" ]; then
        warn "Running as root without SUDO_USER. Some steps may not work correctly."
        if [ "$NON_INTERACTIVE" != "true" ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check systemd is available
    if ! command -v systemctl &> /dev/null; then
        error_exit "systemctl not found. Systemd is required."
    fi
    
    # Check if systemd is running
    if ! systemctl is-system-running --quiet 2>/dev/null; then
        warn "Systemd may not be running. Continuing anyway..."
    fi
    
    info "Service will run as user: $SERVICE_USER"
    echo ""
    
    # ============================================================
    # Step 2: Install system dependencies
    # ============================================================
    step "2/11" "Installing system dependencies..."
    
    sudo apt-get update -qq || error_exit "Failed to update package list"
    
    # Install required packages
    sudo apt-get install -y -qq \
        python3 \
        python3-venv \
        python3-pip \
        git \
        curl \
        jq \
        sqlite3 \
        systemd \
        || error_exit "Failed to install system dependencies"
    
    info "System dependencies installed"
    
    # Verify Python version
    PYTHON3_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    info "Python version: $PYTHON3_VERSION"
    
    # Check if Python 3.10+ is available
    if [ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON3_VERSION" | sort -V | head -n1)" != "$PYTHON_VERSION" ]; then
        warn "Python $PYTHON3_VERSION detected. Python 3.10+ recommended."
        if [ "$NON_INTERACTIVE" != "true" ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    echo ""
    
    # ============================================================
    # Step 3: Verify project files
    # ============================================================
    step "3/11" "Verifying project files..."
    
    # Check if we're in the project root
    if [ ! -f "$BOT_DIR/src/main.py" ]; then
        error_exit "src/main.py not found. Are you in the project root?"
    fi
    
    # Check if src/cli/main.py exists (actual CLI entrypoint)
    if [ ! -f "$BOT_DIR/src/cli/main.py" ]; then
        error_exit "src/cli/main.py not found. Project structure may be incomplete."
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "$BOT_DIR/requirements.txt" ]; then
        error_exit "requirements.txt not found in $BOT_DIR"
    fi
    
    # Check if config.example.yaml exists
    if [ ! -f "$BOT_DIR/config.example.yaml" ]; then
        warn "config.example.yaml not found. You'll need to create config.yaml manually."
    fi
    
    # Verify critical modules exist (including funding strategy and optimizer integration)
    CRITICAL_MODULES=(
        "src/config.py"
        "src/exchange/bybit_client.py"
        "src/data/ohlcv_store.py"
        "src/data/downloader.py"
        "src/execution/executor.py"
        "src/state/portfolio.py"
        "src/backtest/backtester.py"
        "src/optimizer/optimizer.py"
        "src/optimizer/timeframe_analyzer.py"
        "src/signals/trend.py"
        "src/signals/cross_sectional.py"
        "src/signals/funding_opportunity.py"
        "src/signals/funding_carry.py"
        "src/risk/position_sizing.py"
        "src/risk/portfolio_limits.py"
        "src/universe/selector.py"
        "src/universe/store.py"
        "src/reporting/discord_reporter.py"
        "src/cli/main.py"
    )
    
    MISSING_MODULES=()
    for module in "${CRITICAL_MODULES[@]}"; do
        if [ ! -f "$BOT_DIR/$module" ]; then
            MISSING_MODULES+=("$module")
        fi
    done
    
    if [ ${#MISSING_MODULES[@]} -gt 0 ]; then
        warn "Some critical modules are missing:"
        for module in "${MISSING_MODULES[@]}"; do
            warn "  - $module"
        done
        if [ "$NON_INTERACTIVE" != "true" ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        info "All critical modules found"
    fi
    
    # Verify funding strategy modules (funding opportunity + optimizer integration)
    FUNDING_MODULES=(
        "src/signals/funding_opportunity.py"
        "src/reporting/discord_reporter.py"  # Includes funding metrics in reports
    )
    
    FUNDING_MISSING=()
    for module in "${FUNDING_MODULES[@]}"; do
        if [ ! -f "$BOT_DIR/$module" ]; then
            FUNDING_MISSING+=("$module")
        fi
    done
    
    if [ ${#FUNDING_MISSING[@]} -eq 0 ]; then
        info "✓ All funding strategy modules found"
    else
        warn "Some funding strategy modules are missing:"
        for module in "${FUNDING_MISSING[@]}"; do
            warn "  - $module"
        done
    fi
    
    # Verify optimizer includes funding optimization
    if grep -q "optimize_funding_strategy" "$BOT_DIR/src/optimizer/optimizer.py" 2>/dev/null; then
        info "✓ Funding optimizer integration found in optimizer.py"
    else
        warn "Funding optimizer integration not found in optimizer.py (funding optimization may not be available)"
    fi
    
    # Verify scripts directory exists (optional, scripts are standalone)
    if [ ! -d "$BOT_DIR/scripts" ]; then
        warn "scripts/ directory not found (optional, for standalone analysis scripts)"
    else
        info "scripts/ directory found"
        # Check for important scripts
        if [ -f "$BOT_DIR/scripts/optimize_and_compare_timeframes.py" ]; then
            info "  ✓ optimize_and_compare_timeframes.py found"
        fi
        if [ -f "$BOT_DIR/scripts/download_historical_data.py" ]; then
            info "  ✓ download_historical_data.py found"
        fi
    fi
    
    info "Project files verified"
    echo ""
    
    # ============================================================
    # Step 4: Create virtual environment
    # ============================================================
    step "4/11" "Setting up Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        info "Virtual environment already exists at $VENV_DIR, skipping creation..."
    else
        python3 -m venv "$VENV_DIR" || error_exit "Failed to create virtual environment"
        info "Virtual environment created at $VENV_DIR"
    fi
    
    # Verify venv was created
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        error_exit "Virtual environment Python executable not found at $VENV_DIR/bin/python"
    fi
    
    echo ""
    
    # ============================================================
    # Step 5: Install Python dependencies
    # ============================================================
    step "5/11" "Installing Python dependencies..."
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate" || error_exit "Failed to activate virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip -q || error_exit "Failed to upgrade pip"
    
    # Install requirements
    pip install -r "$BOT_DIR/requirements.txt" -q || error_exit "Failed to install Python dependencies"
    
    info "Python dependencies installed"
    echo ""
    
    # ============================================================
    # Step 6: Create necessary directories
    # ============================================================
    step "6/11" "Creating directories..."
    
    # Create core directories
    mkdir -p "$BOT_DIR/data" || error_exit "Failed to create data directory"
    mkdir -p "$BOT_DIR/logs" || error_exit "Failed to create logs directory"
    
    # Create results directory for optimizer/analysis outputs
    mkdir -p "$BOT_DIR/results" || warn "Failed to create results directory (non-critical)"
    
    # Create optimizer_results directory for funding optimizer results (JSON files)
    mkdir -p "$BOT_DIR/optimizer_results" || warn "Failed to create optimizer_results directory (non-critical)"
    
    # Set permissions
    chmod 755 "$BOT_DIR/data" "$BOT_DIR/logs" "$BOT_DIR/results" "$BOT_DIR/optimizer_results" 2>/dev/null || warn "Failed to set directory permissions"
    
    # Set ownership if running as root
    if [ "$EUID" -eq 0 ] && [ "$SERVICE_USER" != "root" ]; then
        chown -R "$SERVICE_USER:$SERVICE_USER" "$BOT_DIR/data" "$BOT_DIR/logs" "$BOT_DIR/results" "$BOT_DIR/optimizer_results" 2>/dev/null || warn "Failed to set directory ownership"
    fi
    
    info "Directories created: data/, logs/, results/, optimizer_results/"
    echo ""
    
    # ============================================================
    # Step 7: Configure environment file
    # ============================================================
    step "7/11" "Configuring environment file..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        touch "$ENV_FILE" || error_exit "Failed to create .env file"
        chmod 600 "$ENV_FILE" || warn "Failed to set .env file permissions"
        info "Created .env file"
    else
        info ".env file already exists"
    fi
    
    # Set ownership if running as root
    if [ "$EUID" -eq 0 ] && [ "$SERVICE_USER" != "root" ]; then
        chown "$SERVICE_USER:$SERVICE_USER" "$ENV_FILE" || warn "Failed to set .env file ownership"
    fi
    
    echo ""
    
    # ============================================================
    # Step 8: Copy config file
    # ============================================================
    step "8/11" "Setting up configuration file..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "$BOT_DIR/config.example.yaml" ]; then
            cp "$BOT_DIR/config.example.yaml" "$CONFIG_FILE" || error_exit "Failed to copy config.example.yaml"
            info "Created config.yaml from example"
            
            # Verify funding strategy sections exist in the new config
            if grep -q "funding_opportunity:" "$CONFIG_FILE" 2>/dev/null; then
                info "✓ Funding opportunity strategy section found in config"
            else
                warn "Funding opportunity strategy section not found in config.example.yaml"
            fi
            
            if grep -q "funding:" "$CONFIG_FILE" 2>/dev/null && grep -A 1 "optimizer:" "$CONFIG_FILE" | grep -q "funding:" 2>/dev/null; then
                info "✓ Funding optimizer section found in config"
            else
                warn "Funding optimizer section not found in config.example.yaml"
            fi
        else
            warn "config.example.yaml not found. You'll need to create config.yaml manually."
        fi
    else
        info "config.yaml already exists, skipping copy (preserving existing configuration)"
        
        # Check if existing config has funding sections (informational only)
        if grep -q "funding_opportunity:" "$CONFIG_FILE" 2>/dev/null; then
            info "✓ Existing config.yaml contains funding_opportunity section"
        else
            warn "Existing config.yaml does not contain funding_opportunity section (funding strategy may not be enabled)"
        fi
        
        if grep -q "funding:" "$CONFIG_FILE" 2>/dev/null && grep -A 1 "optimizer:" "$CONFIG_FILE" | grep -q "funding:" 2>/dev/null; then
            info "✓ Existing config.yaml contains funding optimizer section"
        else
            warn "Existing config.yaml does not contain funding optimizer section (funding optimization may not be enabled)"
        fi
    fi
    
    # Set ownership if running as root
    if [ "$EUID" -eq 0 ] && [ "$SERVICE_USER" != "root" ]; then
        chown "$SERVICE_USER:$SERVICE_USER" "$CONFIG_FILE" || warn "Failed to set config file ownership"
    fi
    
    echo ""
    
    # ============================================================
    # Step 9: Interactive configuration
    # ============================================================
    step "9/11" "Interactive configuration..."
    echo ""
    
    # Read API keys
    if ! grep -q "^BYBIT_API_KEY=" "$ENV_FILE" 2>/dev/null; then
        if [ "$NON_INTERACTIVE" != "true" ]; then
            echo -e "${YELLOW}Enter your Bybit API Key:${NC}"
            read -r BYBIT_API_KEY
            if [ -n "$BYBIT_API_KEY" ]; then
                echo "BYBIT_API_KEY=$BYBIT_API_KEY" >> "$ENV_FILE"
                info "Bybit API Key saved"
            fi
        else
            warn "BYBIT_API_KEY not set in .env and NON_INTERACTIVE=true; skipping prompt."
        fi
    else
        info "Bybit API Key already configured"
    fi
    
    if ! grep -q "^BYBIT_API_SECRET=" "$ENV_FILE" 2>/dev/null; then
        if [ "$NON_INTERACTIVE" != "true" ]; then
            echo -e "${YELLOW}Enter your Bybit API Secret:${NC}"
            read -rs BYBIT_API_SECRET
            echo ""
            if [ -n "$BYBIT_API_SECRET" ]; then
                echo "BYBIT_API_SECRET=$BYBIT_API_SECRET" >> "$ENV_FILE"
                info "Bybit API Secret saved"
            fi
        else
            warn "BYBIT_API_SECRET not set in .env and NON_INTERACTIVE=true; skipping prompt."
        fi
    else
        info "Bybit API Secret already configured"
    fi
    
    # Account type / testnet
    USE_TESTNET_VALUE=""
    if ! grep -q "^BYBIT_TESTNET=" "$ENV_FILE" 2>/dev/null; then
        if [ "$NON_INTERACTIVE" != "true" ]; then
            echo -e "${YELLOW}Use Bybit testnet? (Y/n)${NC}"
            read -r USE_TESTNET
            if [[ "$USE_TESTNET" =~ ^[Nn]$ ]]; then
                echo "BYBIT_TESTNET=false" >> "$ENV_FILE"
                USE_TESTNET_VALUE="false"
            else
                echo "BYBIT_TESTNET=true" >> "$ENV_FILE"
                USE_TESTNET_VALUE="true"
            fi
            info "Bybit testnet setting saved"
        else
            # Default to testnet in non-interactive mode
            echo "BYBIT_TESTNET=true" >> "$ENV_FILE"
            USE_TESTNET_VALUE="true"
            info "BYBIT_TESTNET set to true (non-interactive default)"
        fi
    else
        # Read existing value
        USE_TESTNET_VALUE=$(grep "^BYBIT_TESTNET=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' | tr -d "'" | tr -d ' ')
        info "Bybit testnet setting already configured: $USE_TESTNET_VALUE"
    fi
    
    # Update config.yaml with testnet and mode settings
    if [ -f "$CONFIG_FILE" ] && [ -n "$USE_TESTNET_VALUE" ]; then
        info "Updating config.yaml with testnet setting: $USE_TESTNET_VALUE"
        
        # Determine testnet boolean and mode based on user's choice
        if [ "$USE_TESTNET_VALUE" = "true" ] || [ "$USE_TESTNET_VALUE" = "True" ] || [ "$USE_TESTNET_VALUE" = "TRUE" ] || [ "$USE_TESTNET_VALUE" = "1" ]; then
            TESTNET_BOOL="true"
            EXCHANGE_MODE="testnet"
        else
            TESTNET_BOOL="false"
            EXCHANGE_MODE="live"
        fi
        
        # Use Python to safely update YAML file (more reliable than sed)
        # This preserves YAML structure and formatting
        "$VENV_DIR/bin/python" << EOF 2>/dev/null || warn "Failed to update config.yaml (this is non-critical)"
import yaml
import sys

try:
    config_path = "$CONFIG_FILE"
    testnet_bool_str = "$TESTNET_BOOL"
    exchange_mode = "$EXCHANGE_MODE"
    
    # Convert string to boolean
    testnet_bool = testnet_bool_str.lower() in ('true', '1', 'yes', 'on')
    
    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Ensure exchange section exists
    if 'exchange' not in config:
        config['exchange'] = {}
    
    # Update testnet and mode
    config['exchange']['testnet'] = testnet_bool
    config['exchange']['mode'] = exchange_mode
    
    # Write back to file, preserving comments and structure as much as possible
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    sys.exit(0)
except Exception as e:
    print(f"Error updating config.yaml: {e}", file=sys.stderr)
    sys.exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            info "Updated config.yaml: testnet=$TESTNET_BOOL, mode=$EXCHANGE_MODE"
        else
            # Fallback to sed if Python fails
            warn "Python YAML update failed, using sed fallback..."
            if grep -q "^  testnet:" "$CONFIG_FILE" 2>/dev/null; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    sed -i '' "s/^  testnet:.*/  testnet: $TESTNET_BOOL/" "$CONFIG_FILE"
                else
                    sed -i "s/^  testnet:.*/  testnet: $TESTNET_BOOL/" "$CONFIG_FILE"
                fi
            fi
            if grep -q "^  mode:" "$CONFIG_FILE" 2>/dev/null; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    sed -i '' "s/^  mode:.*/  mode: $EXCHANGE_MODE/" "$CONFIG_FILE"
                else
                    sed -i "s/^  mode:.*/  mode: $EXCHANGE_MODE/" "$CONFIG_FILE"
                fi
            fi
        fi
    else
        if [ ! -f "$CONFIG_FILE" ]; then
            warn "config.yaml not found, cannot update testnet/mode settings"
        fi
    fi
    
    # Discord webhook
    if ! grep -q "^DISCORD_WEBHOOK_URL=" "$ENV_FILE" 2>/dev/null; then
        if [ "$NON_INTERACTIVE" != "true" ]; then
            echo -e "${YELLOW}Enter your Discord webhook URL (or press Enter to skip):${NC}"
            read -r DISCORD_WEBHOOK
            if [ -n "$DISCORD_WEBHOOK" ]; then
                echo "DISCORD_WEBHOOK_URL=$DISCORD_WEBHOOK" >> "$ENV_FILE"
                info "Discord webhook URL saved"
            fi
        else
            warn "DISCORD_WEBHOOK_URL not set in .env and NON_INTERACTIVE=true; skipping prompt."
        fi
    else
        info "Discord webhook URL already configured"
    fi
    
    echo ""
    
    # ============================================================
    # Step 10: Create systemd services
    # ============================================================
    step "10/11" "Creating systemd services..."
    
    # Main live bot service
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    
    info "Creating ${SERVICE_NAME}.service..."
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF || error_exit "Failed to create service file"
[Unit]
Description=Bybit Trading Bot - Live Trading
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$BOT_DIR
EnvironmentFile=$ENV_FILE
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m src.main --config $CONFIG_FILE live
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Optimizer service (runs both main strategy and funding strategy optimization)
    OPTIMIZER_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}-optimizer.service"
    info "Creating ${SERVICE_NAME}-optimizer.service..."
    sudo tee "$OPTIMIZER_SERVICE_FILE" > /dev/null <<EOF || error_exit "Failed to create optimizer service file"
[Unit]
Description=Bybit Trading Bot - Parameter Optimizer (Main + Funding Strategies)
After=network.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$BOT_DIR
EnvironmentFile=$ENV_FILE
Environment="PATH=$VENV_DIR/bin"
# Runs main strategy optimization, then funding strategy optimization (if enabled in config)
ExecStart=$VENV_DIR/bin/python -m src.main --config $CONFIG_FILE optimize --use-universe-history
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Optimizer timer
    OPTIMIZER_TIMER_FILE="/etc/systemd/system/${SERVICE_NAME}-optimizer.timer"
    info "Creating ${SERVICE_NAME}-optimizer.timer..."
    sudo tee "$OPTIMIZER_TIMER_FILE" > /dev/null <<EOF || error_exit "Failed to create optimizer timer file"
[Unit]
Description=Daily parameter optimization for Bybit Trading Bot
Requires=${SERVICE_NAME}-optimizer.service

[Timer]
OnCalendar=daily
UTC=true
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Universe build service (daily universe refresh)
    UNIVERSE_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}-universe.service"
    info "Creating ${SERVICE_NAME}-universe.service..."
    sudo tee "$UNIVERSE_SERVICE_FILE" > /dev/null <<EOF || error_exit "Failed to create universe service file"
[Unit]
Description=Bybit Trading Bot - Universe Builder
After=network.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$BOT_DIR
EnvironmentFile=$ENV_FILE
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m src.main --config $CONFIG_FILE universe-build
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Universe build timer
    UNIVERSE_TIMER_FILE="/etc/systemd/system/${SERVICE_NAME}-universe.timer"
    info "Creating ${SERVICE_NAME}-universe.timer..."
    sudo tee "$UNIVERSE_TIMER_FILE" > /dev/null <<EOF || error_exit "Failed to create universe timer file"
[Unit]
Description=Daily universe build for Bybit Trading Bot
Requires=${SERVICE_NAME}-universe.service

[Timer]
# Run a little after midnight UTC so data is fresh before the trading day
OnCalendar=01:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Report service
    REPORT_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}-report.service"
    info "Creating ${SERVICE_NAME}-report.service..."
    sudo tee "$REPORT_SERVICE_FILE" > /dev/null <<EOF || error_exit "Failed to create report service file"
[Unit]
Description=Bybit Trading Bot - Daily Discord Report
After=network.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$BOT_DIR
EnvironmentFile=$ENV_FILE
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m src.main --config $CONFIG_FILE report
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Report timer
    REPORT_TIMER_FILE="/etc/systemd/system/${SERVICE_NAME}-report.timer"
    info "Creating ${SERVICE_NAME}-report.timer..."
    sudo tee "$REPORT_TIMER_FILE" > /dev/null <<EOF || error_exit "Failed to create report timer file"
[Unit]
Description=Daily Discord report for Bybit Trading Bot
Requires=${SERVICE_NAME}-report.service

[Timer]
OnCalendar=09:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload || error_exit "Failed to reload systemd daemon"
    info "Systemd services created"
    echo ""
    
    # ============================================================
    # Step 11: Verify installation
    # ============================================================
    step "11/11" "Verifying installation..."
    
    # Verify Python entry point works
    info "Testing Python entry point..."
    if (cd "$BOT_DIR" && "$VENV_DIR/bin/python" -m src.main --help &>/dev/null); then
        info "✓ Entry point verified (CLI works)"
    else
        warn "Entry point test failed. CLI may not work correctly."
    fi
    
    # Verify systemd units are valid
    info "Verifying systemd units..."
    for unit in "$SERVICE_FILE" "$OPTIMIZER_SERVICE_FILE" "$OPTIMIZER_TIMER_FILE" "$UNIVERSE_SERVICE_FILE" "$UNIVERSE_TIMER_FILE" "$REPORT_SERVICE_FILE" "$REPORT_TIMER_FILE"; do
        if sudo systemctl cat "$(basename "$unit")" &>/dev/null; then
            info "✓ $(basename "$unit") is valid"
        else
            warn "$(basename "$unit") validation failed"
        fi
    done
    
    # Enable services (but don't start live bot automatically)
    info "Enabling services..."
    
    sudo systemctl enable "${SERVICE_NAME}.service" &>/dev/null || warn "Failed to enable ${SERVICE_NAME}.service"
    sudo systemctl enable "${SERVICE_NAME}-universe.timer" &>/dev/null || warn "Failed to enable ${SERVICE_NAME}-universe.timer"
    sudo systemctl enable "${SERVICE_NAME}-optimizer.timer" &>/dev/null || warn "Failed to enable ${SERVICE_NAME}-optimizer.timer"
    sudo systemctl enable "${SERVICE_NAME}-report.timer" &>/dev/null || warn "Failed to enable ${SERVICE_NAME}-report.timer"
    
    # Start timers (they're scheduled, won't run immediately unless time matches)
    sudo systemctl start "${SERVICE_NAME}-universe.timer" &>/dev/null || warn "Failed to start universe timer"
    sudo systemctl start "${SERVICE_NAME}-optimizer.timer" &>/dev/null || warn "Failed to start optimizer timer"
    sudo systemctl start "${SERVICE_NAME}-report.timer" &>/dev/null || warn "Failed to start report timer"
    
    info "Services enabled (live bot not started automatically - see next steps)"
    echo ""
    
    # ============================================================
    # Installation complete
    # ============================================================
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Installation Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Bot directory: $BOT_DIR"
    echo "Virtual environment: $VENV_DIR"
    echo "Config file: $CONFIG_FILE"
    echo "Environment file: $ENV_FILE"
    echo "Service name: $SERVICE_NAME"
    echo "Service user: $SERVICE_USER"
    echo ""
    
    # Show service status
    echo -e "${BLUE}Service Status:${NC}"
    if sudo systemctl is-enabled "${SERVICE_NAME}.service" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${SERVICE_NAME}.service: enabled"
    else
        echo -e "  ${RED}✗${NC} ${SERVICE_NAME}.service: not enabled"
    fi
    
    if sudo systemctl is-enabled "${SERVICE_NAME}-universe.timer" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${SERVICE_NAME}-universe.timer: enabled"
    else
        echo -e "  ${RED}✗${NC} ${SERVICE_NAME}-universe.timer: not enabled"
    fi
    
    if sudo systemctl is-enabled "${SERVICE_NAME}-optimizer.timer" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${SERVICE_NAME}-optimizer.timer: enabled"
    else
        echo -e "  ${RED}✗${NC} ${SERVICE_NAME}-optimizer.timer: not enabled"
    fi
    
    if sudo systemctl is-enabled "${SERVICE_NAME}-report.timer" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${SERVICE_NAME}-report.timer: enabled"
    else
        echo -e "  ${RED}✗${NC} ${SERVICE_NAME}-report.timer: not enabled"
    fi
    
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo ""
    echo "1. Review and edit configuration:"
    echo "   nano $CONFIG_FILE"
    echo ""
    echo "2. Review environment variables (API keys):"
    echo "   cat $ENV_FILE"
    echo ""
    echo "3. Run an initial universe build (recommended):"
    echo "   cd $BOT_DIR"
    echo "   source venv/bin/activate"
    echo "   python -m src.main universe-build --config config.yaml"
    echo ""
    echo "4. Test the bot in backtest mode:"
    echo "   cd $BOT_DIR"
    echo "   source venv/bin/activate"
    echo "   python -m src.main backtest --config config.yaml"
    echo ""
    echo "5. Test universe optimization (optional):"
    echo "   python -m src.main optimize-universe --config config.yaml --start 2023-01-01 --end 2024-01-01 --n-combinations 50"
    echo ""
    echo "6. Test parameter optimization (includes both main and funding strategies if enabled):"
    echo "   python -m src.main optimize --config config.yaml --use-universe-history"
    echo ""
    echo "7. Test funding strategy (if enabled in config):"
    echo "   python -m pytest tests/test_funding_opportunity.py -v"
    echo "   python -m pytest tests/test_funding_optimizer.py -v"
    echo ""
    echo "8. Run timeframe comparison analysis (optional, requires historical data):"
    echo "   python scripts/optimize_and_compare_timeframes.py --config config.yaml --start 2022-01-01 --end 2024-12-31"
    echo ""
    echo -e "${RED}IMPORTANT: Before starting live trading:${NC}"
    echo ""
    echo "   - Set exchange.mode to 'testnet' or 'paper' in config.yaml"
    echo "   - Test thoroughly in testnet/paper mode first"
    echo "   - Only set exchange.mode to 'live' when ready for real trading"
    echo ""
    echo "9. Start the bot (when ready):"
    echo "   sudo systemctl start $SERVICE_NAME"
    echo ""
    echo "10. Monitor the bot:"
    echo "   sudo systemctl status $SERVICE_NAME"
    echo "   sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo "11. Check scheduled tasks:"
    echo "   sudo systemctl list-timers ${SERVICE_NAME}-*"
    echo ""
    echo "12. Monitor optimizer runs (check logs for both main and funding optimization):"
    echo "   sudo journalctl -u ${SERVICE_NAME}-optimizer.service -f"
    echo "   # Look for 'MAIN STRATEGY OPTIMIZATION START' and 'FUNDING STRATEGY OPTIMIZATION START'"
    echo ""
    echo "13. Check funding optimizer results:"
    echo "   cat optimizer_results/funding_best.json"
    echo ""
    echo "14. Available analysis scripts:"
    echo "   - scripts/optimize_and_compare_timeframes.py - Compare timeframes with optimized parameters"
    echo "   - scripts/download_historical_data.py - Download historical OHLCV data"
    echo "   - scripts/funding_parameter_tune.py - Manual funding parameter tuning"
    echo ""
    echo -e "${GREEN}For more information, see README.md${NC}"
    echo ""
}

# Run main function
main "$@"
