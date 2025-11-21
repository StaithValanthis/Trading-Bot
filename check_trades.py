#!/usr/bin/env python3
"""Check recent trades, orders, and diagnose why trades might be skipped."""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

def check_database(db_path: Path):
    """Check trades and orders in database."""
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        print(f"   Expected path: {db_path.absolute()}")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get recent closed trades (last 7 days)
    print("=" * 80)
    print("📊 RECENT CLOSED TRADES (Last 7 Days)")
    print("=" * 80)
    cursor.execute("""
        SELECT 
            symbol, 
            side, 
            size, 
            entry_price, 
            exit_price, 
            exit_time, 
            pnl, 
            reason 
        FROM trades 
        WHERE datetime(exit_time) >= datetime('now', '-7 days')
        ORDER BY exit_time DESC
    """)
    trades = cursor.fetchall()
    if trades:
        print(f"\nFound {len(trades)} closed trades:\n")
        print(f"{'Exit Time':<20} {'Symbol':<12} {'Side':<6} {'Size':<10} "
              f"{'Entry $':<10} {'Exit $':<10} {'PnL $':<12} {'Reason':<20}")
        print("-" * 80)
        for trade in trades:
            symbol, side, size, entry, exit_p, exit_time, pnl, reason = trade
            pnl_str = f"${pnl:+.2f}" if pnl else "$0.00"
            print(f"{exit_time:<20} {symbol:<12} {side:<6} {size:<10.4f} "
                  f"${entry:<9.2f} ${exit_p:<9.2f} {pnl_str:<12} {reason or 'N/A':<20}")
    else:
        print("\n⚠️  No closed trades in the last 7 days")
    
    # Get recent orders (last 7 days)
    print("\n" + "=" * 80)
    print("📝 RECENT ORDERS (Last 7 Days)")
    print("=" * 80)
    cursor.execute("""
        SELECT 
            symbol, 
            side, 
            size, 
            price, 
            order_type, 
            reason, 
            created_at 
        FROM orders 
        WHERE datetime(created_at) >= datetime('now', '-7 days')
        ORDER BY created_at DESC
    """)
    orders = cursor.fetchall()
    if orders:
        print(f"\nFound {len(orders)} orders:\n")
        print(f"{'Created At':<20} {'Symbol':<12} {'Side':<6} {'Size':<10} "
              f"{'Price $':<10} {'Type':<12} {'Reason':<20}")
        print("-" * 80)
        for order in orders:
            symbol, side, size, price, order_type, reason, created_at = order
            price_str = f"${price:.2f}" if price else "N/A"
            print(f"{created_at:<20} {symbol:<12} {side:<6} {size:<10.4f} "
                  f"{price_str:<10} {order_type:<12} {reason or 'N/A':<20}")
    else:
        print("\n⚠️  No orders in the last 7 days")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS (Last 7 Days)")
    print("=" * 80)
    
    # Total trades
    cursor.execute("""
        SELECT COUNT(*), SUM(pnl), AVG(pnl)
        FROM trades 
        WHERE datetime(exit_time) >= datetime('now', '-7 days')
    """)
    count, total_pnl, avg_pnl = cursor.fetchone()
    count = count or 0
    total_pnl = total_pnl or 0.0
    avg_pnl = avg_pnl or 0.0
    
    print(f"\nTotal Trades: {count}")
    if count > 0:
        print(f"Total PnL: ${total_pnl:+.2f}")
        print(f"Average PnL per Trade: ${avg_pnl:+.2f}")
    
    # Total orders
    cursor.execute("""
        SELECT COUNT(*)
        FROM orders 
        WHERE datetime(created_at) >= datetime('now', '-7 days')
    """)
    order_count = cursor.fetchone()[0] or 0
    print(f"Total Orders: {order_count}")
    
    conn.close()
    return True

def main():
    """Main function."""
    # Default database path (adjust if different)
    db_path = Path("data/trading_bot.db")
    
    # Allow override via command line
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    
    print("\n🔍 TRADING BOT - TRADE DIAGNOSTICS")
    print("=" * 80)
    print(f"Checking database: {db_path.absolute()}\n")
    
    if not check_database(db_path):
        print("\n💡 TIP: Check your config.yaml for the correct data.db_path")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("💡 NEXT STEPS")
    print("=" * 80)
    print("""
1. Check systemd logs for position execution:
   sudo journalctl -u bybit-bot -n 500 | grep -E "Executed|Position updated|Skipping"

2. Check recent rebalance cycles:
   sudo journalctl -u bybit-bot -n 200 | grep -E "REBALANCING|Selected symbols|Reconciling"

3. Check current portfolio state:
   sudo journalctl -u bybit-bot -n 50 | grep -E "Heartbeat|positions="

4. If positions are being skipped due to "already exists", this is expected behavior
   to prevent stacking positions for the same symbol.

5. See CHECK_TRADES_GUIDE.md for detailed instructions.
""")
    print("=" * 80)

if __name__ == "__main__":
    main()

