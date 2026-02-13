#!/usr/bin/env python3
"""
Prediction Market Monitor
=========================
Checks last analysis results and alerts on high-deviation opportunities.
Runs via cron after daily analysis.
"""

import json
import os
import requests
from pathlib import Path
from datetime import datetime

OPPORTUNITIES_FILE = Path("/home/ubuntu/.openclaw/workspace/notes/mispricing-opportunities.md")
ALERT_THRESHOLD = 0.50  # Alert if deviation >50%
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TG_CHAT_ID", "404662067")


def send_telegram(message: str):
    """Send Telegram alert."""
    if not TELEGRAM_BOT_TOKEN:
        print("No Telegram token, skipping alert")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
        if resp.status_code == 200:
            print("Telegram alert sent")
        else:
            print(f"Telegram failed: {resp.status_code}")
    except Exception as e:
        print(f"Telegram error: {e}")


def parse_opportunities():
    """Parse mispricing opportunities from markdown file."""
    if not OPPORTUNITIES_FILE.exists():
        return []
    
    content = OPPORTUNITIES_FILE.read_text()
    opportunities = []
    
    # Simple parsing: look for deviation lines
    for line in content.split("\n"):
        if line.startswith("- **Deviation**:"):
            try:
                pct_str = line.split(":")[1].strip().rstrip("%")
                deviation = float(pct_str) / 100.0
                # Get question from previous lines (rough parsing)
                opportunities.append({"deviation": deviation, "line": line})
            except:
                continue
    
    return opportunities


def main():
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Monitor check")
    
    opps = parse_opportunities()
    if not opps:
        print("No opportunities found")
        return
    
    # Find high-deviation opportunities
    high_dev = [o for o in opps if o["deviation"] >= ALERT_THRESHOLD]
    
    if high_dev:
        msg = f"🚨 **High Mispricing Alert**\n\n"
        msg += f"Found {len(high_dev)} opportunities with >50% deviation!\n\n"
        msg += f"Check: /home/ubuntu/.openclaw/workspace/notes/mispricing-opportunities.md"
        send_telegram(msg)
        print(f"Alert sent: {len(high_dev)} high-deviation opportunities")
    else:
        print(f"No high-deviation opportunities ({len(opps)} total)")


if __name__ == "__main__":
    main()
