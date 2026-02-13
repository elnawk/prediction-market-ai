#!/usr/bin/env python3
"""
One-shot market collector: run scanner once and save to JSON.
"""

import sys
import json
from pathlib import Path

# Import scanner functions
sys.path.insert(0, str(Path(__file__).parent))
from prediction_market_scanner import (
    fetch_polymarket_markets,
    fetch_limitless_markets,
    fetch_predict_markets,
    log
)

OUTPUT_FILE = Path("/home/ubuntu/.openclaw/workspace/notes/markets.json")


def main():
    log.info("Collecting markets (one-shot)...")
    
    all_markets = []
    
    # Polymarket
    log.info("Fetching Polymarket...")
    poly = fetch_polymarket_markets()
    all_markets.extend(poly)
    log.info(f"  → {len(poly)} markets")
    
    # Limitless
    log.info("Fetching Limitless...")
    lim = fetch_limitless_markets()
    all_markets.extend(lim)
    log.info(f"  → {len(lim)} markets")
    
    # Predict.fun
    log.info("Fetching Predict.fun...")
    pred = fetch_predict_markets()
    all_markets.extend(pred)
    log.info(f"  → {len(pred)} markets")
    
    log.info(f"Total: {len(all_markets)} markets collected")
    
    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(all_markets, indent=2))
    log.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
