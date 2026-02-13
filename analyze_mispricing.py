#!/usr/bin/env python3
"""
Phase 2: AI Mispricing Detection
=================================
Reads markets from scanner, asks Claude to estimate true probability,
flags significant deviations as mispricing opportunities.

Uses Smart Router (localhost:4001) for cost-efficient Claude access.
"""

import json
import requests
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Config ---
SMART_ROUTER = "http://localhost:4001/v1/messages"
MISPRICING_THRESHOLD = 0.15  # Flag if AI estimate differs by 15%+
MIN_CONFIDENCE = 0.6  # Only analyze if AI is confident enough
ANALYSIS_LIMIT = 10  # Analyze top N markets per run (cost control)
OUTPUT_FILE = Path("/home/ubuntu/.openclaw/workspace/notes/mispricing-opportunities.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mispricing] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("mispricing")


def call_claude(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Call Smart Router (localhost:4001) for Claude inference."""
    try:
        body = {
            "model": "claude-sonnet-4-5-20250929",  # Will auto-route to haiku if simple
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(SMART_ROUTER, json=body, timeout=60)
        if resp.status_code != 200:
            log.error(f"Smart Router returned {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "")
        return text.strip()
    except Exception as e:
        log.error(f"Claude call failed: {e}")
        return None


def analyze_market(market: dict) -> Optional[dict]:
    """
    Ask Claude to estimate true probability for a market.
    Returns: {
        "market": market dict,
        "ai_probability": float (0-1),
        "market_price": float,
        "deviation": float,
        "reasoning": str
    }
    """
    question = market.get("question", "")
    description = market.get("description", "")[:500]
    platform = market.get("platform", "unknown")
    yes_price = market.get("yes_price", 0)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""You are a prediction market analyst. Today is {today}.

Market question: "{question}"
Description: {description}
Platform: {platform}
Current market price (YES): ${yes_price:.2f} (implies {yes_price*100:.1f}% probability)

Your task:
1. Estimate the TRUE probability (0-100%) based on your knowledge
2. Rate your confidence (0-100%)
3. Briefly explain your reasoning (1-2 sentences)

Respond in this EXACT format:
PROBABILITY: [number]
CONFIDENCE: [number]
REASONING: [text]

Example:
PROBABILITY: 65
CONFIDENCE: 80
REASONING: Based on recent polls and historical trends, the outcome is likely but not certain.
"""
    
    response = call_claude(prompt, max_tokens=500)
    if not response:
        return None
    
    # Parse response
    try:
        lines = response.split("\n")
        prob_line = [l for l in lines if l.startswith("PROBABILITY:")][0]
        conf_line = [l for l in lines if l.startswith("CONFIDENCE:")][0]
        reasoning_line = [l for l in lines if l.startswith("REASONING:")][0]
        
        ai_prob = float(prob_line.split(":")[1].strip()) / 100.0
        confidence = float(conf_line.split(":")[1].strip()) / 100.0
        reasoning = reasoning_line.split(":", 1)[1].strip()
        
        deviation = abs(ai_prob - yes_price)
        
        return {
            "market": market,
            "ai_probability": ai_prob,
            "confidence": confidence,
            "market_price": yes_price,
            "deviation": deviation,
            "reasoning": reasoning,
        }
    except Exception as e:
        log.warning(f"Failed to parse Claude response: {e}")
        log.debug(f"Response was: {response[:200]}")
        return None


def load_markets() -> list:
    """Load markets from scanner output (if exists)."""
    # Scanner should save to a JSON file — we'll check common locations
    possible_paths = [
        Path("/home/ubuntu/.openclaw/workspace/notes/markets.json"),
        Path("/home/ubuntu/projects/prediction-market-ai/markets.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                log.info(f"Loaded {len(data)} markets from {path}")
                return data
            except Exception as e:
                log.warning(f"Failed to load {path}: {e}")
    
    log.warning("No market data found — scanner may not have run yet")
    return []


def save_opportunities(opportunities: list):
    """Save mispricing opportunities to markdown file."""
    if not opportunities:
        log.info("No mispricings found")
        return
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    content = f"# Mispricing Opportunities\n\n"
    content += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    content += f"Found {len(opportunities)} potential mispricings:\n\n"
    
    for i, opp in enumerate(opportunities, 1):
        market = opp["market"]
        content += f"## {i}. {market['question']}\n\n"
        content += f"- **Platform**: {market['platform']}\n"
        content += f"- **Market price**: ${opp['market_price']:.2f} ({opp['market_price']*100:.1f}%)\n"
        content += f"- **AI estimate**: {opp['ai_probability']*100:.1f}%\n"
        content += f"- **Deviation**: {opp['deviation']*100:.1f}%\n"
        content += f"- **AI confidence**: {opp['confidence']*100:.0f}%\n"
        content += f"- **Reasoning**: {opp['reasoning']}\n"
        content += f"- **URL**: {market.get('url', 'N/A')}\n\n"
    
    OUTPUT_FILE.write_text(content)
    log.info(f"Saved {len(opportunities)} opportunities to {OUTPUT_FILE}")


def main():
    log.info("=== Phase 2: Mispricing Analysis ===")
    
    # Load markets
    markets = load_markets()
    if not markets:
        log.error("No markets to analyze — exiting")
        return
    
    # Filter: only high-volume/liquidity markets
    filtered = [
        m for m in markets
        if m.get("volume_24h", 0) >= 500 and m.get("liquidity", 0) >= 1000
    ]
    
    log.info(f"Filtered to {len(filtered)} high-activity markets")
    
    # Sort by volume, take top N
    sorted_markets = sorted(filtered, key=lambda m: m.get("volume_24h", 0), reverse=True)
    to_analyze = sorted_markets[:ANALYSIS_LIMIT]
    
    log.info(f"Analyzing top {len(to_analyze)} markets...")
    
    opportunities = []
    
    for i, market in enumerate(to_analyze, 1):
        log.info(f"[{i}/{len(to_analyze)}] Analyzing: {market['question'][:60]}...")
        
        result = analyze_market(market)
        if not result:
            continue
        
        # Check if confident + significant deviation
        if result["confidence"] >= MIN_CONFIDENCE and result["deviation"] >= MISPRICING_THRESHOLD:
            opportunities.append(result)
            log.info(f"  → MISPRICING: AI={result['ai_probability']*100:.1f}%, Market={result['market_price']*100:.1f}%, Δ={result['deviation']*100:.1f}%")
        else:
            log.info(f"  → OK: AI={result['ai_probability']*100:.1f}%, Market={result['market_price']*100:.1f}%")
        
        # Rate limit (1 req/sec for Smart Router)
        time.sleep(1.5)
    
    # Save results
    save_opportunities(opportunities)
    
    if opportunities:
        log.info(f"✅ Found {len(opportunities)} mispricing opportunities!")
    else:
        log.info("No significant mispricings detected")


if __name__ == "__main__":
    main()
