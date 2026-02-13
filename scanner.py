#!/usr/bin/env python3
"""
Polymarket Arbitrage & Mispricing Scanner
- Intra-market: YES + NO < $1.00 (guaranteed profit)
- AI mispricing: LLM judges if market price seems off vs. common knowledge
- Korea-friendly: Polymarket only (crypto wallet, no geo-restriction)
"""

import requests
import time
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SCAN_INTERVAL = 300  # 5 min
INTRA_SPREAD_THRESHOLD = 0.02  # YES+NO < $0.98 → 2%+ guaranteed profit
MISPRICING_THRESHOLD = 0.15  # AI thinks price is off by 15%+
MIN_LIQUIDITY = 1000  # Skip illiquid markets
MIN_VOLUME_24H = 500  # Skip dead markets

LOG_FILE = Path("/home/ubuntu/.openclaw/workspace/notes/arbitrage-log.md")
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"

# Smart Router for AI mispricing detection
SMART_ROUTER = "http://localhost:4001/v1/messages"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("poly-scanner")


# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────

def fetch_markets(limit=500):
    """Fetch active Polymarket markets from Gamma API."""
    markets = []
    offset = 0
    while len(markets) < limit:
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={"closed": "false", "limit": 100, "offset": offset},
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=30
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            markets.extend(batch)
            offset += 100
            if len(batch) < 100:
                break
        except Exception as e:
            log.error(f"Fetch error at offset {offset}: {e}")
            break
    return markets


def fetch_events(limit=200):
    """Fetch active events (groups of related markets)."""
    events = []
    offset = 0
    while len(events) < limit:
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_API}/events",
                params={"closed": "false", "limit": 100, "offset": offset},
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=30
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            events.extend(batch)
            offset += 100
            if len(batch) < 100:
                break
        except Exception as e:
            log.error(f"Events fetch error: {e}")
            break
    return events


def parse_market(m):
    """Parse a raw Polymarket market into standardized format."""
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        if len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
        return {
            "id": m.get("id"),
            "question": m.get("question", ""),
            "slug": m.get("slug", ""),
            "yes_price": yes_price,
            "no_price": no_price,
            "total": round(yes_price + no_price, 6),
            "spread": round(1.0 - yes_price - no_price, 6),
            "volume": m.get("volumeNum", 0),
            "volume_24h": m.get("volume24hr", 0),
            "liquidity": m.get("liquidityNum", 0),
            "description": (m.get("description") or "")[:500],
            "end_date": m.get("endDate", ""),
            "group_item_title": m.get("groupItemTitle", ""),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# Strategy 1: Intra-market Arbitrage (YES + NO < $1)
# ──────────────────────────────────────────────

def find_spread_arbs(markets):
    """Find markets where YES + NO < $1 (guaranteed profit if you buy both)."""
    arbs = []
    for m in markets:
        if m["spread"] > INTRA_SPREAD_THRESHOLD and m["liquidity"] >= MIN_LIQUIDITY:
            profit_per_dollar = m["spread"] / m["total"] if m["total"] > 0 else 0
            arbs.append({
                "type": "spread",
                "question": m["question"],
                "slug": m["slug"],
                "yes_price": m["yes_price"],
                "no_price": m["no_price"],
                "total": m["total"],
                "spread": m["spread"],
                "profit_pct": round(profit_per_dollar * 100, 2),
                "liquidity": m["liquidity"],
                "volume_24h": m["volume_24h"],
            })
    arbs.sort(key=lambda x: -x["spread"])
    return arbs


# ──────────────────────────────────────────────
# Strategy 2: AI Mispricing Detection
# ──────────────────────────────────────────────

def ai_evaluate_batch(markets, batch_size=10):
    """Use LLM to evaluate if market prices seem mispriced."""
    # Filter to interesting markets (enough volume, binary yes/no)
    candidates = [
        m for m in markets
        if m["volume_24h"] >= MIN_VOLUME_24H
        and m["liquidity"] >= MIN_LIQUIDITY
        and 0.05 < m["yes_price"] < 0.95  # Skip near-certain outcomes
    ]
    
    if not candidates:
        return []
    
    # Take top markets by volume for AI analysis
    candidates.sort(key=lambda x: -x["volume_24h"])
    candidates = candidates[:batch_size * 3]  # Pre-filter pool
    
    # Build batch prompt
    market_list = []
    for i, m in enumerate(candidates[:batch_size]):
        market_list.append(
            f"{i+1}. \"{m['question']}\" — Current YES price: ${m['yes_price']:.2f} "
            f"(implies {m['yes_price']*100:.0f}% probability) "
            f"[Vol24h: ${m['volume_24h']:,.0f}, Liq: ${m['liquidity']:,.0f}]"
        )
    
    prompt = f"""You are a prediction market analyst. Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.

Analyze these Polymarket prices. For each, estimate what you think the TRUE probability should be based on your knowledge. Flag any where the market price seems significantly wrong (off by 15%+ from your estimate).

Markets:
{chr(10).join(market_list)}

Respond in JSON array format only. For each market:
{{"idx": 1, "market_prob": 0.65, "my_estimate": 0.45, "diff": 0.20, "reasoning": "brief reason", "direction": "overpriced"}}

Only include markets where |diff| >= 0.15. If none are mispriced, return [].
Return ONLY the JSON array, no other text."""

    try:
        # Try Smart Router first
        resp = requests.post(
            SMART_ROUTER,
            json={
                "model": "light",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # Support both OpenAI and Anthropic response formats
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        elif "content" in data:
            texts = data["content"]
            content = texts[0]["text"] if isinstance(texts, list) else texts
        else:
            content = "[]"
        
        # Check for router failure message
        if "실패" in content or "error" in content.lower()[:20]:
            log.warning("Smart Router returned error, skipping AI analysis")
            return []
        
        # Extract JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?\s*", "", content).rstrip("`").strip()
        
        results = json.loads(content)
        if not isinstance(results, list):
            return []
        
        # Map back to markets
        mispricings = []
        for r in results:
            idx = r.get("idx", 0) - 1
            if 0 <= idx < len(candidates):
                m = candidates[idx]
                diff = abs(r.get("diff", 0))
                if diff >= MISPRICING_THRESHOLD:
                    mispricings.append({
                        "type": "mispricing",
                        "question": m["question"],
                        "slug": m["slug"],
                        "market_price": m["yes_price"],
                        "ai_estimate": r.get("my_estimate", 0),
                        "diff": round(diff, 4),
                        "diff_pct": round(diff * 100, 2),
                        "direction": r.get("direction", "unknown"),
                        "reasoning": r.get("reasoning", ""),
                        "volume_24h": m["volume_24h"],
                        "liquidity": m["liquidity"],
                    })
        
        mispricings.sort(key=lambda x: -x["diff"])
        return mispricings
    
    except Exception as e:
        log.warning(f"AI evaluation failed: {e}")
        return []


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def write_log(spread_arbs, mispricings, scan_time, market_count):
    """Append findings to markdown log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("# Polymarket Arbitrage & Mispricing Log\n\n")
            f.write("Auto-generated by prediction-market-scanner.py\n\n---\n\n")

    total = len(spread_arbs) + len(mispricings)
    
    with open(LOG_FILE, "a") as f:
        f.write(f"## Scan: {scan_time}\n\n")
        f.write(f"- Markets scanned: {market_count}\n")
        f.write(f"- Spread arbs: {len(spread_arbs)} | AI mispricings: {len(mispricings)}\n\n")

        if not total:
            f.write("_No opportunities found._\n\n---\n\n")
            return

        # Spread arbitrage
        for i, a in enumerate(spread_arbs, 1):
            f.write(f"### 💰 Spread #{i}: {a['question'][:80]}\n\n")
            f.write(f"- YES: ${a['yes_price']:.4f} + NO: ${a['no_price']:.4f} = **${a['total']:.4f}**\n")
            f.write(f"- Spread: ${a['spread']:.4f} → **{a['profit_pct']:.2f}% guaranteed profit**\n")
            f.write(f"- Liquidity: ${a['liquidity']:,.0f} | 24h Vol: ${a['volume_24h']:,.0f}\n")
            f.write(f"- 🔗 https://polymarket.com/event/{a['slug']}\n\n")

        # AI mispricings
        for i, a in enumerate(mispricings, 1):
            emoji = "📈" if a["direction"] == "underpriced" else "📉"
            action = "BUY YES" if a["direction"] == "underpriced" else "SELL YES / BUY NO"
            f.write(f"### {emoji} Mispricing #{i}: {a['question'][:80]}\n\n")
            f.write(f"- Market: ${a['market_price']:.2f} ({a['market_price']*100:.0f}%) → AI estimate: ${a['ai_estimate']:.2f} ({a['ai_estimate']*100:.0f}%)\n")
            f.write(f"- **Diff: {a['diff_pct']:.1f}%** — {a['direction']}\n")
            f.write(f"- Reasoning: {a['reasoning']}\n")
            f.write(f"- Action: **{action}**\n")
            f.write(f"- Liquidity: ${a['liquidity']:,.0f} | 24h Vol: ${a['volume_24h']:,.0f}\n")
            f.write(f"- 🔗 https://polymarket.com/event/{a['slug']}\n\n")

        f.write("---\n\n")

    # Trim log if too large (keep last 500KB)
    if LOG_FILE.stat().st_size > 500_000:
        content = LOG_FILE.read_text()
        # Keep header + last ~400KB
        header_end = content.find("---\n\n") + 5
        header = content[:header_end]
        rest = content[header_end:]
        trimmed = rest[-400_000:]
        # Find clean break point
        break_point = trimmed.find("\n## Scan:")
        if break_point > 0:
            trimmed = trimmed[break_point + 1:]
        LOG_FILE.write_text(header + trimmed)


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────

def run_scan():
    """Execute one scan cycle."""
    scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(f"Starting scan at {scan_time}")

    # Fetch & parse
    raw = fetch_markets(limit=500)
    markets = [m for m in (parse_market(r) for r in raw) if m]
    log.info(f"Fetched {len(raw)} raw → {len(markets)} parsed markets")

    # Strategy 1: Spread arbs
    spread_arbs = find_spread_arbs(markets)
    log.info(f"Spread arbs: {len(spread_arbs)}")

    # Strategy 2: AI mispricing (every scan)
    mispricings = ai_evaluate_batch(markets, batch_size=15)
    log.info(f"AI mispricings: {len(mispricings)}")

    # Log
    write_log(spread_arbs, mispricings, scan_time, len(markets))
    return spread_arbs, mispricings


def main():
    log.info("Polymarket Scanner started (Korea-friendly, single platform)")
    log.info(f"Interval: {SCAN_INTERVAL}s | Spread threshold: {INTRA_SPREAD_THRESHOLD*100}% | AI threshold: {MISPRICING_THRESHOLD*100}%")

    while True:
        try:
            spreads, misps = run_scan()
            total = len(spreads) + len(misps)
            if total:
                log.info(f"⚡ {total} opportunities logged!")
            else:
                log.info("No opportunities this scan.")
        except Exception as e:
            log.error(f"Scan error: {e}", exc_info=True)

        log.info(f"Sleeping {SCAN_INTERVAL}s...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
