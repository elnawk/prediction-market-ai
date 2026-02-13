#!/usr/bin/env python3
"""
Multi-Platform Prediction Market Scanner
=========================================
Platforms: Polymarket, Kalshi, Limitless, Predict.fun
Strategies:
  1. Intra-market spread arbitrage (YES + NO < $1)
  2. Cross-platform arbitrage (same event, different prices across 4 platforms)
  3. AI mispricing detection (LLM judges if price seems off)

All public API endpoints — no API keys required.
"""

import requests
import time
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations
from difflib import SequenceMatcher

# --- Config ---
SCAN_INTERVAL = 300  # 5 min
INTRA_SPREAD_THRESHOLD = 0.02  # YES+NO < $0.98
CROSS_ARB_THRESHOLD = 0.05  # 5% price diff across platforms
MISPRICING_THRESHOLD = 0.15  # AI thinks price is off by 15%+
MIN_LIQUIDITY = 1000
MIN_VOLUME_24H = 500
SIMILARITY_THRESHOLD = 0.65  # For fuzzy matching market titles

LOG_FILE = Path("/home/ubuntu/.openclaw/workspace/notes/arbitrage-log.md")
SMART_ROUTER = "http://localhost:4001/v1/messages"

# --- API Endpoints ---
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
LIMITLESS_API = "https://api.limitless.exchange"
PREDICT_API = "https://api-testnet.predict.fun"  # Testnet (no API key needed)

# Rate limit delays (seconds between requests per platform)
RATE_LIMITS = {
    "polymarket": 0.2,
    "limitless": 0.3,
    "predict": 0.3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("multi-scanner")


# ══════════════════════════════════════════════
# Standardized Market Format
# ══════════════════════════════════════════════

def make_market(platform, id_, question, yes_price, no_price, volume=0,
                volume_24h=0, liquidity=0, description="", end_date="",
                url="", slug="", extra=None):
    """Create a standardized market dict."""
    return {
        "platform": platform,
        "id": id_,
        "question": question,
        "yes_price": yes_price,
        "no_price": no_price,
        "total": round(yes_price + no_price, 6),
        "spread": round(1.0 - yes_price - no_price, 6),
        "volume": volume,
        "volume_24h": volume_24h,
        "liquidity": liquidity,
        "description": description[:500],
        "end_date": end_date,
        "url": url,
        "slug": slug,
        "extra": extra or {},
    }


# ══════════════════════════════════════════════
# Platform Fetchers
# ══════════════════════════════════════════════

def fetch_polymarket(limit=500):
    """Fetch active Polymarket markets."""
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
            for m in batch:
                parsed = _parse_polymarket(m)
                if parsed:
                    markets.append(parsed)
            offset += 100
            if len(batch) < 100:
                break
            time.sleep(RATE_LIMITS["polymarket"])
        except Exception as e:
            log.error(f"Polymarket fetch error at offset {offset}: {e}")
            break
    log.info(f"Polymarket: {len(markets)} markets")
    return markets


def _parse_polymarket(m):
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        if len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
        slug = m.get("slug", "")
        return make_market(
            platform="polymarket",
            id_=m.get("id"),
            question=m.get("question", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=m.get("volumeNum", 0),
            volume_24h=m.get("volume24hr", 0),
            liquidity=m.get("liquidityNum", 0),
            description=m.get("description", ""),
            end_date=m.get("endDate", ""),
            url=f"https://polymarket.com/event/{slug}",
            slug=slug,
            extra={"conditionId": m.get("conditionId", "")},
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def fetch_limitless(limit=500):
    """Fetch active Limitless markets."""
    markets = []
    try:
        resp = requests.get(
            f"{LIMITLESS_API}/markets/active",
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else data
        for m in raw[:limit]:
            parsed = _parse_limitless(m)
            if parsed:
                markets.append(parsed)
    except Exception as e:
        log.error(f"Limitless fetch error: {e}")
    log.info(f"Limitless: {len(markets)} markets")
    return markets


def _parse_limitless(m):
    try:
        prices = m.get("prices", [])
        if len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
        slug = m.get("slug", "")
        vol = m.get("volume", "0")
        volume = float(vol) if vol else 0
        return make_market(
            platform="limitless",
            id_=m.get("id"),
            question=m.get("title", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            volume_24h=0,  # Not provided directly
            liquidity=0,
            description=m.get("description", ""),
            end_date=m.get("expirationDate", ""),
            url=f"https://limitless.exchange/markets/{slug}",
            slug=slug,
            extra={
                "conditionId": m.get("conditionId", ""),
                "tradeType": m.get("tradeType", ""),
                "collateral": m.get("collateralToken", {}).get("symbol", ""),
                "metadata": m.get("metadata", {}),
            },
        )
    except (ValueError, TypeError):
        return None


def fetch_predict(limit=500):
    """Fetch active Predict.fun markets (testnet, no API key)."""
    markets = []
    cursor = None
    while len(markets) < limit:
        try:
            params = {"first": 100}
            if cursor:
                params["after"] = cursor
            resp = requests.get(
                f"{PREDICT_API}/v1/markets",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                break
            batch = data.get("data", [])
            if not batch:
                break
            for m in batch:
                parsed = _parse_predict(m)
                if parsed:
                    markets.append(parsed)
            cursor = data.get("cursor")
            if not cursor or len(batch) < 100:
                break
            time.sleep(RATE_LIMITS["predict"])
        except Exception as e:
            log.error(f"Predict.fun fetch error: {e}")
            break
    log.info(f"Predict.fun: {len(markets)} markets")
    return markets


def _parse_predict(m):
    try:
        if m.get("status") not in ("REGISTERED", "FUNDED"):
            return None
        # Predict doesn't return prices in the list endpoint — fetch orderbook
        # For efficiency, we'll estimate from available data or fetch top-of-book
        market_id = m.get("id")
        yes_price, no_price = _predict_get_prices(market_id)
        if yes_price is None:
            return None
        return make_market(
            platform="predict",
            id_=market_id,
            question=m.get("question", "") or m.get("title", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=0,
            volume_24h=0,
            liquidity=0,
            description=m.get("description", ""),
            end_date="",
            url=f"https://predict.fun/market/{market_id}",
            slug=str(market_id),
            extra={
                "polymarketConditionIds": m.get("polymarketConditionIds", []),
                "kalshiMarketTicker": m.get("kalshiMarketTicker"),
                "categorySlug": m.get("categorySlug", ""),
                "isNegRisk": m.get("isNegRisk", False),
            },
        )
    except (ValueError, TypeError):
        return None


def _predict_get_prices(market_id):
    """Get best bid/ask prices from Predict.fun orderbook."""
    try:
        resp = requests.get(
            f"{PREDICT_API}/v1/markets/{market_id}/orderbook",
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            return None, None
        ob = data["data"]
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        # Best bid = highest bid price (YES price estimate)
        # Best ask = lowest ask price
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 1
        # Mid price as YES price
        if best_bid > 0 and best_ask < 1:
            yes_price = round((best_bid + best_ask) / 2, 4)
        elif best_bid > 0:
            yes_price = best_bid
        elif best_ask < 1:
            yes_price = best_ask
        else:
            return None, None
        no_price = round(1.0 - yes_price, 4)
        return yes_price, no_price
    except Exception:
        return None, None


# ══════════════════════════════════════════════
# Strategy 1: Intra-market Spread Arbitrage
# ══════════════════════════════════════════════

def find_spread_arbs(all_markets):
    """Find markets where YES + NO < $1."""
    arbs = []
    for m in all_markets:
        if m["spread"] > INTRA_SPREAD_THRESHOLD:
            # Skip liquidity filter for non-Polymarket (data may not be available)
            if m["platform"] == "polymarket" and m["liquidity"] < MIN_LIQUIDITY:
                continue
            profit_per_dollar = m["spread"] / m["total"] if m["total"] > 0 else 0
            arbs.append({
                "type": "spread",
                "platform": m["platform"],
                "question": m["question"],
                "url": m["url"],
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


# ══════════════════════════════════════════════
# Strategy 2: Cross-Platform Arbitrage
# ══════════════════════════════════════════════

def normalize_question(q):
    """Normalize a question string for matching."""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q)
    # Remove common prefixes
    for prefix in ["will ", "will the ", "is ", "does ", "do "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
    return q


def similarity(a, b):
    """Fuzzy string similarity."""
    return SequenceMatcher(None, a, b).ratio()


def match_markets_by_condition_id(all_markets):
    """Match markets across platforms using condition IDs (Predict.fun provides polymarketConditionIds)."""
    matches = []
    # Build conditionId index for Polymarket
    poly_by_condition = {}
    for m in all_markets:
        if m["platform"] == "polymarket":
            cid = m["extra"].get("conditionId", "")
            if cid:
                poly_by_condition[cid] = m

    # Match Predict.fun markets that reference Polymarket conditionIds
    for m in all_markets:
        if m["platform"] == "predict":
            for pcid in m["extra"].get("polymarketConditionIds", []):
                if pcid in poly_by_condition:
                    matches.append((m, poly_by_condition[pcid]))

    return matches


def find_cross_platform_arbs(all_markets):
    """Find same-event price discrepancies across platforms."""
    arbs = []

    # Group markets by platform
    by_platform = {}
    for m in all_markets:
        by_platform.setdefault(m["platform"], []).append(m)

    # Method 1: Condition ID matching (Predict <-> Polymarket)
    id_matches = match_markets_by_condition_id(all_markets)
    for m1, m2 in id_matches:
        arb = _check_cross_arb(m1, m2)
        if arb:
            arbs.append(arb)

    # Method 2: Fuzzy title matching across all platform pairs
    platforms = list(by_platform.keys())
    seen_pairs = set()  # Avoid duplicates from ID matching

    for p1, p2 in combinations(platforms, 2):
        for m1 in by_platform[p1]:
            q1 = normalize_question(m1["question"])
            if len(q1) < 10:
                continue
            for m2 in by_platform[p2]:
                pair_key = (str(m1["id"]), str(m2["id"]),
                            m1["platform"], m2["platform"])
                if pair_key in seen_pairs:
                    continue
                q2 = normalize_question(m2["question"])
                if len(q2) < 10:
                    continue
                sim = similarity(q1, q2)
                if sim >= SIMILARITY_THRESHOLD:
                    seen_pairs.add(pair_key)
                    arb = _check_cross_arb(m1, m2, sim)
                    if arb:
                        arbs.append(arb)

    arbs.sort(key=lambda x: -x["diff"])
    return arbs


def _check_cross_arb(m1, m2, similarity_score=1.0):
    """Check if two markets have an arbitrage opportunity."""
    # Skip if either price is near 0 or 1 (likely stale/illiquid)
    for m in (m1, m2):
        if m["yes_price"] < 0.01 or m["yes_price"] > 0.99:
            return None
    diff = abs(m1["yes_price"] - m2["yes_price"])
    if diff < CROSS_ARB_THRESHOLD:
        return None

    # Determine which to buy/sell
    if m1["yes_price"] < m2["yes_price"]:
        buy_platform, sell_platform = m1, m2
    else:
        buy_platform, sell_platform = m2, m1

    return {
        "type": "cross_platform",
        "question": m1["question"][:120],
        "buy": {
            "platform": buy_platform["platform"],
            "yes_price": buy_platform["yes_price"],
            "url": buy_platform["url"],
        },
        "sell": {
            "platform": sell_platform["platform"],
            "yes_price": sell_platform["yes_price"],
            "url": sell_platform["url"],
        },
        "diff": round(diff, 4),
        "diff_pct": round(diff * 100, 2),
        "similarity": round(similarity_score, 3),
    }


# ══════════════════════════════════════════════
# Strategy 3: AI Mispricing Detection
# ══════════════════════════════════════════════

def ai_evaluate_batch(markets, batch_size=15):
    """Use LLM to evaluate if market prices seem mispriced."""
    candidates = [
        m for m in markets
        if (m["volume_24h"] >= MIN_VOLUME_24H or m["platform"] != "polymarket")
        and 0.05 < m["yes_price"] < 0.95
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda x: -(x["volume_24h"] or x["volume"]))
    candidates = candidates[:batch_size * 3]

    market_list = []
    for i, m in enumerate(candidates[:batch_size]):
        market_list.append(
            f"{i+1}. [{m['platform'].upper()}] \"{m['question']}\" — YES: ${m['yes_price']:.2f} "
            f"({m['yes_price']*100:.0f}% implied)"
        )

    prompt = f"""You are a prediction market analyst. Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.

Analyze these prediction market prices from multiple platforms. For each, estimate the TRUE probability based on your knowledge. Flag any where the market price seems significantly wrong (off by 15%+).

Markets:
{chr(10).join(market_list)}

Respond in JSON array format only. For each mispriced market:
{{"idx": 1, "market_prob": 0.65, "my_estimate": 0.45, "diff": 0.20, "reasoning": "brief reason", "direction": "overpriced"}}

Only include markets where |diff| >= 0.15. If none are mispriced, return [].
Return ONLY the JSON array, no other text."""

    try:
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
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        elif "content" in data:
            texts = data["content"]
            content = texts[0]["text"] if isinstance(texts, list) else texts
        else:
            content = "[]"

        if "실패" in content or "error" in content.lower()[:20]:
            log.warning("Smart Router returned error, skipping AI analysis")
            return []

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?\s*", "", content).rstrip("`").strip()

        results = json.loads(content)
        if not isinstance(results, list):
            return []

        mispricings = []
        for r in results:
            idx = r.get("idx", 0) - 1
            if 0 <= idx < len(candidates):
                m = candidates[idx]
                diff = abs(r.get("diff", 0))
                if diff >= MISPRICING_THRESHOLD:
                    mispricings.append({
                        "type": "mispricing",
                        "platform": m["platform"],
                        "question": m["question"],
                        "url": m["url"],
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


# ══════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════

def write_log(spread_arbs, cross_arbs, mispricings, scan_time, counts):
    """Append findings to markdown log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("# Multi-Platform Prediction Market Arbitrage Log\n\n")
            f.write("Platforms: Polymarket, Limitless, Predict.fun\n")
            f.write("Auto-generated by prediction-market-scanner.py\n\n---\n\n")

    total = len(spread_arbs) + len(cross_arbs) + len(mispricings)

    with open(LOG_FILE, "a") as f:
        f.write(f"## Scan: {scan_time}\n\n")
        f.write(f"- Markets: {' | '.join(f'{k}: {v}' for k, v in counts.items())}\n")
        f.write(f"- Spread arbs: {len(spread_arbs)} | Cross-platform: {len(cross_arbs)} | AI mispricings: {len(mispricings)}\n\n")

        if not total:
            f.write("_No opportunities found._\n\n---\n\n")
            return

        # Spread arbitrage
        for i, a in enumerate(spread_arbs[:10], 1):
            f.write(f"### 💰 Spread #{i} [{a['platform'].upper()}]: {a['question'][:80]}\n\n")
            f.write(f"- YES: ${a['yes_price']:.4f} + NO: ${a['no_price']:.4f} = **${a['total']:.4f}**\n")
            f.write(f"- Spread: ${a['spread']:.4f} → **{a['profit_pct']:.2f}% guaranteed profit**\n")
            f.write(f"- 🔗 {a.get('url', '')}\n\n")

        # Cross-platform arbitrage
        for i, a in enumerate(cross_arbs[:10], 1):
            f.write(f"### 🔄 Cross-Arb #{i}: {a['question']}\n\n")
            f.write(f"- BUY YES on **{a['buy']['platform'].upper()}** @ ${a['buy']['yes_price']:.4f}\n")
            f.write(f"- SELL YES on **{a['sell']['platform'].upper()}** @ ${a['sell']['yes_price']:.4f}\n")
            f.write(f"- **Diff: {a['diff_pct']:.1f}%** (similarity: {a['similarity']:.2f})\n")
            f.write(f"- 🔗 Buy: {a['buy']['url']}\n")
            f.write(f"- 🔗 Sell: {a['sell']['url']}\n\n")

        # AI mispricings
        for i, a in enumerate(mispricings[:10], 1):
            emoji = "📈" if a["direction"] == "underpriced" else "📉"
            action = "BUY YES" if a["direction"] == "underpriced" else "BUY NO"
            f.write(f"### {emoji} Mispricing #{i} [{a['platform'].upper()}]: {a['question'][:80]}\n\n")
            f.write(f"- Market: ${a['market_price']:.2f} → AI: ${a['ai_estimate']:.2f} (**{a['diff_pct']:.1f}% off**)\n")
            f.write(f"- {a['direction']} — {a['reasoning']}\n")
            f.write(f"- Action: **{action}**\n")
            f.write(f"- 🔗 {a.get('url', '')}\n\n")

        f.write("---\n\n")

    # Trim if too large
    if LOG_FILE.stat().st_size > 500_000:
        content = LOG_FILE.read_text()
        header_end = content.find("---\n\n") + 5
        header = content[:header_end]
        rest = content[header_end:]
        trimmed = rest[-400_000:]
        break_point = trimmed.find("\n## Scan:")
        if break_point > 0:
            trimmed = trimmed[break_point + 1:]
        LOG_FILE.write_text(header + trimmed)


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run_scan():
    """Execute one scan cycle across all platforms."""
    scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(f"Starting multi-platform scan at {scan_time}")

    # Fetch from all platforms
    poly_markets = fetch_polymarket(limit=500)
    time.sleep(1)
    limitless_markets = fetch_limitless(limit=500)
    time.sleep(1)
    # Predict.fun: fetch markets then prices (slower due to per-market orderbook calls)
    predict_markets = fetch_predict(limit=100)  # Limited due to orderbook calls

    all_markets = poly_markets + limitless_markets + predict_markets
    counts = {
        "polymarket": len(poly_markets),
        "limitless": len(limitless_markets),
        "predict": len(predict_markets),
        "total": len(all_markets),
    }
    log.info(f"Total markets: {counts}")

    # Strategy 1: Spread arbs
    spread_arbs = find_spread_arbs(all_markets)
    log.info(f"Spread arbs: {len(spread_arbs)}")

    # Strategy 2: Cross-platform arbs
    cross_arbs = find_cross_platform_arbs(all_markets)
    log.info(f"Cross-platform arbs: {len(cross_arbs)}")

    # Strategy 3: AI mispricing
    mispricings = ai_evaluate_batch(all_markets, batch_size=15)
    log.info(f"AI mispricings: {len(mispricings)}")

    # Log
    write_log(spread_arbs, cross_arbs, mispricings, scan_time, counts)
    return spread_arbs, cross_arbs, mispricings


def main():
    log.info("Multi-Platform Prediction Market Scanner started")
    log.info(f"Platforms: Polymarket, Limitless, Predict.fun")
    log.info(f"Interval: {SCAN_INTERVAL}s | Spread: {INTRA_SPREAD_THRESHOLD*100}% | Cross: {CROSS_ARB_THRESHOLD*100}% | AI: {MISPRICING_THRESHOLD*100}%")

    while True:
        try:
            spreads, cross, misps = run_scan()
            total = len(spreads) + len(cross) + len(misps)
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
