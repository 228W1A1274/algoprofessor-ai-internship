"""
mcp_servers/market_data/server.py
----------------------------------
Market Data MCP Server — PORT 8001

Simulates a real-time market data feed.
Prompt 5+ will replace dummy data with live exchange API calls.

Endpoints:
  GET /price?symbol=SPY   → current price snapshot
  GET /trend              → market trend + breadth indicators
  GET /volume             → volume statistics
"""

import random
from datetime import datetime

from fastapi import FastAPI, Query

app = FastAPI(title="MCP: Market Data", version="1.0.0")

# ── Dummy base prices (stable across calls for realism) ───────────
BASE_PRICES = {
    "SPY":  512.50,
    "QQQ":  438.20,
    "AAPL": 189.75,
    "TSLA": 242.10,
    "NVDA": 875.30,
    "MSFT": 415.60,
}


@app.get("/price")
def get_price(symbol: str = Query(default="SPY", description="Ticker symbol")):
    """Return current price snapshot for a given ticker."""
    base = BASE_PRICES.get(symbol.upper(), round(random.uniform(50, 600), 2))
    change = round(random.uniform(-8, 8), 2)
    return {
        "symbol":     symbol.upper(),
        "price":      round(base + random.uniform(-2, 2), 4),
        "open":       round(base - random.uniform(0, 3), 4),
        "high":       round(base + random.uniform(1, 5), 4),
        "low":        round(base - random.uniform(1, 5), 4),
        "change":     change,
        "change_pct": round(change / base * 100, 4),
        "timestamp":  datetime.utcnow().isoformat(),
        "source":     "mcp:market_data",
    }


@app.get("/trend")
def get_trend():
    """Return overall market trend and breadth indicators."""
    trend = random.choice(["bullish", "bearish", "neutral"])
    return {
        "trend":                trend,
        "advance_decline_ratio": round(random.uniform(0.6, 2.4), 2),
        "new_highs":            random.randint(40, 300),
        "new_lows":             random.randint(5, 80),
        "vix":                  round(random.uniform(12, 30), 2),
        "breadth":              "positive" if trend == "bullish" else ("negative" if trend == "bearish" else "neutral"),
        "timestamp":            datetime.utcnow().isoformat(),
        "source":               "mcp:market_data",
    }


@app.get("/volume")
def get_volume(symbol: str = Query(default="SPY")):
    """Return volume statistics for a symbol."""
    return {
        "symbol":          symbol.upper(),
        "volume":          random.randint(50_000_000, 120_000_000),
        "avg_volume_10d":  80_000_000,
        "relative_volume": round(random.uniform(0.8, 1.8), 2),
        "dollar_volume":   round(random.uniform(1e10, 6e10), 2),
        "timestamp":       datetime.utcnow().isoformat(),
        "source":          "mcp:market_data",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
