"""
mcp_servers/news/server.py
---------------------------
News MCP Server — PORT 8002

Simulates a financial news feed with sentiment scoring.
Prompt 5+ will replace with a live news API (e.g. NewsAPI, Bloomberg).

Endpoints:
  GET /headlines          → latest news headlines
  GET /sentiment          → aggregated sentiment score
"""

import random
from datetime import datetime, timedelta

from fastapi import FastAPI, Query

app = FastAPI(title="MCP: News Feed", version="1.0.0")

# ── Dummy headline pool ───────────────────────────────────────────
HEADLINE_POOL = [
    ("Fed holds rates steady; markets rally on dovish tone", "positive"),
    ("Inflation data beats expectations — CPI down 0.2%", "positive"),
    ("S&P 500 hits record high amid tech earnings surge", "positive"),
    ("Analysts upgrade SPY target to $540 on strong breadth", "positive"),
    ("Goldman upgrades US equities to Overweight", "positive"),
    ("Recession fears resurface as PMI data disappoints", "negative"),
    ("Banking sector under pressure after regional lender warns", "negative"),
    ("Geopolitical tensions weigh on global risk appetite", "negative"),
    ("Oil spikes 4% on OPEC supply cut announcement", "negative"),
    ("Yield curve inversion deepens — analysts warn of slowdown", "negative"),
    ("Earnings season kicks off with mixed results", "neutral"),
    ("Fed speakers offer no new guidance ahead of FOMC", "neutral"),
    ("Markets await jobs report — consensus 180k payrolls", "neutral"),
]


def _make_article(headline: str, sentiment: str, offset_minutes: int) -> dict:
    return {
        "headline":  headline,
        "sentiment": sentiment,
        "source":    random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ", "FT"]),
        "published": (datetime.utcnow() - timedelta(minutes=offset_minutes)).isoformat(),
    }


@app.get("/headlines")
def get_headlines(limit: int = Query(default=5, ge=1, le=10)):
    """Return recent financial news headlines."""
    sample = random.sample(HEADLINE_POOL, min(limit, len(HEADLINE_POOL)))
    articles = [_make_article(h, s, i * 12) for i, (h, s) in enumerate(sample)]
    return {
        "count":     len(articles),
        "articles":  articles,
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "mcp:news",
    }


@app.get("/sentiment")
def get_sentiment(symbol: str = Query(default="market")):
    """Return aggregated sentiment score for a symbol or the overall market."""
    pos = random.randint(2, 8)
    neg = random.randint(0, 4)
    neu = random.randint(1, 4)
    total = pos + neg + neu
    dominant = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    return {
        "symbol":    symbol.upper(),
        "sentiment": dominant,
        "scores": {
            "positive": pos,
            "negative": neg,
            "neutral":  neu,
        },
        "score_pct": round((pos - neg) / total, 4),
        "articles_analysed": total,
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "mcp:news",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)
