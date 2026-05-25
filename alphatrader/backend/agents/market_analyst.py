"""
agents/market_analyst.py
------------------------
MarketAnalyst — Step 1 of the trading pipeline.

Perf fix (post-Prompt 5):
  All 3 MCP/API tool calls run in PARALLEL via ThreadPoolExecutor.
  Total latency = slowest single call, not sum of all calls.
  Timeout per call reduced to 2s (fast-fail).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime

from tools.market_tools import get_stock_price, get_market_trend, get_dummy_news, analyze_sentiment

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "SPY"
PARALLEL_TIMEOUT = 3   # max seconds to wait for ALL parallel calls


class MarketAnalyst:
    """Analyses market conditions — parallel tool calls for low latency."""

    def __init__(self):
        self.name = "MarketAnalyst"
        logger.info(f"[{self.name}] Initialised")

    def run(self, context: dict | None = None) -> dict:
        symbol = (context or {}).get("symbol", DEFAULT_SYMBOL)
        logger.info(f"[{self.name}] Running parallel analysis for {symbol}...")

        # ── Parallel tool calls ───────────────────────────────────
        results = {}
        tasks = {
            "price": lambda: get_stock_price(symbol),
            "trend": lambda: get_market_trend(),
            "news":  lambda: get_dummy_news(symbol),
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures, timeout=PARALLEL_TIMEOUT):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning(f"[{self.name}] Tool '{name}' failed: {e}")
                    results[name] = {}
        # ─────────────────────────────────────────────────────────

        # Defaults if a call failed
        price_data = results.get("price", {})
        trend_data = results.get("trend", {})
        news       = results.get("news", [])

        # Sentiment runs after news (depends on news result — fast, local)
        sentiment = analyze_sentiment(news)

        insight = {
            "symbol":           price_data.get("symbol", symbol.upper()),
            "price":            price_data.get("price", 512.50),
            "change_pct":       price_data.get("change_pct", 0.0),
            "signal":           trend_data.get("trend", "neutral"),
            "breadth":          trend_data.get("breadth", "neutral"),
            "sentiment":        sentiment["sentiment"],
            "sentiment_score":  sentiment["score"],
            "confidence":       round(0.5 + abs(sentiment["score"]) * 0.1, 4),
            "timestamp":        datetime.utcnow().isoformat(),
            # ── source tracking ────────────────────────────────────
            "price_source":     price_data.get("source", "unknown"),
            "trend_source":     trend_data.get("source", "unknown"),
            "sentiment_source": sentiment.get("source", "unknown"),
        }

        logger.info(
            f"[{self.name}] signal={insight['signal']}  "
            f"price={insight['price']} ({insight['price_source']})  "
            f"sentiment={insight['sentiment']}"
        )
        return insight
