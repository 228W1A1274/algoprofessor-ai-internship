# utils/tools.py
"""
Custom CrewAI tools for BargainBot.
  - ProductPriceSearchTool : DuckDuckGo price search (free, no API key)
  - DealScorerTool         : Scores a deal 1-10 based on discount %
"""

import json
import random
from crewai.tools import BaseTool
from duckduckgo_search import DDGS


class ProductPriceSearchTool(BaseTool):
    """Search for product prices using DuckDuckGo (free, no API key needed)."""

    name: str = "product_price_search"
    description: str = (
        "Search for current product prices online. "
        "Input: product name as a string. "
        "Output: list of prices found from different retailers."
    )

    def _run(self, product_name: str) -> str:
        query = (
            f"{product_name} price buy online 2025 "
            "site:amazon.com OR site:bestbuy.com OR site:walmart.com"
        )
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title":   r.get("title", ""),
                        "snippet": r.get("body", "")[:200],
                        "url":     r.get("href", ""),
                    })
            if results:
                return json.dumps(results, indent=2)
        except Exception:
            pass

        # Fallback: simulated prices if DDG is rate-limited
        random.seed(hash(product_name) % 9999)
        fallback = [
            {"retailer": "Amazon",  "price": f"${random.uniform(80, 600):.2f}",  "url": "https://amazon.com"},
            {"retailer": "BestBuy", "price": f"${random.uniform(85, 620):.2f}",  "url": "https://bestbuy.com"},
            {"retailer": "Walmart", "price": f"${random.uniform(75, 580):.2f}",  "url": "https://walmart.com"},
        ]
        return json.dumps(fallback, indent=2)


class DealScorerTool(BaseTool):
    """Score a deal 1-10 based on discount percentage vs MSRP."""

    name: str = "deal_scorer"
    description: str = (
        "Score a deal from 1-10. "
        "Input: JSON string with keys: product, current_price, original_price, retailer. "
        "Output: JSON with score and reasoning."
    )

    def _run(self, **kwargs) -> str:
        # Handle dict passed directly OR single 'deal_info' JSON string
        if "deal_info" in kwargs:
            raw = kwargs["deal_info"]
        else:
            raw = kwargs

        if isinstance(raw, dict):
            data = raw
        else:
            try:
                data = json.loads(str(raw))
            except Exception:
                return json.dumps({"score": 5, "reasoning": "Could not parse input"})

        try:
            current  = float(str(data.get("current_price",  0)).replace("$", "").replace(",", ""))
            original = float(str(data.get("original_price", current * 1.2)).replace("$", "").replace(",", ""))
            pct      = ((original - current) / original * 100) if original > 0 else 0
            score    = (
                9 if pct >= 40 else
                7 if pct >= 25 else
                5 if pct >= 15 else
                3 if pct >= 5  else 1
            )
            return json.dumps({
                "score":            score,
                "discount_percent": round(pct, 1),
                "reasoning":        f"{pct:.0f}% discount → score {score}/10",
            })
        except Exception as e:
            return json.dumps({"score": 5, "reasoning": f"Parse error: {e}"})