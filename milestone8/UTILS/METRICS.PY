# utils/metrics.py
"""
Deal metrics utilities for BargainBot.
Computes summary statistics across a list of scored deals.
"""

from __future__ import annotations
import json
from typing import Any


def compute_deal_metrics(deals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate metrics for a list of deal results.

    Parameters
    ----------
    deals : list of dicts, each containing at least:
        - 'retailer'  (str)
        - 'price'     (float or str like '$299.99')
        - 'score'     (int 1-10)

    Returns
    -------
    dict with keys: min_price, max_price, avg_price, best_score,
                    best_retailer, price_range, num_deals
    """
    if not deals:
        return {}

    def to_float(val: Any) -> float:
        return float(str(val).replace("$", "").replace(",", ""))

    prices = []
    scores = []
    for d in deals:
        try:
            prices.append(to_float(d.get("price", 0)))
        except ValueError:
            pass
        try:
            scores.append(int(d.get("score", 0)))
        except (ValueError, TypeError):
            pass

    if not prices:
        return {"error": "No valid prices found"}

    best_idx     = scores.index(max(scores)) if scores else 0
    best_deal    = deals[best_idx]

    return {
        "num_deals":      len(deals),
        "min_price":      f"${min(prices):.2f}",
        "max_price":      f"${max(prices):.2f}",
        "avg_price":      f"${sum(prices)/len(prices):.2f}",
        "price_range":    f"${max(prices) - min(prices):.2f}",
        "best_score":     max(scores) if scores else "N/A",
        "best_retailer":  best_deal.get("retailer", "Unknown"),
        "best_price":     f"${to_float(best_deal.get('price', 0)):.2f}",
    }


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """Return a human-readable markdown string of the metrics dict."""
    if not metrics or "error" in metrics:
        return f"⚠️ Metrics unavailable: {metrics.get('error', 'unknown')}"

    lines = [
        "### 📊 Price Range Analysis",
        f"- Deals found   : {metrics['num_deals']}",
        f"- Lowest price  : {metrics['min_price']}",
        f"- Highest price : {metrics['max_price']}",
        f"- Average price : {metrics['avg_price']}",
        f"- Price spread  : {metrics['price_range']}",
        f"- Best deal     : {metrics['best_retailer']} @ {metrics['best_price']} (score {metrics['best_score']}/10)",
    ]
    return "\n".join(lines)