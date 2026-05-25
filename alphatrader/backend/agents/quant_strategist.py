"""
agents/quant_strategist.py
--------------------------
QuantStrategist — Step 2 of the trading pipeline.

Prompt 2: derived action from a simple signal string.
Prompt 3: uses calculate_momentum(), generate_trade_signal(), risk_reward_ratio()
          from tools/quant_tools.py for richer, data-driven strategy output.
"""

import logging
import random
from datetime import datetime

from tools.quant_tools import calculate_momentum, generate_trade_signal, risk_reward_ratio

logger = logging.getLogger(__name__)


class QuantStrategist:
    """Generates a trading strategy from market insight using quant tools."""

    def __init__(self):
        self.name = "QuantStrategist"
        logger.info(f"[{self.name}] Initialised")

    def run(self, market_insight: dict) -> dict:
        """
        Build a trading strategy.

        Args:
            market_insight: Output dict from MarketAnalyst.run().

        Returns:
            strategy dict consumed by RiskGuardian.
        """
        logger.info(f"[{self.name}] Building strategy from signal: {market_insight.get('signal')}")

        # ── Tool calls ────────────────────────────────────────────
        # Simulate a short price series around the current price
        price = market_insight.get("price", 512.50)
        dummy_prices = [round(price * (1 + random.uniform(-0.005, 0.005)), 4) for _ in range(20)]
        dummy_prices.append(price)   # ensure last element is the live price

        momentum    = calculate_momentum(dummy_prices)
        trade_sig   = generate_trade_signal(market_insight)
        action      = trade_sig["action"]

        entry       = price
        stop_loss   = round(entry * 0.985, 4)    # 1.5% below entry
        take_profit = round(entry * 1.025, 4)    # 2.5% above entry
        rr          = risk_reward_ratio(entry, stop_loss, take_profit)
        # ─────────────────────────────────────────────────────────

        strategy = {
            "action":       action,
            "ticker":       market_insight.get("symbol", "SPY"),
            "quantity":     10,
            "entry_price":  entry,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "momentum":     momentum["momentum"],
            "risk_reward":  rr["ratio"],
            "confidence":   trade_sig["confidence"],
            "rationale":    f"{action} signal — momentum={momentum['momentum']:+.2f}%, R:R={rr['ratio']}",
            "timestamp":    datetime.utcnow().isoformat(),
        }

        logger.info(f"[{self.name}] {action} {strategy['ticker']} @ {entry}  R:R={rr['ratio']}")
        return strategy
