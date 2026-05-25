"""
agents/execution_engine.py
--------------------------
ExecutionEngine — Step 4 (final) of the trading pipeline.

Prompt 2: returned a hardcoded confirmation string.
Prompt 3: calls place_order(), log_trade() from tools/execution_tools.py
          for realistic paper-trading order simulation.
"""

import logging
from datetime import datetime

from tools.execution_tools import place_order, log_trade, audit_log

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Places paper-trade orders and logs every execution decision."""

    def __init__(self):
        self.name = "ExecutionEngine"
        logger.info(f"[{self.name}] Initialised")

    def run(self, strategy: dict, risk_result: dict) -> dict:
        """
        Execute the trade if risk approved; audit either way.

        Args:
            strategy:    Output dict from QuantStrategist.run().
            risk_result: Output dict from RiskGuardian.run().

        Returns:
            execution_result dict — the final pipeline output.
        """
        if not risk_result.get("approved", False):
            logger.warning(f"[{self.name}] Trade BLOCKED — {risk_result.get('message')}")
            blocked = {
                "status":    "blocked",
                "reason":    risk_result.get("message", "Risk check failed"),
                "order_id":  None,
                "timestamp": datetime.utcnow().isoformat(),
            }
            audit_log("trade_blocked", blocked)
            return blocked

        ticker   = strategy["ticker"]
        action   = strategy["action"]
        quantity = strategy["quantity"]
        price    = strategy["entry_price"]

        logger.info(f"[{self.name}] Placing order: {action} {quantity}x {ticker} @ {price}")

        # ── Tool calls ────────────────────────────────────────────
        order     = place_order(ticker, action, quantity, price)
        log_entry = log_trade(order)
        # ─────────────────────────────────────────────────────────

        execution_result = {
            "status":       order.get("status", "executed"),
            "order_id":     order["order_id"],
            "ticker":       order["ticker"],
            "action":       order["action"],
            "quantity":     order["quantity"],
            "filled_price": order["filled_price"],
            "fees_usd":     order["fees_usd"],
            "net_cost":     order["net_cost"],
            "log_id":       log_entry["log_id"],
            "source":       order.get("source", "unknown"),
            "message":      "Trade executed successfully (paper trading)",
            "timestamp":    datetime.utcnow().isoformat(),
        }

        logger.info(
            f"[{self.name}] Order {order['order_id']} filled @ {order['filled_price']}  "
            f"fees=${order['fees_usd']}"
        )
        return execution_result