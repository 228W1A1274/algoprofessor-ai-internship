"""
agents/risk_guardian.py
-----------------------
RiskGuardian — Step 3 of the trading pipeline.

Prompt 3: ran all checks locally via risk_tools.
Prompt 4: tries MCP Risk server first (port 8005);
          falls back to local tool checks if MCP is offline.
"""

import logging
from datetime import datetime

from tools.risk_tools import (
    mcp_check_risk,
    check_position_limit, compliance_check, stop_loss_validation,
    validate_trade_size, risk_score_calculation,
)

logger = logging.getLogger(__name__)


class RiskGuardian:
    """Validates a trading strategy — MCP-first, local fallback."""

    def __init__(self):
        self.name = "RiskGuardian"
        logger.info(f"[{self.name}] Initialised")

    def run(self, strategy: dict) -> dict:
        ticker   = strategy.get("ticker", "SPY")
        quantity = strategy.get("quantity", 1)
        entry    = strategy.get("entry_price", 0.0)
        sl       = strategy.get("stop_loss", 0.0)
        action   = strategy.get("action", "BUY")

        logger.info(f"[{self.name}] Validating {action} {ticker} qty={quantity}")

        # ── Try MCP Risk server first ──────────────────────────────
        mcp_result = mcp_check_risk(ticker, action, quantity, entry, sl)

        if mcp_result:
            logger.info(f"[{self.name}] MCP result: {mcp_result.get('status')} score={mcp_result.get('risk_score')}")
            return {
                "approved":     mcp_result["approved"],
                "status":       mcp_result["status"],
                "checks":       mcp_result["checks"],
                "risk_score":   mcp_result["risk_score"],
                "check_detail": {k: ("OK" if v else "FAILED") for k, v in mcp_result["checks"].items()},
                "message":      mcp_result["message"],
                "source":       "mcp:risk",
                "timestamp":    datetime.utcnow().isoformat(),
            }

        # ── Fallback: local tool checks ────────────────────────────
        logger.warning(f"[{self.name}] MCP offline — running local checks")
        pos_check  = check_position_limit(quantity, ticker)
        comp_check = compliance_check(ticker)
        sl_check   = stop_loss_validation(entry, sl)
        size_check = validate_trade_size(quantity, entry)

        all_checks = {
            "position_limit": pos_check["passed"],
            "compliance":     comp_check["passed"],
            "stop_loss":      sl_check["passed"],
            "trade_size":     size_check["passed"],
        }
        score_data = risk_score_calculation(all_checks)
        approved   = score_data["approved"]

        result = {
            "approved":     approved,
            "status":       "approved" if approved else "rejected",
            "checks":       all_checks,
            "risk_score":   score_data["risk_score"],
            "check_detail": {
                "position_limit": pos_check["message"],
                "compliance":     comp_check["message"],
                "stop_loss":      sl_check["message"],
                "trade_size":     size_check["message"],
            },
            "message":  "All risk checks passed" if approved else "One or more risk checks failed",
            "source":   "local_fallback",
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(f"[{self.name}] Local result: {result['status']} score={score_data['risk_score']}")
        return result
