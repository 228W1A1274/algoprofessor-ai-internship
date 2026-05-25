"""
mcp_servers/risk/server.py
---------------------------
Risk Engine MCP Server — PORT 8005

Simulates an independent risk management service.
Prompt 6+ will enforce live portfolio-level limits.

Endpoints:
  POST /check-risk        → validate a proposed trade
  GET  /limits            → return current risk limit configuration
"""

import random
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MCP: Risk Engine", version="1.0.0")

# ── Risk configuration (future: load from config.py / DB) ─────────
RISK_CONFIG = {
    "max_position_size":  50,
    "max_drawdown_pct":   10.0,
    "max_trade_value":    25_000.0,
    "max_risk_exposure":  0.20,
    "min_cash_reserve":   0.10,
    "max_risk_score":     0.75,
    "blacklist":          ["SCAM", "FRAUD", "TEST123"],
}


class TradePayload(BaseModel):
    ticker:    str
    action:    str
    quantity:  int
    price:     float
    stop_loss: float = 0.0


@app.post("/check-risk")
def check_risk(trade: TradePayload):
    """
    Run all risk checks against a proposed trade.
    Returns approved=True/False with per-check detail.
    """
    checks = {}

    # Position size
    checks["position_limit"] = trade.quantity <= RISK_CONFIG["max_position_size"]

    # Trade value
    trade_value = trade.quantity * trade.price
    checks["trade_value"] = trade_value <= RISK_CONFIG["max_trade_value"]

    # Compliance / blacklist
    checks["compliance"] = trade.ticker.upper() not in RISK_CONFIG["blacklist"]

    # Stop-loss sanity (must be set and within 3% of entry)
    if trade.stop_loss > 0 and trade.price > 0:
        loss_pct = abs(trade.price - trade.stop_loss) / trade.price
        checks["stop_loss"] = loss_pct <= 0.03
    else:
        checks["stop_loss"] = False

    # Drawdown — simulated current portfolio drawdown
    current_dd = round(random.uniform(0, 8), 2)
    checks["drawdown"] = current_dd <= RISK_CONFIG["max_drawdown_pct"]

    failed     = sum(1 for v in checks.values() if not v)
    risk_score = round(failed / len(checks), 4)
    approved   = risk_score < RISK_CONFIG["max_risk_score"]

    return {
        "approved":    approved,
        "status":      "approved" if approved else "rejected",
        "risk_score":  risk_score,
        "checks":      checks,
        "trade_value": round(trade_value, 2),
        "message":     "All checks passed" if approved else "One or more risk checks failed",
        "timestamp":   datetime.utcnow().isoformat(),
        "source":      "mcp:risk",
    }


@app.get("/limits")
def get_limits():
    """Return the current risk limit configuration."""
    return {
        "limits":    RISK_CONFIG,
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "mcp:risk",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8005, reload=True)
