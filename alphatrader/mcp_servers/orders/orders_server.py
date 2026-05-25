"""
mcp_servers/orders/server.py
-----------------------------
Orders MCP Server — PORT 8004

Simulates a broker order management system (paper trading).
Prompt 6+ will replace with Alpaca API integration.

Endpoints:
  POST /place-order       → submit a new order
  GET  /order-status      → poll order status by ID
"""

import random
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="MCP: Order Management", version="1.0.0")

# In-memory order store (resets on restart — fine for paper trading)
_orders: dict = {}


class OrderRequest(BaseModel):
    ticker:   str
    action:   str          # BUY | SELL | HOLD
    quantity: int
    price:    float


@app.post("/place-order")
def place_order(order: OrderRequest):
    """
    Accept and simulate execution of a paper-trade order.
    Returns a fill confirmation with slippage applied.
    """
    order_id    = f"ORD-{uuid.uuid4().hex[:12].upper()}"
    slippage    = round(order.price * random.uniform(-0.001, 0.001), 4)
    filled_price = round(order.price + slippage, 4)
    fee_rate    = 0.0005
    fees        = round(order.quantity * filled_price * fee_rate, 4)

    record = {
        "order_id":      order_id,
        "ticker":        order.ticker.upper(),
        "action":        order.action.upper(),
        "quantity":      order.quantity,
        "requested_price": order.price,
        "filled_price":  filled_price,
        "slippage":      slippage,
        "fees_usd":      fees,
        "net_cost":      round(order.quantity * filled_price + fees, 4),
        "status":        "filled",
        "placed_at":     datetime.utcnow().isoformat(),
        "source":        "mcp:orders",
    }
    _orders[order_id] = record
    return record


@app.get("/order-status")
def order_status(order_id: str = Query(..., description="Order ID to look up")):
    """Return the current status of an order by ID."""
    if order_id in _orders:
        return _orders[order_id]
    # Order not found — return a dummy pending response
    return {
        "order_id": order_id,
        "status":   "not_found",
        "message":  f"Order {order_id} not found in this session",
        "source":   "mcp:orders",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8004, reload=True)
