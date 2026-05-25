"""
tools/execution_tools.py
------------------------
Order execution tools — used by ExecutionEngine.

Prompt 5: place_order(), get_order_status(), cancel_order()
          now use Alpaca Paper Trading API (real calls, free).
          Falls back to MCP Orders server, then structured simulation.
"""

import os
import random
import uuid
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# ── Alpaca config ─────────────────────────────────────────────────
ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type":        "application/json",
}

MCP_ORDERS        = os.getenv("MCP_ORDERS_URL",        "http://localhost:8004")
MCP_NOTIFICATIONS = os.getenv("MCP_NOTIFICATIONS_URL", "http://localhost:8006")
TIMEOUT           = 1


def _alpaca_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{ALPACA_BASE}{endpoint}", json=payload,
                          headers=ALPACA_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[execution_tools] Alpaca POST failed ({endpoint}): {e}")
        return None


def _alpaca_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{ALPACA_BASE}{endpoint}",
                         headers=ALPACA_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[execution_tools] Alpaca GET failed ({endpoint}): {e}")
        return None


def _alpaca_delete(endpoint: str) -> bool:
    try:
        r = requests.delete(f"{ALPACA_BASE}{endpoint}",
                            headers=ALPACA_HEADERS, timeout=TIMEOUT)
        return r.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"[execution_tools] Alpaca DELETE failed ({endpoint}): {e}")
        return False


def _mcp_post(url: str, payload: dict) -> dict | None:
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"MCP POST failed ({url}): {e}")
        return None


def _mcp_get(url: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"MCP GET failed ({url}): {e}")
        return None


# ── Helpers ───────────────────────────────────────────────────────

def generate_order_id() -> str:
    return f"ORD-{uuid.uuid4().hex[:12].upper()}"


def trade_confirmation(order_id, ticker, action, quantity, price) -> dict:
    return {
        "order_id": order_id, "ticker": ticker, "action": action,
        "quantity": quantity, "filled_price": price,
        "total_value": round(quantity * price, 2),
        "status": "filled", "confirmed_at": datetime.utcnow().isoformat(),
    }


def calculate_fees(quantity: int, price: float, fee_rate: float = 0.0005) -> dict:
    fees = round(quantity * price * fee_rate, 4)
    return {"quantity": quantity, "price": price, "fee_rate": fee_rate, "fees_usd": fees}


def slippage_estimation(price: float, volatility_factor: float = 0.001) -> dict:
    slip = round(price * volatility_factor * random.uniform(-1, 1), 4)
    return {"expected_price": price, "slippage_amount": slip,
            "estimated_fill": round(price + slip, 4)}


def execution_latency() -> dict:
    ms = round(random.uniform(2, 45), 2)
    return {"latency_ms": ms,
            "quality": "excellent" if ms < 10 else ("good" if ms < 25 else "acceptable")}


# ── REAL: place_order ─────────────────────────────────────────────

def place_order(ticker: str, action: str, quantity: int, price: float) -> dict:
    """
    REAL — Submit a paper-trade order to Alpaca.
    Falls back to MCP Orders server, then structured simulation.
    """
    # Skip HOLD orders — no execution needed
    if action.upper() == "HOLD":
        return {
            "order_id": None, "ticker": ticker.upper(), "action": "HOLD",
            "quantity": 0, "filled_price": price, "fees_usd": 0.0,
            "net_cost": 0.0, "status": "skipped",
            "message": "HOLD signal — no order placed",
            "placed_at": datetime.utcnow().isoformat(), "source": "logic",
        }

    if ALPACA_KEY and ALPACA_SECRET:
        payload = {
            "symbol":        ticker.upper(),
            "qty":           str(quantity),
            "side":          action.lower(),   # "buy" or "sell"
            "type":          "market",
            "time_in_force": "day",
        }
        data = _alpaca_post("/v2/orders", payload)
        if data:
            filled_price = float(data.get("filled_avg_price") or price)
            fees = calculate_fees(quantity, filled_price)
            logger.info(f"[execution_tools] Alpaca order {data.get('id')} placed")
            return {
                "order_id":       data.get("id", generate_order_id()),
                "ticker":         data.get("symbol", ticker.upper()),
                "action":         data.get("side", action).upper(),
                "quantity":       int(data.get("qty", quantity)),
                "requested_price": price,
                "filled_price":   filled_price,
                "fees_usd":       fees["fees_usd"],
                "net_cost":       round(quantity * filled_price + fees["fees_usd"], 4),
                "status":         data.get("status", "accepted"),
                "placed_at":      data.get("created_at", datetime.utcnow().isoformat()),
                "source":         "alpaca",
            }

    # MCP fallback
    mcp = _mcp_post(f"{MCP_ORDERS}/place-order",
                    {"ticker": ticker, "action": action, "quantity": quantity, "price": price})
    if mcp:
        mcp["source"] = "mcp:orders"
        return mcp

    # Structured simulation fallback
    slip = slippage_estimation(price)
    filled = slip["estimated_fill"]
    fees = calculate_fees(quantity, filled)
    return {
        "order_id": generate_order_id(), "ticker": ticker.upper(),
        "action": action.upper(), "quantity": quantity,
        "requested_price": price, "filled_price": filled,
        "fees_usd": fees["fees_usd"],
        "net_cost": round(quantity * filled + fees["fees_usd"], 4),
        "status": "filled", "placed_at": datetime.utcnow().isoformat(),
        "source": "simulation",
    }


# ── REAL: get_order_status ────────────────────────────────────────

def get_order_status(order_id: str) -> dict:
    """REAL — Poll Alpaca for order status."""
    if ALPACA_KEY and ALPACA_SECRET and order_id:
        data = _alpaca_get(f"/v2/orders/{order_id}")
        if data:
            return {
                "order_id":    data.get("id"),
                "status":      data.get("status"),
                "filled_qty":  data.get("filled_qty", 0),
                "filled_price": data.get("filled_avg_price"),
                "source":      "alpaca",
            }

    mcp = _mcp_get(f"{MCP_ORDERS}/order-status", {"order_id": order_id})
    if mcp:
        return mcp

    return {
        "order_id": order_id,
        "status": random.choice(["filled", "partially_filled", "pending"]),
        "filled_qty": random.randint(1, 10),
        "remaining_qty": random.randint(0, 5),
        "confidence": round(random.uniform(0.7, 1.0), 4),
        "source": "simulation",
    }


# ── REAL: cancel_order ────────────────────────────────────────────

def cancel_order(order_id: str) -> dict:
    """REAL — Cancel an open order on Alpaca."""
    if ALPACA_KEY and ALPACA_SECRET and order_id:
        success = _alpaca_delete(f"/v2/orders/{order_id}")
        if success:
            logger.info(f"[execution_tools] Alpaca order {order_id} cancelled")
            return {
                "order_id": order_id, "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "source": "alpaca",
            }

    return {
        "order_id": order_id, "status": "cancelled",
        "cancelled_at": datetime.utcnow().isoformat(),
        "message": f"Order {order_id} cancelled (simulated)",
        "source": "simulation",
    }


# ── Logging & Audit ───────────────────────────────────────────────

def log_trade(order: dict) -> dict:
    payload = {
        "event":   "trade_executed",
        "level":   "info",
        "message": f"{order.get('action')} {order.get('quantity')}x "
                   f"{order.get('ticker')} @ {order.get('filled_price')}",
        "payload": order,
    }
    data = _mcp_post(f"{MCP_NOTIFICATIONS}/send", payload)
    if data:
        return {"log_id": data.get("notification_id"), "status": "logged",
                "source": "mcp:notifications"}
    return {
        "log_id": f"LOG-{uuid.uuid4().hex[:8].upper()}",
        "order_id": order.get("order_id"), "status": "logged",
        "source": "simulation",
    }


def audit_log(event: str, payload: dict) -> dict:
    body = {
        "event":   event,
        "level":   "warning" if "block" in event or "reject" in event else "info",
        "message": f"Audit: {event}",
        "payload": {k: str(v)[:80] for k, v in payload.items()},
    }
    data = _mcp_post(f"{MCP_NOTIFICATIONS}/send", body)
    if data:
        return {"audit_id": data.get("notification_id"), "event": event,
                "source": "mcp:notifications"}
    return {
        "audit_id": f"AUD-{uuid.uuid4().hex[:8].upper()}",
        "event": event, "logged_at": datetime.utcnow().isoformat(),
        "source": "simulation",
    }
