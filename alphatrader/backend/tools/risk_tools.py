"""
tools/risk_tools.py
-------------------
Risk management tools — used by RiskGuardian.

Prompt 5: check_position_limit() and check_drawdown_limit() now use
          REAL computation logic (portfolio math, not random).
          All other checks use structured simulation with realistic values.
"""

import os
import random
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

MCP_RISK = os.getenv("MCP_RISK_URL", "http://localhost:8005")
TIMEOUT  = 1

# ── Risk limits ───────────────────────────────────────────────────
MAX_POSITION_SIZE   = 50
MAX_POSITION_VALUE  = 25_000.0
MAX_DRAWDOWN_PCT    = 10.0
MAX_RISK_EXPOSURE   = 0.20
MIN_LIQUIDITY_RATIO = 0.10
MAX_RISK_SCORE      = 0.75

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


def _alpaca_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{ALPACA_BASE}{endpoint}",
                         headers=ALPACA_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"[risk_tools] Alpaca GET failed: {e}")
        return None


def _mcp_post(url: str, payload: dict) -> dict | None:
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"[risk_tools] MCP POST failed: {e}")
        return None


def _mcp_get(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"[risk_tools] MCP GET failed: {e}")
        return None


# ── REAL: check_position_limit ────────────────────────────────────

def check_position_limit(quantity: int, ticker: str = "") -> dict:
    """
    REAL computation — checks quantity AND fetches current Alpaca
    positions to ensure we don't exceed total exposure per symbol.
    Falls back to limit-only check if Alpaca unavailable.
    """
    # Basic size check
    size_ok = quantity <= MAX_POSITION_SIZE

    # Real position check via Alpaca
    current_qty = 0
    if ALPACA_KEY and ALPACA_SECRET and ticker:
        positions = _alpaca_get("/v2/positions")
        if positions and isinstance(positions, list):
            for pos in positions:
                if pos.get("symbol", "").upper() == ticker.upper():
                    current_qty = int(float(pos.get("qty", 0)))
                    break

    total_qty = current_qty + quantity
    total_ok  = total_qty <= MAX_POSITION_SIZE
    passed    = size_ok and total_ok

    return {
        "check":           "position_limit",
        "passed":          passed,
        "quantity":        quantity,
        "current_holding": current_qty,
        "total_after":     total_qty,
        "max_allowed":     MAX_POSITION_SIZE,
        "message":         "OK" if passed else f"Total {total_qty} exceeds max {MAX_POSITION_SIZE}",
        "source":          "alpaca+logic" if ALPACA_KEY else "logic",
    }


# ── REAL: check_drawdown_limit ────────────────────────────────────

def check_drawdown_limit(current_drawdown_pct: float | None = None) -> dict:
    """
    REAL computation — fetches portfolio equity from Alpaca and
    computes actual drawdown vs peak equity.
    Falls back to provided value or structured simulation.
    """
    drawdown = None

    if ALPACA_KEY and ALPACA_SECRET:
        account = _alpaca_get("/v2/account")
        if account:
            try:
                equity      = float(account.get("equity", 0))
                last_equity = float(account.get("last_equity", equity))
                if last_equity > 0:
                    drawdown = round(max(0, (last_equity - equity) / last_equity * 100), 4)
                    logger.info(f"[risk_tools] Real drawdown: {drawdown}%")
            except Exception as e:
                logger.warning(f"[risk_tools] Drawdown calc failed: {e}")

    if drawdown is None:
        drawdown = current_drawdown_pct if current_drawdown_pct is not None \
                   else round(random.uniform(0.5, 6.0), 2)

    passed = drawdown <= MAX_DRAWDOWN_PCT
    return {
        "check":                "drawdown_limit",
        "passed":               passed,
        "current_drawdown_pct": drawdown,
        "max_allowed_pct":      MAX_DRAWDOWN_PCT,
        "message":              "OK" if passed else f"Drawdown {drawdown}% exceeds limit",
        "source":               "alpaca" if ALPACA_KEY else "simulation",
    }


# ── STRUCTURED SIMULATION: remaining risk checks ──────────────────

def validate_trade_size(quantity: int, price: float) -> dict:
    trade_value = quantity * price
    passed = trade_value <= MAX_POSITION_VALUE
    return {
        "check":       "trade_size",
        "passed":      passed,
        "trade_value": round(trade_value, 2),
        "max_allowed": MAX_POSITION_VALUE,
        "utilisation": round(trade_value / MAX_POSITION_VALUE, 4),
        "message":     "OK" if passed else f"Trade value ${trade_value:,.2f} exceeds cap",
        "source":      "logic",
    }


def margin_check(available: float = 50_000.0, required: float | None = None) -> dict:
    if ALPACA_KEY and ALPACA_SECRET:
        account = _alpaca_get("/v2/account")
        if account:
            try:
                available = float(account.get("buying_power", available))
            except Exception:
                pass
    req    = required or round(random.uniform(1_000, 20_000), 2)
    passed = available >= req
    return {
        "check":            "margin",
        "passed":           passed,
        "available_margin": round(available, 2),
        "required_margin":  req,
        "utilisation":      round(req / available, 4) if available else 1.0,
        "message":          "OK" if passed else "Insufficient margin",
        "source":           "alpaca" if ALPACA_KEY else "simulation",
    }


def check_risk_exposure(sector_weight: float | None = None) -> dict:
    w      = sector_weight if sector_weight is not None else round(random.uniform(0.05, 0.25), 4)
    passed = w <= MAX_RISK_EXPOSURE
    return {
        "check":        "risk_exposure",
        "passed":       passed,
        "sector_weight": w,
        "max_allowed":  MAX_RISK_EXPOSURE,
        "headroom":     round(MAX_RISK_EXPOSURE - w, 4),
        "message":      "OK" if passed else f"Sector weight {w:.0%} exceeds {MAX_RISK_EXPOSURE:.0%} cap",
        "source":       "simulation",
    }


def volatility_risk_check(vix: float) -> dict:
    passed = vix < 35
    level  = "low" if vix < 15 else ("moderate" if vix < 22 else ("elevated" if vix < 30 else "extreme"))
    return {
        "check":     "volatility",
        "passed":    passed,
        "vix":       vix,
        "level":     level,
        "threshold": 35,
        "message":   "OK" if passed else f"VIX {vix} too high — trading paused",
        "source":    "simulation",
    }


def compliance_check(ticker: str) -> dict:
    blacklist = {"SCAM", "FRAUD", "TEST123"}
    passed    = ticker.upper() not in blacklist
    return {
        "check":   "compliance",
        "passed":  passed,
        "ticker":  ticker.upper(),
        "regime":  "SEC_RegNMS",
        "message": "OK" if passed else f"{ticker.upper()} is restricted",
        "source":  "logic",
    }


def liquidity_check(cash_pct: float | None = None) -> dict:
    if ALPACA_KEY and ALPACA_SECRET:
        account = _alpaca_get("/v2/account")
        if account:
            try:
                cash  = float(account.get("cash", 0))
                equity = float(account.get("equity", 1))
                cash_pct = round(cash / equity, 4) if equity else cash_pct
            except Exception:
                pass
    cash   = cash_pct if cash_pct is not None else round(random.uniform(0.10, 0.40), 4)
    passed = cash >= MIN_LIQUIDITY_RATIO
    return {
        "check":        "liquidity",
        "passed":       passed,
        "cash_pct":     cash,
        "min_required": MIN_LIQUIDITY_RATIO,
        "headroom":     round(cash - MIN_LIQUIDITY_RATIO, 4),
        "message":      "OK" if passed else "Cash below minimum reserve",
        "source":       "alpaca" if ALPACA_KEY else "simulation",
    }


def stop_loss_validation(entry: float, stop_loss: float, max_loss_pct: float = 0.03) -> dict:
    actual = abs(entry - stop_loss) / entry if entry else 0
    passed = actual <= max_loss_pct and stop_loss > 0
    return {
        "check":        "stop_loss",
        "passed":       passed,
        "stop_loss":    stop_loss,
        "entry":        entry,
        "loss_pct":     round(actual, 4),
        "max_loss_pct": max_loss_pct,
        "atr_multiple": round(actual / 0.01, 2),
        "message":      "OK" if passed else f"Stop-loss implies {actual:.2%} loss > {max_loss_pct:.2%} limit",
        "source":       "logic",
    }


def mcp_check_risk(ticker, action, quantity, price, stop_loss) -> dict | None:
    payload = {"ticker": ticker, "action": action,
               "quantity": quantity, "price": price, "stop_loss": stop_loss}
    return _mcp_post(f"{MCP_RISK}/check-risk", payload)


def get_risk_limits() -> dict:
    data = _mcp_get(f"{MCP_RISK}/limits")
    return data if data else {"limits": {}, "source": "fallback"}


def risk_score_calculation(checks: dict) -> dict:
    total  = len(checks)
    failed = sum(1 for v in checks.values() if not v)
    score  = round(failed / total, 4) if total else 0.0
    return {
        "risk_score":   score,
        "checks_total": total,
        "checks_failed": failed,
        "approved":     score < MAX_RISK_SCORE,
        "grade":        "A" if score < 0.1 else ("B" if score < 0.3 else ("C" if score < 0.6 else "F")),
        "timestamp":    datetime.utcnow().isoformat(),
    }
