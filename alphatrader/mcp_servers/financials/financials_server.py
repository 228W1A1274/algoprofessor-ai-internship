"""
mcp_servers/financials/server.py
---------------------------------
Financials MCP Server — PORT 8003

Simulates SEC filing data (balance sheet, income statement).
Prompt 5+ will replace with EDGAR API / Financial Statements MCP.

Endpoints:
  GET /balance-sheet?symbol=AAPL    → assets, liabilities, equity
  GET /income-statement?symbol=AAPL → revenue, net income, EPS
"""

import random
from datetime import datetime

from fastapi import FastAPI, Query

app = FastAPI(title="MCP: Financials", version="1.0.0")


@app.get("/balance-sheet")
def get_balance_sheet(symbol: str = Query(default="AAPL")):
    """Return a dummy balance sheet snapshot (in USD millions)."""
    total_assets = round(random.uniform(200_000, 500_000), 2)
    total_liab   = round(total_assets * random.uniform(0.4, 0.7), 2)
    equity       = round(total_assets - total_liab, 2)
    return {
        "symbol":          symbol.upper(),
        "period":          "Q4-2024",
        "currency":        "USD_millions",
        "total_assets":    total_assets,
        "total_liabilities": total_liab,
        "shareholders_equity": equity,
        "cash_and_equivalents": round(random.uniform(10_000, 80_000), 2),
        "long_term_debt":  round(random.uniform(20_000, 120_000), 2),
        "debt_to_equity":  round(total_liab / equity, 4),
        "timestamp":       datetime.utcnow().isoformat(),
        "source":          "mcp:financials",
    }


@app.get("/income-statement")
def get_income_statement(symbol: str = Query(default="AAPL")):
    """Return a dummy income statement (in USD millions)."""
    revenue    = round(random.uniform(80_000, 400_000), 2)
    gross      = round(revenue * random.uniform(0.35, 0.55), 2)
    net_income = round(gross * random.uniform(0.2, 0.45), 2)
    eps        = round(net_income / random.uniform(1_000, 8_000), 4)
    return {
        "symbol":          symbol.upper(),
        "period":          "Q4-2024",
        "currency":        "USD_millions",
        "revenue":         revenue,
        "gross_profit":    gross,
        "operating_income": round(gross * random.uniform(0.5, 0.85), 2),
        "net_income":      net_income,
        "eps":             eps,
        "pe_ratio":        round(random.uniform(18, 45), 2),
        "revenue_growth_yoy_pct": round(random.uniform(-5, 25), 2),
        "timestamp":       datetime.utcnow().isoformat(),
        "source":          "mcp:financials",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8003, reload=True)
