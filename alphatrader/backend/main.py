"""
main.py
-------
FastAPI entry point for AlphaTrader.

Prompt 7: Added Prometheus metrics endpoint /metrics
          Tracks: request count, cycle duration, trade decisions, errors.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

from config import settings
from agent_orchestrator import AgentOrchestrator

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────

# Total API requests per endpoint
REQUEST_COUNT = Counter(
    "alphatrader_request_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Full trading cycle duration in seconds
CYCLE_DURATION = Histogram(
    "alphatrader_cycle_duration_seconds",
    "Trading cycle duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Trade decision counters
TRADE_DECISIONS = Counter(
    "alphatrader_trade_decisions_total",
    "Trade decisions made",
    ["decision"]   # BUY / SELL / HOLD / BLOCKED
)

# Error counter
ERROR_COUNT = Counter(
    "alphatrader_errors_total",
    "Total errors in trading pipeline",
    ["component"]
)

# Currently active trading cycles
ACTIVE_CYCLES = Gauge(
    "alphatrader_active_cycles",
    "Number of trading cycles currently running"
)

# MCP call latency per service
MCP_LATENCY = Histogram(
    "alphatrader_mcp_latency_seconds",
    "MCP service call latency",
    ["service"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# ── Orchestrator ──────────────────────────────────────────────────
orchestrator = AgentOrchestrator()


# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENV}]")
    orchestrator.initialize()
    yield
    logger.info("Shutting down...")
    orchestrator.shutdown()


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Autonomous AI Trading Floor — 4-Agent Pipeline with Monitoring",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: track every request ──────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    return response


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Root health check."""
    return {
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env":     settings.ENV,
        "status":  "healthy",
    }


@app.post("/start", tags=["Orchestrator"])
def start_workflow():
    """
    Trigger one full trading cycle.
    Pipeline: MarketAnalyst → QuantStrategist → RiskGuardian → ExecutionEngine
    """
    ACTIVE_CYCLES.inc()
    start = time.time()

    try:
        result = orchestrator.run_trading_cycle()

        # Record cycle duration
        duration = time.time() - start
        CYCLE_DURATION.observe(duration)

        # Record trade decision
        decision = result.get("final_decision", {}).get("decision", "UNKNOWN")
        TRADE_DECISIONS.labels(decision=decision).inc()

        # Standardised summary field for demo visibility
        fd = result.get("final_decision", {})
        result["summary"] = {
            "action":     fd.get("decision"),
            "price":      fd.get("entry_price"),
            "confidence": fd.get("risk_score"),
            "source":     result.get("data_sources", {}).get("price"),
            "status":     result.get("execution_action", {}).get("status"),
        }

        return result

    except Exception as e:
        ERROR_COUNT.labels(component="pipeline").inc()
        logger.error(f"Pipeline error: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        ACTIVE_CYCLES.dec()


@app.get("/status", tags=["Orchestrator"])
def agent_status():
    """Return current status of all registered agents."""
    return orchestrator.get_status()


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus metrics endpoint.
    Scraped by Prometheus every 15s.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )