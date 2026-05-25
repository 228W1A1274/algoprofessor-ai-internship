"""
mcp_servers/notifications/server.py
-------------------------------------
Notification MCP Server — PORT 8006

Simulates a notification and audit-log service.
Prompt 6+ will integrate with Slack / email / webhook delivery.

Endpoints:
  POST /send              → send a notification event
  GET  /history           → retrieve recent notification history
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MCP: Notifications", version="1.0.0")

# In-memory log (resets on restart)
_history: list = []
MAX_HISTORY = 100


class NotificationRequest(BaseModel):
    event:    str                    # e.g. "trade_executed", "risk_rejected"
    level:    str = "info"           # info | warning | error
    message:  str
    payload:  Optional[dict] = None  # arbitrary extra data


@app.post("/send")
def send_notification(notif: NotificationRequest):
    """Accept and store a notification event."""
    record = {
        "notification_id": f"NTF-{uuid.uuid4().hex[:8].upper()}",
        "event":           notif.event,
        "level":           notif.level,
        "message":         notif.message,
        "payload":         notif.payload or {},
        "delivered":       True,         # simulated delivery
        "channel":         "internal",   # future: slack | email | webhook
        "timestamp":       datetime.utcnow().isoformat(),
        "source":          "mcp:notifications",
    }
    _history.append(record)
    # Keep history bounded
    if len(_history) > MAX_HISTORY:
        _history.pop(0)
    return record


@app.get("/history")
def get_history(limit: int = 10):
    """Return the most recent notification events."""
    recent = _history[-limit:][::-1]   # newest first
    return {
        "count":     len(recent),
        "events":    recent,
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "mcp:notifications",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8006, reload=True)
