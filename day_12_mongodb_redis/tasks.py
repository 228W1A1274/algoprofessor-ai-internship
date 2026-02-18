"""
Day 12 Deliverable: tasks.py
Topics: Celery + Redis broker, task queuing, rate limiting, distributed locks

HOW TO RUN:
  Terminal 1 (start worker):  celery -A tasks worker --loglevel=info
  Terminal 2 (send tasks):    python3 run_tasks.py
"""

import time
import redis
from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ─────────────────────────────────────────────
# APP SETUP  — Redis as both broker AND backend
# ─────────────────────────────────────────────
app = Celery(
    "day12_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

# Optional config
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    result_expires=3600,          # results expire in 1 hour
)


# ─────────────────────────────────────────────
# TASK 1: Basic background task
# ─────────────────────────────────────────────
@app.task(name="tasks.reverse_string")
def reverse_string(text: str) -> str:
    """Simple task: reverse a string in background."""
    logger.info(f"Reversing: {text}")
    time.sleep(1)                 # simulate work
    return text[::-1]


# ─────────────────────────────────────────────
# TASK 2: Heavy computation (simulated)
# ─────────────────────────────────────────────
@app.task(name="tasks.process_data", bind=True, max_retries=3)
def process_data(self, data: dict) -> dict:
    """Process data with automatic retry on failure."""
    try:
        logger.info(f"Processing: {data}")
        time.sleep(2)             # simulate heavy computation
        result = {
            "input":       data,
            "word_count":  len(str(data).split()),
            "processed_at": time.time(),
            "status":      "success"
        }
        return result
    except Exception as exc:
        logger.error(f"Task failed: {exc}. Retrying...")
        raise self.retry(exc=exc, countdown=5)   # retry after 5 sec


# ─────────────────────────────────────────────
# TASK 3: Rate limiting — max 5 tasks/minute
# ─────────────────────────────────────────────
@app.task(name="tasks.send_notification", rate_limit="5/m")
def send_notification(user_id: str, message: str) -> str:
    """Rate-limited task: max 5 notifications per minute."""
    logger.info(f"Sending to {user_id}: {message}")
    time.sleep(0.5)
    return f"Notification sent to user {user_id}"


# ─────────────────────────────────────────────
# TASK 4: Distributed Lock  — prevent duplicate runs
# ─────────────────────────────────────────────
@app.task(name="tasks.generate_report")
def generate_report(report_id: str) -> str:
    """
    Uses a Redis distributed lock so that even if this task is
    accidentally triggered twice, only one instance runs at a time.
    """
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    lock_key = f"lock:report:{report_id}"

    # nx=True  → only set if not exists (atomic)
    # ex=60    → auto-release after 60 sec (safety net)
    acquired = r.set(lock_key, "locked", nx=True, ex=60)

    if not acquired:
        logger.warning(f"Report {report_id} already running — skipping.")
        return f"SKIPPED: report {report_id} already in progress"

    try:
        logger.info(f"Generating report {report_id}...")
        time.sleep(3)             # simulate report generation
        return f"Report {report_id} generated successfully"
    finally:
        r.delete(lock_key)        # always release the lock
        logger.info(f"Lock released for report {report_id}")


# ─────────────────────────────────────────────
# TASK 5: Chained tasks (pipeline)
# ─────────────────────────────────────────────
@app.task(name="tasks.fetch_data")
def fetch_data(source: str) -> list:
    """Step 1: simulate fetching raw data."""
    time.sleep(1)
    return [f"{source}_item_{i}" for i in range(5)]


@app.task(name="tasks.transform_data")
def transform_data(items: list) -> list:
    """Step 2: transform the data (uppercase)."""
    time.sleep(1)
    return [item.upper() for item in items]


@app.task(name="tasks.save_data")
def save_data(items: list) -> str:
    """Step 3: simulate saving to DB."""
    time.sleep(1)
    return f"Saved {len(items)} items to database"