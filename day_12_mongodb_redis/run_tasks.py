"""
run_tasks.py — Send tasks to the Celery worker
Run this AFTER starting the worker in another terminal:
    celery -A tasks worker --loglevel=info
"""

from tasks import reverse_string, process_data, send_notification, generate_report
from tasks import fetch_data, transform_data, save_data
from celery import chain
import time

def main():
    print("🚀 Sending tasks to Celery worker...\n")

    # ── Task 1: reverse string ───────────────
    print("── Task 1: Reverse String ──────────────")
    result = reverse_string.delay("Hello Internship Day 12")
    print(f"  Task ID: {result.id}")
    print(f"  Result:  {result.get(timeout=10)}\n")

    # ── Task 2: process data with retry ─────
    print("── Task 2: Process Data ────────────────")
    result = process_data.delay({"user": "Shanmukha", "action": "completed_day12"})
    print(f"  Task ID: {result.id}")
    print(f"  Result:  {result.get(timeout=15)}\n")

    # ── Task 3: rate-limited notifications ──
    print("── Task 3: Rate-Limited Notifications ──")
    for i in range(3):
        result = send_notification.delay(f"user_{i+1}", f"Message {i+1}")
        print(f"  Sent task {i+1}: {result.id}")
    time.sleep(2)
    print()

    # ── Task 4: distributed lock ─────────────
    print("── Task 4: Distributed Lock ────────────")
    r1 = generate_report.delay("monthly_report")
    r2 = generate_report.delay("monthly_report")    # duplicate — will be skipped
    print(f"  Task 1 result: {r1.get(timeout=15)}")
    print(f"  Task 2 result: {r2.get(timeout=15)}\n")

    # ── Task 5: chained pipeline ─────────────
    print("── Task 5: Chained Tasks (pipeline) ────")
    pipeline = chain(
        fetch_data.s("database"),
        transform_data.s(),
        save_data.s()
    )
    result = pipeline.delay()
    print(f"  Pipeline result: {result.get(timeout=30)}\n")

    print("✅ All tasks completed!")

if __name__ == "__main__":
    main()