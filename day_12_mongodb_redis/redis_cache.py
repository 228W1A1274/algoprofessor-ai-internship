"""
Day 12 Deliverable: redis_cache.py
Topics: Strings, Hashes, Lists, Sets, Sorted Sets, Pub/Sub, Streams,
        Geospatial, HyperLogLog, Caching patterns, TTL, Eviction
Run:  python3 redis_cache.py
Note: Requires Redis running → in Ubuntu terminal: sudo service redis-server start
"""

import redis
import json
import time
import threading

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def check_connection():
    try:
        r.ping()
        print("✅ Redis connected!\n")
    except redis.ConnectionError:
        print("❌ Redis not running. Start it: sudo service redis-server start")
        exit(1)

def flush_test_keys():
    """Clean up all demo keys before running."""
    for key in r.keys("demo:*"):
        r.delete(key)


# ─────────────────────────────────────────────
# 1. STRINGS  — most basic type, used for cache
# ─────────────────────────────────────────────
def demo_strings():
    print("── 1. STRINGS ──────────────────────────")
    r.set("demo:name", "Shanmukha")
    r.set("demo:score", 99)
    r.set("demo:session", "abc123", ex=30)          # TTL = 30 seconds

    print(f"  GET name       → {r.get('demo:name')}")
    print(f"  GET score      → {r.get('demo:score')}")
    print(f"  TTL session    → {r.ttl('demo:session')} sec")

    r.incr("demo:score", 1)
    print(f"  INCR score     → {r.get('demo:score')}")

    r.append("demo:name", " Intern")
    print(f"  APPEND name    → {r.get('demo:name')}")


# ─────────────────────────────────────────────
# 2. HASHES  — like a dict / object store
# ─────────────────────────────────────────────
def demo_hashes():
    print("\n── 2. HASHES ───────────────────────────")
    r.hset("demo:user:1", mapping={
        "name": "Shanmukha",
        "role": "Intern",
        "score": "99",
        "city": "Hyderabad"
    })
    print(f"  HGET name      → {r.hget('demo:user:1', 'name')}")
    print(f"  HGETALL        → {r.hgetall('demo:user:1')}")

    r.hincrby("demo:user:1", "score", 1)
    print(f"  HINCRBY score  → {r.hget('demo:user:1', 'score')}")

    r.hdel("demo:user:1", "city")
    print(f"  HDEL city. Keys now: {r.hkeys('demo:user:1')}")


# ─────────────────────────────────────────────
# 3. LISTS  — queues / recent items
# ─────────────────────────────────────────────
def demo_lists():
    print("\n── 3. LISTS ────────────────────────────")
    key = "demo:activity"
    r.delete(key)

    r.rpush(key, "login", "viewed_dashboard", "clicked_profile")  # add to right
    r.lpush(key, "app_start")                                      # add to left (most recent)

    print(f"  Full list      → {r.lrange(key, 0, -1)}")
    print(f"  LLEN           → {r.llen(key)}")
    print(f"  LPOP           → {r.lpop(key)}")
    print(f"  After pop      → {r.lrange(key, 0, -1)}")

    # Trim to last 3 (common for recent-activity feeds)
    r.ltrim(key, -3, -1)
    print(f"  After LTRIM 3  → {r.lrange(key, 0, -1)}")


# ─────────────────────────────────────────────
# 4. SETS  — unique items, tags, memberships
# ─────────────────────────────────────────────
def demo_sets():
    print("\n── 4. SETS ─────────────────────────────")
    r.delete("demo:skills:shanmukha", "demo:skills:alice")

    r.sadd("demo:skills:shanmukha", "Python", "MongoDB", "Redis", "Docker")
    r.sadd("demo:skills:alice",     "Python", "Java", "Kafka", "Docker")

    print(f"  Shanmukha skills → {r.smembers('demo:skills:shanmukha')}")
    print(f"  SISMEMBER Redis  → {r.sismember('demo:skills:shanmukha', 'Redis')}")
    print(f"  INTERSECTION     → {r.sinter('demo:skills:shanmukha', 'demo:skills:alice')}")
    print(f"  UNION            → {r.sunion('demo:skills:shanmukha', 'demo:skills:alice')}")
    print(f"  DIFFERENCE       → {r.sdiff('demo:skills:shanmukha', 'demo:skills:alice')}")


# ─────────────────────────────────────────────
# 5. SORTED SETS  — leaderboards, rankings
# ─────────────────────────────────────────────
def demo_sorted_sets():
    print("\n── 5. SORTED SETS (Leaderboard) ────────")
    key = "demo:leaderboard"
    r.delete(key)

    r.zadd(key, {"Shanmukha": 99, "Alice": 88, "Bob": 81, "Carol": 91, "Dave": 73})

    # Top 3
    top3 = r.zrange(key, 0, 2, rev=True, withscores=True)
    print(f"  Top 3          → {top3}")

    # Rank of a user (0-indexed)
    rank = r.zrevrank(key, "Shanmukha")
    print(f"  Shanmukha rank → #{rank + 1}")

    # Score range: users scoring 80–100
    in_range = r.zrangebyscore(key, 80, 100, withscores=True)
    print(f"  Score 80-100   → {in_range}")

    # Increment score
    r.zincrby(key, 5, "Bob")
    print(f"  Bob after +5   → {r.zscore(key, 'Bob')}")


# ─────────────────────────────────────────────
# 6. PUB/SUB  — real-time messaging
# ─────────────────────────────────────────────
def demo_pubsub():
    print("\n── 6. PUB/SUB ──────────────────────────")
    received = []

    def subscriber():
        sub = r.pubsub()
        sub.subscribe("demo:notifications")
        for msg in sub.listen():
            if msg["type"] == "message":
                received.append(msg["data"])
                if len(received) >= 2:
                    sub.unsubscribe()
                    break

    t = threading.Thread(target=subscriber, daemon=True)
    t.start()
    time.sleep(0.1)  # give subscriber time to connect

    r.publish("demo:notifications", "Task completed: mongo_crud.py")
    r.publish("demo:notifications", "Task completed: agg_pipeline.py")
    t.join(timeout=3)
    print(f"  Messages received: {received}")


# ─────────────────────────────────────────────
# 7. STREAMS  — persistent event log
# ─────────────────────────────────────────────
def demo_streams():
    print("\n── 7. STREAMS ──────────────────────────")
    key = "demo:events"
    r.delete(key)

    # Add events
    r.xadd(key, {"user": "Shanmukha", "action": "login"})
    r.xadd(key, {"user": "Alice",     "action": "purchase"})
    r.xadd(key, {"user": "Bob",       "action": "logout"})

    events = r.xrange(key, "-", "+")
    print(f"  Stream length  → {r.xlen(key)}")
    for eid, data in events:
        print(f"  [{eid}] {data}")


# ─────────────────────────────────────────────
# 8. GEOSPATIAL  — location-based search
# ─────────────────────────────────────────────
def demo_geospatial():
    print("\n── 8. GEOSPATIAL ───────────────────────")
    key = "demo:cities"
    r.delete(key)

    # GEOADD key lng lat member
    r.geoadd(key, [78.4867, 17.3850, "Hyderabad"])
    r.geoadd(key, [72.8777, 19.0760, "Mumbai"])
    r.geoadd(key, [77.5946, 12.9716, "Bangalore"])
    r.geoadd(key, [77.2090, 28.6139, "Delhi"])

    # Distance between two cities
    dist = r.geodist(key, "Hyderabad", "Bangalore", unit="km")
    print(f"  Hyderabad→Bangalore = {dist:.1f} km")

    # Cities within 600 km of Hyderabad
    nearby = r.geosearch(
        key,
        longitude=78.4867, latitude=17.3850,
        radius=600, unit="km",
        sort="ASC", withcoord=True, withdist=True
    )
    print("  Cities within 600 km of Hyderabad:")
    for city in nearby:
        print(f"    {city[0]:12s} {city[1]:.1f} km  ({city[2][1]:.2f}°N, {city[2][0]:.2f}°E)")


# ─────────────────────────────────────────────
# 9. HYPERLOGLOG  — approximate unique counts
# ─────────────────────────────────────────────
def demo_hyperloglog():
    print("\n── 9. HYPERLOGLOG (unique visitors) ────")
    r.delete("demo:visitors:day1", "demo:visitors:day2")

    # Simulate 1000 visitors (some overlap)
    for i in range(1000):
        r.pfadd("demo:visitors:day1", f"user_{i}")
    for i in range(700, 1500):
        r.pfadd("demo:visitors:day2", f"user_{i}")

    day1_count = r.pfcount("demo:visitors:day1")
    day2_count = r.pfcount("demo:visitors:day2")
    total_unique = r.pfcount("demo:visitors:day1", "demo:visitors:day2")

    print(f"  Day 1 unique   ≈ {day1_count}")
    print(f"  Day 2 unique   ≈ {day2_count}")
    print(f"  Total unique   ≈ {total_unique}  (actual: 1500)")


# ─────────────────────────────────────────────
# 10. TTL & EVICTION POLICIES
# ─────────────────────────────────────────────
def demo_ttl_eviction():
    print("\n── 10. TTL & EVICTION ──────────────────")
    r.set("demo:short_lived", "expires soon", ex=5)
    r.set("demo:permanent",   "lives forever")

    print(f"  TTL short_lived → {r.ttl('demo:short_lived')} sec")
    print(f"  TTL permanent   → {r.ttl('demo:permanent')} (-1 = no expiry)")

    # PERSIST removes the TTL
    r.set("demo:promote", "was temporary", ex=60)
    r.persist("demo:promote")
    print(f"  TTL after PERSIST → {r.ttl('demo:promote')} (-1 = now permanent)")

    # Get current eviction policy
    policy = r.config_get("maxmemory-policy")
    print(f"  Current eviction policy → {policy}")
    print("  Common policies: noeviction | allkeys-lru | volatile-lru | allkeys-lfu")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    check_connection()
    flush_test_keys()
    demo_strings()
    demo_hashes()
    demo_lists()
    demo_sets()
    demo_sorted_sets()
    demo_pubsub()
    demo_streams()
    demo_geospatial()
    demo_hyperloglog()
    demo_ttl_eviction()
    print("\n✅ redis_cache.py completed successfully!\n")