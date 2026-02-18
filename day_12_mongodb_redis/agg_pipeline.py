"""
Day 12 Deliverable: agg_pipeline.py
Topics: Aggregation pipeline — $match, $group, $project, $sort, $lookup, $unwind
Run: python3 agg_pipeline.py  (run mongo_crud.py first to seed data)
"""

from pymongo import MongoClient
import datetime

client = MongoClient("mongodb://localhost:27017")
db = client["day12_db"]


# ─────────────────────────────────────────────
# SEED DATA (in case you run this file alone)
# ─────────────────────────────────────────────
def seed_data():
    db["users"].drop()
    db["orders"].drop()

    db["users"].insert_many([
        {"name": "Shanmukha", "role": "Intern",    "dept": "Engineering", "score": 99, "skills": ["Python", "MongoDB", "Redis"]},
        {"name": "Alice",     "role": "Engineer",  "dept": "Engineering", "score": 88, "skills": ["Java", "Kafka"]},
        {"name": "Bob",       "role": "Intern",    "dept": "HR",          "score": 81, "skills": ["Python", "Docker"]},
        {"name": "Carol",     "role": "Manager",   "dept": "Engineering", "score": 91, "skills": ["Leadership", "Python"]},
        {"name": "Dave",      "role": "Engineer",  "dept": "HR",          "score": 73, "skills": ["Go", "Redis"]},
    ])

    db["orders"].insert_many([
        {"user": "Shanmukha", "product": "Laptop",  "amount": 80000, "status": "completed"},
        {"user": "Shanmukha", "product": "Mouse",   "amount": 1500,  "status": "completed"},
        {"user": "Alice",     "product": "Monitor", "amount": 25000, "status": "pending"},
        {"user": "Bob",       "product": "Keyboard","amount": 3000,  "status": "completed"},
        {"user": "Carol",     "product": "Laptop",  "amount": 80000, "status": "completed"},
        {"user": "Dave",      "product": "Webcam",  "amount": 5000,  "status": "cancelled"},
    ])
    print("✅ Seed data inserted.\n")


# ─────────────────────────────────────────────
# 1. $match + $group + $sort  — most-used combo
# ─────────────────────────────────────────────
def pipeline_group_by_role():
    print("── Pipeline 1: Average score by role ──")
    pipeline = [
        {"$match": {"score": {"$gte": 70}}},            # filter first (fast)
        {"$group": {
            "_id": "$role",                              # group by role
            "avg_score":  {"$avg": "$score"},
            "max_score":  {"$max": "$score"},
            "total_users": {"$sum": 1}
        }},
        {"$sort": {"avg_score": -1}},                   # highest avg first
        {"$project": {
            "role": "$_id",
            "avg_score": {"$round": ["$avg_score", 1]},
            "max_score": 1,
            "total_users": 1,
            "_id": 0
        }}
    ]
    results = list(db["users"].aggregate(pipeline))
    for r in results:
        print(f"  {r}")


# ─────────────────────────────────────────────
# 2. $unwind  — flatten arrays
# ─────────────────────────────────────────────
def pipeline_skill_count():
    print("\n── Pipeline 2: Most common skills ──")
    pipeline = [
        {"$unwind": "$skills"},                         # one doc per skill
        {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    results = list(db["users"].aggregate(pipeline))
    for r in results:
        print(f"  {r['_id']:15s} → {r['count']} users")


# ─────────────────────────────────────────────
# 3. $lookup  — JOIN with another collection
# ─────────────────────────────────────────────
def pipeline_user_orders():
    print("\n── Pipeline 3: Users with their orders ($lookup) ──")
    pipeline = [
        {"$lookup": {
            "from":         "orders",
            "localField":   "name",
            "foreignField": "user",
            "as":           "user_orders"
        }},
        {"$project": {
            "name": 1,
            "role": 1,
            "total_spent": {"$sum": "$user_orders.amount"},
            "order_count": {"$size": "$user_orders"},
            "_id": 0
        }},
        {"$sort": {"total_spent": -1}}
    ]
    results = list(db["users"].aggregate(pipeline))
    for r in results:
        print(f"  {r['name']:12s}  orders: {r['order_count']}  spent: ₹{r['total_spent']:,}")


# ─────────────────────────────────────────────
# 4. $bucket  — range grouping
# ─────────────────────────────────────────────
def pipeline_score_buckets():
    print("\n── Pipeline 4: Score distribution ($bucket) ──")
    pipeline = [
        {"$bucket": {
            "groupBy": "$score",
            "boundaries": [0, 70, 80, 90, 101],
            "default": "Other",
            "output": {
                "count": {"$sum": 1},
                "names": {"$push": "$name"}
            }
        }}
    ]
    results = list(db["users"].aggregate(pipeline))
    labels = {0: "0-69 (Low)", 70: "70-79 (OK)", 80: "80-89 (Good)", 90: "90-100 (Excellent)"}
    for r in results:
        label = labels.get(r["_id"], str(r["_id"]))
        print(f"  {label:25s}: {r['names']}")


# ─────────────────────────────────────────────
# 5. $addFields + $filter  — computed fields
# ─────────────────────────────────────────────
def pipeline_add_fields():
    print("\n── Pipeline 5: Add computed fields ──")
    pipeline = [
        {"$addFields": {
            "grade": {
                "$switch": {
                    "branches": [
                        {"case": {"$gte": ["$score", 90]}, "then": "A"},
                        {"case": {"$gte": ["$score", 80]}, "then": "B"},
                        {"case": {"$gte": ["$score", 70]}, "then": "C"},
                    ],
                    "default": "F"
                }
            },
            "python_user": {"$in": ["Python", "$skills"]}
        }},
        {"$project": {"name": 1, "score": 1, "grade": 1, "python_user": 1, "_id": 0}},
        {"$sort": {"score": -1}}
    ]
    results = list(db["users"].aggregate(pipeline))
    for r in results:
        print(f"  {r['name']:12s}  score: {r['score']}  grade: {r['grade']}  Python: {r['python_user']}")


# ─────────────────────────────────────────────
# 6. $facet  — multiple pipelines in one query
# ─────────────────────────────────────────────
def pipeline_facet():
    print("\n── Pipeline 6: Multi-facet analytics ($facet) ──")
    pipeline = [
        {"$facet": {
            "by_dept": [
                {"$group": {"_id": "$dept", "count": {"$sum": 1}, "avg": {"$avg": "$score"}}},
                {"$sort": {"avg": -1}}
            ],
            "by_role": [
                {"$group": {"_id": "$role", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "top_scorer": [
                {"$sort": {"score": -1}},
                {"$limit": 1},
                {"$project": {"name": 1, "score": 1, "_id": 0}}
            ]
        }}
    ]
    result = list(db["users"].aggregate(pipeline))[0]
    print(f"  By Dept: {result['by_dept']}")
    print(f"  By Role: {result['by_role']}")
    print(f"  Top Scorer: {result['top_scorer']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    seed_data()
    pipeline_group_by_role()
    pipeline_skill_count()
    pipeline_user_orders()
    pipeline_score_buckets()
    pipeline_add_fields()
    pipeline_facet()
    print("\n✅ agg_pipeline.py completed successfully!\n")