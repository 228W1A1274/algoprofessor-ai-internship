"""
Day 12 Deliverable: mongo_crud.py
Topics: MongoDB document model, CRUD operations, Atlas Search, Geospatial queries
Run: python3 mongo_crud.py
"""

from pymongo import MongoClient, GEOSPHERE
from pymongo.errors import ConnectionFailure
import datetime

# ─────────────────────────────────────────────
# 1. CONNECTION
# ─────────────────────────────────────────────
def get_db():
    client = MongoClient("mongodb://localhost:27017")
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB!")
    except ConnectionFailure:
        print("❌ MongoDB not running. Start it first.")
        exit(1)
    return client["day12_db"]


# ─────────────────────────────────────────────
# 2. CREATE
# ─────────────────────────────────────────────
def create_documents(db):
    print("\n── CREATE ──────────────────────────────")
    users = db["users"]
    users.drop()  # clean slate each run

    # Insert one
    users.insert_one({
        "name": "Shanmukha",
        "role": "Intern",
        "skills": ["Python", "MongoDB", "Redis"],
        "score": 95,
        "joined": datetime.datetime(2025, 1, 10),
        "location": {
            "type": "Point",
            "coordinates": [78.4867, 17.3850]   # Hyderabad [lng, lat]
        }
    })

    # Insert many
    users.insert_many([
        {
            "name": "Alice",
            "role": "Engineer",
            "skills": ["Java", "Kafka"],
            "score": 88,
            "joined": datetime.datetime(2024, 6, 1),
            "location": {"type": "Point", "coordinates": [72.8777, 19.0760]}  # Mumbai
        },
        {
            "name": "Bob",
            "role": "Intern",
            "skills": ["Python", "Docker"],
            "score": 76,
            "joined": datetime.datetime(2025, 2, 1),
            "location": {"type": "Point", "coordinates": [77.5946, 12.9716]}  # Bangalore
        },
        {
            "name": "Carol",
            "role": "Manager",
            "skills": ["Leadership", "Python"],
            "score": 91,
            "joined": datetime.datetime(2023, 3, 15),
            "location": {"type": "Point", "coordinates": [77.2090, 28.6139]}  # Delhi
        },
    ])
    print(f"  Inserted 4 documents into 'users'")


# ─────────────────────────────────────────────
# 3. READ
# ─────────────────────────────────────────────
def read_documents(db):
    print("\n── READ ────────────────────────────────")
    users = db["users"]

    # find_one
    user = users.find_one({"name": "Shanmukha"})
    print(f"  find_one → {user['name']}, score: {user['score']}")

    # find with filter
    interns = list(users.find({"role": "Intern"}, {"name": 1, "score": 1, "_id": 0}))
    print(f"  Interns: {interns}")

    # find with sort + limit
    top2 = list(users.find().sort("score", -1).limit(2))
    print(f"  Top 2 scores: {[u['name'] for u in top2]}")

    # count
    count = users.count_documents({"skills": "Python"})
    print(f"  Python users count: {count}")


# ─────────────────────────────────────────────
# 4. UPDATE
# ─────────────────────────────────────────────
def update_documents(db):
    print("\n── UPDATE ──────────────────────────────")
    users = db["users"]

    # update_one with $set
    result = users.update_one(
        {"name": "Shanmukha"},
        {"$set": {"score": 99}, "$push": {"skills": "Redis"}}
    )
    print(f"  update_one matched: {result.matched_count}, modified: {result.modified_count}")

    # update_many with $inc
    result = users.update_many(
        {"role": "Intern"},
        {"$inc": {"score": 5}}
    )
    print(f"  update_many (Interns +5 score) modified: {result.modified_count}")

    # upsert (create if not exists)
    users.update_one(
        {"name": "Dave"},
        {"$set": {"role": "Intern", "score": 70, "skills": ["Go"]}},
        upsert=True
    )
    print("  Upserted Dave (created because he didn't exist)")


# ─────────────────────────────────────────────
# 5. DELETE
# ─────────────────────────────────────────────
def delete_documents(db):
    print("\n── DELETE ──────────────────────────────")
    users = db["users"]

    # delete_one
    result = users.delete_one({"name": "Dave"})
    print(f"  delete_one Dave: deleted {result.deleted_count}")

    # delete_many
    result = users.delete_many({"score": {"$lt": 80}})
    print(f"  delete_many (score < 80): deleted {result.deleted_count}")


# ─────────────────────────────────────────────
# 6. GEOSPATIAL QUERIES (2dsphere index)
# ─────────────────────────────────────────────
def geospatial_queries(db):
    print("\n── GEOSPATIAL ──────────────────────────")
    users = db["users"]

    # Create 2dsphere index on location field
    users.create_index([("location", GEOSPHERE)])
    print("  Created 2dsphere index on 'location'")

    # $near: find users near Hyderabad within 1000 km
    hyderabad = [78.4867, 17.3850]
    nearby = list(users.find({
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": hyderabad},
                "$maxDistance": 1_000_000   # metres = 1000 km
            }
        }
    }, {"name": 1, "location": 1, "_id": 0}))
    print(f"  Users within 1000 km of Hyderabad: {[u['name'] for u in nearby]}")

    # $geoWithin: box covering South India
    south_india = list(users.find({
        "location": {
            "$geoWithin": {
                "$box": [[70, 8], [82, 20]]   # [SW corner, NE corner]
            }
        }
    }, {"name": 1, "_id": 0}))
    print(f"  Users inside South India bounding box: {[u['name'] for u in south_india]}")


# ─────────────────────────────────────────────
# 7. CHANGE STREAMS (motor / real-time demo note)
# ─────────────────────────────────────────────
def change_stream_info():
    print("\n── CHANGE STREAMS (info) ───────────────")
    print("  Change streams require a MongoDB Replica Set.")
    print("  In production, use: `motor` (async) + watch() for real-time event-driven updates.")
    print("  Example (async):")
    print("    async with collection.watch() as stream:")
    print("        async for change in stream:")
    print("            print(change)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    db = get_db()
    create_documents(db)
    read_documents(db)
    update_documents(db)
    delete_documents(db)
    geospatial_queries(db)
    change_stream_info()
    print("\n✅ mongo_crud.py completed successfully!\n")