# Internship Project - Day 11: SQL & PostgreSQL

**Author:** [Your Name]  
**Date:** February 16, 2026  
**Topic:** Relational Databases - PostgreSQL & SQL

---

## 📁 Project Structure

```
InternshipProject/
├── db_schema.sql           # Complete database schema (DDL)
├── sample_data.sql         # Sample data for testing
├── complex_queries.sql     # 40 advanced SQL queries
├── verification_queries.sql # Database verification scripts
├── orm_models.py           # SQLAlchemy ORM models (coming next)
├── async_queries.py        # Async database operations (coming next)
├── README.md               # This file
├── migrations/
│   ├── 001_initial.sql     # Initial schema migration
│   └── 002_add_indexes.sql # Performance indexes migration
└── benchmarks/
    └── query_performance.txt # Query performance results
```

---

## 🚀 Quick Start Guide

### Prerequisites

1. PostgreSQL 16+ installed
2. VS Code installed
3. Basic command line knowledge

### Setup Steps

#### 1. Create Database

```bash
# Open PowerShell or Command Prompt
psql -U postgres

# In psql, create database:
CREATE DATABASE internship_db;
\c internship_db
\q
```

#### 2. Run Schema

```bash
cd C:\Users\YourName\Desktop\InternshipProject
psql -U postgres -d internship_db -f db_schema.sql
```

#### 3. Load Sample Data

```bash
psql -U postgres -d internship_db -f sample_data.sql
```

#### 4. Test Queries

```bash
psql -U postgres -d internship_db -f complex_queries.sql
```

#### 5. Verify Setup

```bash
psql -U postgres -d internship_db -f verification_queries.sql
```

---

## 📚 Topics Covered

### ✅ DDL (Data Definition Language)

- `CREATE TABLE` with constraints
- `ALTER TABLE` modifications
- `DROP` and `TRUNCATE`
- Primary keys, foreign keys
- Indexes for performance
- Views, functions, triggers

### ✅ DML (Data Manipulation Language)

- `INSERT` (single, bulk, with RETURNING)
- `UPDATE` (with conditions, subqueries)
- `DELETE` (with RETURNING)
- Upserts (ON CONFLICT)

### ✅ DQL (Data Query Language)

- `SELECT` with filtering
- `WHERE`, `ORDER BY`, `LIMIT`
- Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- `GROUP BY` and `HAVING`
- Pattern matching (LIKE, ILIKE)

### ✅ Complex JOINs

- `INNER JOIN`
- `LEFT JOIN` / `RIGHT JOIN`
- `FULL OUTER JOIN`
- `CROSS JOIN`
- `SELF JOIN`
- Multi-table joins (4+ tables)

### ✅ CTEs (Common Table Expressions)

- Basic `WITH` clause
- Multiple CTEs in single query
- CTEs with DML operations
- Improved query readability

### ✅ Recursive Queries

- `WITH RECURSIVE` syntax
- Hierarchical data traversal
- Category trees (top-down, bottom-up)
- Date series generation

### ✅ Window Functions

- `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`
- `LEAD()` and `LAG()`
- `FIRST_VALUE()`, `LAST_VALUE()`
- `NTILE()` for bucketing
- Aggregate window functions
- `PARTITION BY` and window frames

---

## 🗄️ Database Schema

### Tables

#### 1. **categories**

Hierarchical product categories

```sql
- category_id (PK)
- category_name
- parent_id (FK) → self-reference
```

#### 2. **users**

Customer information

```sql
- user_id (PK)
- username (UNIQUE)
- email (UNIQUE)
- phone_number
- age (CHECK >= 18)
- created_at, updated_at
```

#### 3. **products**

Product catalog with inventory

```sql
- product_id (PK)
- name
- description
- price (CHECK > 0)
- stock_quantity (CHECK >= 0)
- category
- category_id (FK) → categories
- is_active
```

#### 4. **orders**

Customer orders

```sql
- order_id (PK)
- user_id (FK) → users (CASCADE)
- order_date
- total_amount
- status (pending/processing/shipped/delivered/cancelled)
- shipping_address
```

#### 5. **order_items**

Order line items (many-to-many)

```sql
- order_item_id (PK)
- order_id (FK) → orders (CASCADE)
- product_id (FK) → products (RESTRICT)
- quantity (CHECK > 0)
- unit_price
- subtotal (GENERATED/COMPUTED)
```

### Views

1. **order_summary** - Orders with customer details
2. **inventory_status** - Product stock levels
3. **customer_lifetime_value** - Customer metrics

### Functions & Triggers

1. **update_product_stock()** - Auto-reduce inventory
2. **update_updated_at_column()** - Auto-timestamp
3. **calculate_order_total()** - Compute order totals

---

## 🔍 Key Queries in complex_queries.sql

The file contains **40 queries** covering:

1-6: Basic SELECT, WHERE, filtering  
7-10: Aggregates and GROUP BY  
11-15: Various JOIN types  
16-19: Subqueries  
20-22: CTEs  
23-25: Recursive CTEs  
26-32: Window functions  
33-36: Advanced analytics  
37-38: DML with RETURNING  
39-40: Complex business logic

---

## 📊 Sample Data

- **8 users** (customers)
- **17 products** across 3 main categories
- **8 categories** (hierarchical structure)
- **8 orders** with various statuses
- **15 order items** (order line items)

---

## 🛠️ Useful Commands

### Connect to Database

```bash
psql -U postgres -d internship_db
```

### List Tables

```sql
\dt
```

### Describe Table

```sql
\d users
\d+ products  -- detailed info
```

### Run SQL File

```sql
\i 'C:/path/to/file.sql'
```

### Enable Query Timing

```sql
\timing on
```

### View Query Plan

```sql
EXPLAIN ANALYZE SELECT * FROM products;
```

### Exit psql

```sql
\q
```

---

## 📈 Performance Tips

✅ **Always use indexes** on foreign keys  
✅ **Use EXPLAIN ANALYZE** to check query plans  
✅ **Avoid SELECT \*** in production  
✅ **Use appropriate data types**  
✅ **Add indexes** on frequently queried columns  
✅ **Use CTEs** for complex queries (readability)  
✅ **Test with realistic data volumes**

---

## 🎯 Next Steps (Coming Soon)

- [ ] **PostgreSQL-specific features** (JSONB, advanced indexing, partitioning)
- [ ] **SQLAlchemy ORM** (orm_models.py)
- [ ] **Async queries** (async_queries.py with asyncpg)
- [ ] **Alembic migrations** (proper migration management)
- [ ] **Performance benchmarks** (detailed analysis)

---

## 📝 Common Mistakes to Avoid

❌ Forgetting `WHERE` clause in UPDATE/DELETE  
❌ Not using aliases in complex JOINs  
❌ Confusing `HAVING` vs `WHERE`  
❌ Forgetting `UNBOUNDED FOLLOWING` in `LAST_VALUE()`  
❌ Not testing queries before running on production  
❌ Using `SELECT *` when only few columns needed  
❌ Not creating indexes on foreign keys

---

## 🐛 Troubleshooting

### "database does not exist"

```bash
# Create it first:
psql -U postgres -c "CREATE DATABASE internship_db;"
```

### "permission denied"

```bash
# Make sure you're using postgres user:
psql -U postgres -d internship_db
```

### "relation already exists"

```bash
# Drop and recreate:
psql -U postgres -d internship_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
```

### "syntax error"

```bash
# Check PostgreSQL version:
SELECT version();
# Make sure it's 12+
```

---

## 📧 Contact

For questions or issues, contact: [Your Email]

---

**Status:** ✅ Day 11 - SQL Fundamentals Complete  
**Last Updated:** 2026-02-16  
**PostgreSQL Version:** 16.x  
**Total Queries:** 40+  
**Total Tables:** 5  
**Total Views:** 3  
**Total Functions:** 3

## ✅ Topic 2: PostgreSQL-Specific Features (COMPLETED)

### JSONB

- Created 3 tables with JSONB columns
- JSONB operators: ->, ->>, @>, ?, ?|, ?&
- JSONB functions: jsonb_set, jsonb_agg, jsonb_build_object
- GIN indexes on JSONB for fast queries

### Advanced Indexing

- **B-tree**: Default, best for ranges and sorting
- **GIN**: Full-text search, JSONB, arrays
- **GiST**: Geometric data, ranges
- **BRIN**: Very large tables, time-series
- **Hash**: Equality only (rarely used)

### Table Partitioning

- **Range partitioning**: By date/time (most common)
- **List partitioning**: By discrete values
- **Hash partitioning**: Even distribution
- **Sub-partitioning**: Multi-level partitions
- Partition pruning for query optimization

### Files Created

- `jsonb_examples.sql` - JSONB operations and queries
- `advanced_indexing.sql` - All index types with examples
- `partitioning.sql` - Range, List, Hash partitioning
- `topic2_verification.sql` - Verification queries
