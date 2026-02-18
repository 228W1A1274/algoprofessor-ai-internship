-- ============================================
-- ALL POSTGRESQL INDEX TYPES (BASIC VERSION)
-- ============================================

-- ============================================
-- PART 1: CREATE SAMPLE TABLES
-- ============================================

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location POINT
);

-- Insert sample data
INSERT INTO products (name, category, price, tags, metadata, location)
VALUES
('Gaming Laptop', 'Electronics', 1500,
 ARRAY['gaming','laptop'],
 '{"brand":"ASUS","ram":"16GB"}',
 POINT(10,20)),

('Office Laptop', 'Electronics', 800,
 ARRAY['office','laptop'],
 '{"brand":"Dell","ram":"8GB"}',
 POINT(15,25)),

('Mouse', 'Accessories', 50,
 ARRAY['mouse','wireless'],
 '{"brand":"Logitech"}',
 POINT(5,10));

-- ============================================
-- PART 2: B-TREE INDEX (DEFAULT)
-- ============================================

-- Best for:
-- =, <, >, BETWEEN, ORDER BY

CREATE INDEX idx_price_btree
ON products(price);

-- Test
SELECT * FROM products
WHERE price BETWEEN 500 AND 2000;

-- ============================================
-- PART 3: HASH INDEX
-- ============================================

-- Best for exact match (=)

CREATE INDEX idx_category_hash
ON products USING HASH (category);

-- Test
SELECT * FROM products
WHERE category = 'Electronics';

-- ============================================
-- PART 4: GIN INDEX (JSONB)
-- ============================================

-- Best for JSONB and arrays

CREATE INDEX idx_metadata_gin
ON products USING GIN (metadata);

-- Test JSON search
SELECT *
FROM products
WHERE metadata @> '{"brand":"ASUS"}';

-- ============================================
-- PART 5: GIN INDEX (ARRAY)
-- ============================================

CREATE INDEX idx_tags_gin
ON products USING GIN (tags);

-- Test array search
SELECT *
FROM products
WHERE tags @> ARRAY['gaming'];

-- ============================================
-- PART 6: GiST INDEX (LOCATION)
-- ============================================

-- Best for geometric data

CREATE INDEX idx_location_gist
ON products USING GiST (location);

-- Find nearest location
SELECT *,
location <-> POINT(12,22) AS distance
FROM products
ORDER BY distance
LIMIT 1;

-- ============================================
-- PART 7: BRIN INDEX (LARGE TABLES)
-- ============================================

-- Best for very large tables and dates

CREATE INDEX idx_created_brin
ON products USING BRIN (created_at);

-- Test
SELECT *
FROM products
WHERE created_at >= CURRENT_DATE - INTERVAL '1 day';

-- ============================================
-- PART 8: VIEW ALL INDEXES
-- ============================================

SELECT
indexname,
tablename
FROM pg_indexes
WHERE schemaname = 'public';

-- ============================================
-- PART 9: CHECK INDEX USAGE
-- ============================================

EXPLAIN ANALYZE
SELECT * FROM products
WHERE price = 1500;

-- ============================================
-- SUMMARY
-- ============================================

SELECT 'All basic index types created successfully!' AS message;

SELECT 'Indexes covered: B-tree, Hash, GIN, GiST, BRIN' AS summary;
