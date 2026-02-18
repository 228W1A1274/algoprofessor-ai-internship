-- ============================================
-- ALL PARTITIONING TYPES - POSTGRESQL (BASIC)
-- ============================================

-- ============================================
-- PART 1: RANGE PARTITIONING (BY DATE)
-- ============================================

-- Parent table
CREATE TABLE orders (
    order_id SERIAL,
    user_id INT,
    order_date DATE,
    amount DECIMAL(10,2)
)
PARTITION BY RANGE (order_date);

-- Partitions
CREATE TABLE orders_2024_jan PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_feb PARTITION OF orders
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE orders_future PARTITION OF orders
FOR VALUES FROM ('2024-03-01') TO (MAXVALUE);

-- Insert data
INSERT INTO orders (user_id, order_date, amount)
VALUES
(1,'2024-01-10',500),
(2,'2024-02-15',800),
(3,'2024-03-20',1000);

-- Query
SELECT * FROM orders
WHERE order_date = '2024-02-15';

-- ============================================
-- PART 2: LIST PARTITIONING (BY CATEGORY)
-- ============================================

CREATE TABLE products (
    product_id SERIAL,
    name TEXT,
    category TEXT,
    price DECIMAL
)
PARTITION BY LIST (category);

CREATE TABLE products_electronics
PARTITION OF products
FOR VALUES IN ('Electronics');

CREATE TABLE products_furniture
PARTITION OF products
FOR VALUES IN ('Furniture');

CREATE TABLE products_other
PARTITION OF products
DEFAULT;

-- Insert
INSERT INTO products (name, category, price)
VALUES
('Laptop','Electronics',1200),
('Chair','Furniture',200),
('Book','Education',50);

-- Query
SELECT * FROM products
WHERE category = 'Electronics';

-- ============================================
-- PART 3: HASH PARTITIONING (BY ID)
-- ============================================

CREATE TABLE users (
    user_id INT,
    name TEXT
)
PARTITION BY HASH (user_id);

CREATE TABLE users_part1
PARTITION OF users
FOR VALUES WITH (MODULUS 2, REMAINDER 0);

CREATE TABLE users_part2
PARTITION OF users
FOR VALUES WITH (MODULUS 2, REMAINDER 1);

-- Insert
INSERT INTO users VALUES
(1,'Ram'),
(2,'Sam'),
(3,'John'),
(4,'David');

-- Query
SELECT * FROM users
WHERE user_id = 3;

-- ============================================
-- PART 4: SUB-PARTITIONING
-- RANGE → LIST
-- ============================================

CREATE TABLE sales (
    sale_id SERIAL,
    sale_date DATE,
    region TEXT,
    amount DECIMAL
)
PARTITION BY RANGE (sale_date);

-- Year partition
CREATE TABLE sales_2024
PARTITION OF sales
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
PARTITION BY LIST (region);

-- Sub partitions
CREATE TABLE sales_2024_india
PARTITION OF sales_2024
FOR VALUES IN ('India');

CREATE TABLE sales_2024_us
PARTITION OF sales_2024
FOR VALUES IN ('US');

CREATE TABLE sales_2024_other
PARTITION OF sales_2024
DEFAULT;

-- Insert
INSERT INTO sales (sale_date, region, amount)
VALUES
('2024-05-10','India',5000),
('2024-06-15','US',7000),
('2024-07-01','UK',3000);

-- Query
SELECT * FROM sales
WHERE region='India';

-- ============================================
-- PART 5: VIEW PARTITIONS
-- ============================================

SELECT
parent.relname AS parent_table,
child.relname AS partition
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid;

-- ============================================
-- SUCCESS MESSAGE
-- ============================================

SELECT 'All partition types created successfully!' AS message;

SELECT 'Types: RANGE, LIST, HASH, SUB-PARTITION' AS summary;
