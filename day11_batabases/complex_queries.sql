-- ============================================
-- COMPLEX SQL QUERIES - ALL CONCEPTS
-- Day 11: SQL Practice
-- ============================================

-- ============================================
-- SECTION 1: BASIC QUERIES (DQL)
-- ============================================

-- Query 1: Select all users
SELECT * FROM users ORDER BY created_at DESC;

-- Query 2: Filter with WHERE
SELECT 
    username,
    email,
    age
FROM users
WHERE age > 25
ORDER BY age DESC;

-- Query 3: Pattern matching with LIKE
SELECT 
    product_id,
    name,
    price
FROM products
WHERE name ILIKE '%laptop%'  -- case-insensitive
ORDER BY price DESC;

-- Query 4: Multiple conditions
SELECT 
    name,
    price,
    stock_quantity
FROM products
WHERE category = 'Electronics'
    AND price > 500
    AND stock_quantity > 0
ORDER BY price;

-- Query 5: IN operator
SELECT 
    username,
    email
FROM users
WHERE username IN ('john_doe', 'jane_smith', 'alice_wong');

-- Query 6: BETWEEN for ranges
SELECT 
    name,
    price,
    category
FROM products
WHERE price BETWEEN 100 AND 500
ORDER BY price;

-- ============================================
-- SECTION 2: AGGREGATE FUNCTIONS
-- ============================================

-- Query 7: Basic aggregates
SELECT 
    COUNT(*) AS total_products,
    AVG(price) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    SUM(stock_quantity) AS total_stock
FROM products;

-- Query 8: COUNT with DISTINCT
SELECT 
    COUNT(DISTINCT category) AS unique_categories,
    COUNT(DISTINCT user_id) AS total_customers
FROM products, orders;

-- Query 9: GROUP BY
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price,
    SUM(stock_quantity) AS total_stock
FROM products
GROUP BY category
ORDER BY product_count DESC;

-- Query 10: GROUP BY with HAVING
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price
FROM products
GROUP BY category
HAVING COUNT(*) >= 3
ORDER BY avg_price DESC;

-- ============================================
-- SECTION 3: JOINS
-- ============================================

-- Query 11: INNER JOIN - Orders with customer names
SELECT 
    o.order_id,
    u.username,
    u.email,
    o.order_date,
    o.total_amount,
    o.status
FROM orders o
INNER JOIN users u ON o.user_id = u.user_id
ORDER BY o.order_date DESC;

-- Query 12: Multiple INNER JOINs - Full order details
SELECT 
    u.username AS customer,
    o.order_id,
    o.order_date,
    p.name AS product,
    p.category,
    oi.quantity,
    oi.unit_price,
    oi.subtotal
FROM users u
INNER JOIN orders o ON u.user_id = o.user_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
WHERE o.status = 'delivered'
ORDER BY o.order_date DESC, o.order_id;

-- Query 13: LEFT JOIN - Find users who never ordered
SELECT 
    u.user_id,
    u.username,
    u.email,
    u.created_at
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_id IS NULL;

-- Query 14: LEFT JOIN - All users with order count
SELECT 
    u.username,
    u.email,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.username, u.email
ORDER BY total_spent DESC;

-- Query 15: SELF JOIN - Products in same category
SELECT 
    p1.name AS product1,
    p2.name AS product2,
    p1.category,
    p1.price AS price1,
    p2.price AS price2
FROM products p1
INNER JOIN products p2 ON p1.category = p2.category 
    AND p1.product_id < p2.product_id
WHERE p1.category = 'Electronics'
ORDER BY p1.name;

-- ============================================
-- SECTION 4: SUBQUERIES
-- ============================================

-- Query 16: Subquery in WHERE
SELECT 
    name,
    price,
    category
FROM products
WHERE price > (
    SELECT AVG(price) FROM products
)
ORDER BY price DESC;

-- Query 17: Subquery with IN
SELECT 
    username,
    email
FROM users
WHERE user_id IN (
    SELECT user_id 
    FROM orders 
    WHERE total_amount > 1000
);

-- Query 18: Correlated subquery
SELECT 
    p.name,
    p.category,
    p.price,
    (SELECT AVG(price) 
     FROM products p2 
     WHERE p2.category = p.category) AS category_avg_price
FROM products p
ORDER BY p.category, p.price DESC;

-- Query 19: Subquery in FROM (derived table)
SELECT 
    category,
    avg_price,
    product_count
FROM (
    SELECT 
        category,
        AVG(price) AS avg_price,
        COUNT(*) AS product_count
    FROM products
    GROUP BY category
) AS category_stats
WHERE product_count > 2
ORDER BY avg_price DESC;

-- ============================================
-- SECTION 5: CTEs (Common Table Expressions)
-- ============================================

-- Query 20: Basic CTE
WITH expensive_products AS (
    SELECT 
        product_id,
        name,
        price,
        category
    FROM products
    WHERE price > 500
)
SELECT 
    category,
    COUNT(*) AS expensive_count,
    AVG(price) AS avg_expensive_price
FROM expensive_products
GROUP BY category;

-- Query 21: Multiple CTEs
WITH 
high_value_customers AS (
    SELECT 
        user_id,
        username,
        SUM(total_amount) AS lifetime_value
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.username
    HAVING SUM(total_amount) > 1000
),
recent_orders AS (
    SELECT 
        user_id,
        COUNT(*) AS recent_order_count
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT 
    hvc.username,
    hvc.lifetime_value,
    COALESCE(ro.recent_order_count, 0) AS recent_orders
FROM high_value_customers hvc
LEFT JOIN recent_orders ro ON hvc.user_id = ro.user_id
ORDER BY hvc.lifetime_value DESC;

-- Query 22: CTE with INSERT/UPDATE
WITH updated_prices AS (
    UPDATE products
    SET price = price * 1.05  -- 5% increase
    WHERE category = 'Accessories'
    RETURNING product_id, name, price
)
SELECT 
    'Price Update Summary' AS report,
    COUNT(*) AS products_updated,
    AVG(price) AS new_avg_price
FROM updated_prices;

-- ============================================
-- SECTION 6: RECURSIVE CTEs
-- ============================================

-- Query 23: Category hierarchy (top-down)
WITH RECURSIVE category_tree AS (
    -- Base case: root categories
    SELECT 
        category_id,
        category_name,
        parent_id,
        1 AS level,
        category_name::TEXT AS path
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- Recursive case: get children
    SELECT 
        c.category_id,
        c.category_name,
        c.parent_id,
        ct.level + 1,
        ct.path || ' > ' || c.category_name
    FROM categories c
    INNER JOIN category_tree ct ON c.parent_id = ct.category_id
)
SELECT 
    REPEAT('  ', level - 1) || category_name AS hierarchy,
    level,
    path
FROM category_tree
ORDER BY path;

-- Query 24: Find all parent categories (bottom-up)
WITH RECURSIVE category_path AS (
    -- Start with Laptops
    SELECT 
        category_id,
        category_name,
        parent_id,
        1 AS level
    FROM categories
    WHERE category_name = 'Laptops'
    
    UNION ALL
    
    -- Get parents
    SELECT 
        c.category_id,
        c.category_name,
        c.parent_id,
        cp.level + 1
    FROM categories c
    INNER JOIN category_path cp ON c.category_id = cp.parent_id
)
SELECT 
    category_name,
    level
FROM category_path
ORDER BY level DESC;

-- Query 25: Generate date series for reporting
WITH RECURSIVE date_series AS (
    SELECT DATE '2024-01-01' AS report_date
    UNION ALL
    SELECT report_date + INTERVAL '1 day'
    FROM date_series
    WHERE report_date < DATE '2024-02-29'
)
SELECT 
    ds.report_date,
    COALESCE(COUNT(o.order_id), 0) AS orders_count,
    COALESCE(SUM(o.total_amount), 0)::DECIMAL(10,2) AS daily_revenue
FROM date_series ds
LEFT JOIN orders o ON DATE(o.order_date) = ds.report_date
GROUP BY ds.report_date
ORDER BY ds.report_date
LIMIT 30;

-- ============================================
-- SECTION 7: WINDOW FUNCTIONS
-- ============================================

-- Query 26: ROW_NUMBER - Rank products by price in each category
SELECT 
    category,
    name,
    price,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS price_rank
FROM products
ORDER BY category, price_rank;

-- Query 27: RANK vs DENSE_RANK
SELECT 
    name,
    category,
    price,
    RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS dense_rank
FROM products
ORDER BY category, rank;

-- Query 28: LAG and LEAD - Compare with previous/next
SELECT 
    order_id,
    order_date,
    total_amount,
    LAG(total_amount) OVER (ORDER BY order_date) AS previous_order_amount,
    LEAD(total_amount) OVER (ORDER BY order_date) AS next_order_amount,
    total_amount - LAG(total_amount) OVER (ORDER BY order_date) AS change_from_previous
FROM orders
ORDER BY order_date;

-- Query 29: Running total
SELECT 
    order_date,
    order_id,
    total_amount,
    SUM(total_amount) OVER (ORDER BY order_date) AS running_total,
    AVG(total_amount) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3
FROM orders
ORDER BY order_date;

-- Query 30: FIRST_VALUE and LAST_VALUE
SELECT 
    name,
    category,
    price,
    FIRST_VALUE(name) OVER (PARTITION BY category ORDER BY price) AS cheapest_product,
    FIRST_VALUE(price) OVER (PARTITION BY category ORDER BY price) AS cheapest_price,
    LAST_VALUE(name) OVER (
        PARTITION BY category 
        ORDER BY price
        RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS most_expensive_product,
    LAST_VALUE(price) OVER (
        PARTITION BY category 
        ORDER BY price
        RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS most_expensive_price
FROM products
ORDER BY category, price;

-- Query 31: NTILE - Divide into quartiles
SELECT 
    name,
    price,
    NTILE(4) OVER (ORDER BY price) AS price_quartile,
    CASE NTILE(4) OVER (ORDER BY price)
        WHEN 1 THEN 'Budget'
        WHEN 2 THEN 'Mid-Range'
        WHEN 3 THEN 'Premium'
        WHEN 4 THEN 'Luxury'
    END AS price_tier
FROM products
ORDER BY price;

-- Query 32: Aggregate window functions
SELECT 
    name,
    category,
    price,
    AVG(price) OVER (PARTITION BY category) AS category_avg,
    price - AVG(price) OVER (PARTITION BY category) AS diff_from_avg,
    COUNT(*) OVER (PARTITION BY category) AS category_product_count,
    SUM(stock_quantity) OVER (PARTITION BY category) AS category_total_stock
FROM products
ORDER BY category, price DESC;

-- ============================================
-- SECTION 8: ADVANCED ANALYTICS
-- ============================================

-- Query 33: Customer RFM Analysis
WITH customer_metrics AS (
    SELECT 
        u.user_id,
        u.username,
        MAX(o.order_date) AS last_order_date,
        COUNT(o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.username
)
SELECT 
    username,
    last_order_date,
    CURRENT_DATE - DATE(last_order_date) AS days_since_last_order,
    frequency AS order_count,
    monetary AS total_spent,
    NTILE(5) OVER (ORDER BY last_order_date DESC) AS recency_score,
    NTILE(5) OVER (ORDER BY frequency) AS frequency_score,
    NTILE(5) OVER (ORDER BY monetary) AS monetary_score,
    (NTILE(5) OVER (ORDER BY last_order_date DESC) +
     NTILE(5) OVER (ORDER BY frequency) +
     NTILE(5) OVER (ORDER BY monetary)) AS rfm_total_score
FROM customer_metrics
ORDER BY rfm_total_score DESC;

-- Query 34: Product performance analysis
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    COUNT(DISTINCT oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.subtotal) AS total_revenue,
    RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.subtotal) DESC) AS revenue_rank_in_category,
    ROUND(
        100.0 * SUM(oi.subtotal) / SUM(SUM(oi.subtotal)) OVER (PARTITION BY p.category),
        2
    ) AS pct_of_category_revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name, p.category, p.price
ORDER BY total_revenue DESC NULLS LAST;

-- Query 35: Customer segmentation
SELECT 
    u.username,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent,
    COALESCE(AVG(o.total_amount), 0)::DECIMAL(10,2) AS avg_order_value,
    CASE 
        WHEN SUM(o.total_amount) > 3000 THEN 'VIP'
        WHEN SUM(o.total_amount) > 1500 THEN 'Gold'
        WHEN SUM(o.total_amount) > 500 THEN 'Silver'
        WHEN SUM(o.total_amount) IS NOT NULL THEN 'Bronze'
        ELSE 'No Orders'
    END AS customer_tier
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.username
ORDER BY total_spent DESC;

-- Query 36: Revenue by category and month
SELECT 
    p.category,
    TO_CHAR(o.order_date, 'YYYY-MM') AS order_month,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.subtotal) AS revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status != 'cancelled'
GROUP BY p.category, TO_CHAR(o.order_date, 'YYYY-MM')
ORDER BY order_month, revenue DESC;

-- ============================================
-- SECTION 9: DATA MODIFICATION WITH RETURNING
-- ============================================

-- Query 37: Update with RETURNING
UPDATE products
SET stock_quantity = stock_quantity + 50
WHERE category = 'Accessories'
RETURNING product_id, name, stock_quantity;

-- Query 38: Delete with RETURNING
-- (Commented out to preserve data)
-- DELETE FROM order_items
-- WHERE quantity > 5
-- RETURNING order_item_id, order_id, quantity;

-- ============================================
-- SECTION 10: COMPLEX BUSINESS QUERIES
-- ============================================

-- Query 39: Find products that were never ordered
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.price,
    p.stock_quantity
FROM products p
WHERE p.product_id NOT IN (
    SELECT DISTINCT product_id 
    FROM order_items
)
AND p.is_active = TRUE
ORDER BY p.category, p.price DESC;

-- Query 40: Top 3 products by revenue in each category
WITH product_revenue AS (
    SELECT 
        p.product_id,
        p.name,
        p.category,
        SUM(oi.subtotal) AS total_revenue,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.subtotal) DESC) AS rank
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.name, p.category
)
SELECT 
    category,
    name,
    total_revenue,
    rank
FROM product_revenue
WHERE rank <= 3
ORDER BY category, rank;

-- ============================================
-- SUCCESS MESSAGE
-- ============================================

SELECT 'All 40 complex queries executed successfully!' AS status;
SELECT 'SQL concepts covered: DQL, Aggregates, JOINs, Subqueries, CTEs, Recursive CTEs, Window Functions, Analytics' AS summary;