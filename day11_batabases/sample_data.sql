-- ============================================
-- SAMPLE DATA FOR TESTING
-- ============================================

-- Insert Categories (hierarchical structure)
INSERT INTO categories (category_name, parent_id) VALUES
('Electronics', NULL),           -- 1
('Computers', 1),                -- 2
('Laptops', 2),                  -- 3
('Desktops', 2),                 -- 4
('Accessories', 1),              -- 5
('Furniture', NULL),             -- 6
('Office', 6),                   -- 7
('Home', 6);                     -- 8

-- Insert Users
INSERT INTO users (username, email, phone_number, age) VALUES
('john_doe', 'john@example.com', '555-0101', 25),
('jane_smith', 'jane@example.com', '555-0102', 30),
('bob_jones', 'bob@example.com', '555-0103', 28),
('alice_wong', 'alice@example.com', '555-0104', 35),
('charlie_brown', 'charlie@example.com', '555-0105', 22),
('david_lee', 'david@example.com', '555-0106', 29),
('emma_wilson', 'emma@example.com', '555-0107', 31),
('frank_miller', 'frank@example.com', '555-0108', 27);

-- Insert Products
INSERT INTO products (name, description, price, stock_quantity, category, category_id) VALUES
-- Electronics > Computers > Laptops
('Laptop Pro 15', 'High-performance laptop with 16GB RAM', 1299.99, 50, 'Electronics', 3),
('Laptop Air 13', 'Lightweight laptop for travel', 999.99, 75, 'Electronics', 3),
('Gaming Laptop', 'RGB gaming laptop with RTX 4070', 1899.99, 30, 'Electronics', 3),

-- Electronics > Computers > Desktops
('Desktop Workstation', 'Professional workstation', 1599.99, 25, 'Electronics', 4),
('Gaming Desktop', 'High-end gaming PC', 2299.99, 15, 'Electronics', 4),

-- Electronics > Accessories
('Wireless Mouse', 'Ergonomic wireless mouse', 29.99, 200, 'Accessories', 5),
('Mechanical Keyboard', 'RGB mechanical keyboard', 149.99, 100, 'Accessories', 5),
('USB-C Cable', '2m USB-C charging cable', 19.99, 500, 'Accessories', 5),
('Webcam HD', '1080p webcam', 79.99, 120, 'Electronics', 5),
('Headphones', 'Noise-canceling headphones', 249.99, 80, 'Electronics', 5),

-- Electronics (general)
('Monitor 27"', '4K monitor 27 inch', 399.99, 60, 'Electronics', 1),
('Monitor 32"', '4K monitor 32 inch', 599.99, 40, 'Electronics', 1),

-- Furniture > Office
('Office Chair', 'Ergonomic office chair', 299.99, 75, 'Furniture', 7),
('Standing Desk', 'Adjustable standing desk', 599.99, 30, 'Furniture', 7),
('Desk Lamp', 'LED desk lamp', 49.99, 150, 'Furniture', 7),

-- Furniture > Home
('Bookshelf', 'Wooden bookshelf', 199.99, 50, 'Furniture', 8),
('Coffee Table', 'Modern coffee table', 249.99, 40, 'Furniture', 8);

-- Insert Orders
INSERT INTO orders (user_id, order_date, total_amount, status, shipping_address) VALUES
(1, '2024-01-15 10:30:00', 1349.98, 'delivered', '123 Main St, City, State 12345'),
(2, '2024-01-20 14:45:00', 329.98, 'delivered', '456 Oak Ave, City, State 12346'),
(1, '2024-02-05 09:15:00', 449.98, 'shipped', '123 Main St, City, State 12345'),
(3, '2024-02-10 16:20:00', 679.98, 'processing', '789 Pine Rd, City, State 12347'),
(4, '2024-02-12 11:00:00', 1899.96, 'delivered', '321 Elm St, City, State 12348'),
(2, '2024-02-14 13:30:00', 999.99, 'shipped', '456 Oak Ave, City, State 12346'),
(5, '2024-02-15 16:45:00', 1549.98, 'processing', '654 Maple Dr, City, State 12349'),
(6, '2024-02-16 10:00:00', 749.97, 'pending', '987 Pine St, City, State 12350');

-- Insert Order Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Order 1 (John - delivered)
(1, 1, 1, 1299.99),  -- Laptop Pro
(1, 6, 1, 29.99),    -- Mouse
(1, 8, 1, 19.99),    -- Cable

-- Order 2 (Jane - delivered)
(2, 13, 1, 299.99),  -- Office Chair
(2, 15, 1, 29.99),   -- Desk Lamp

-- Order 3 (John - shipped)
(3, 11, 1, 399.99),  -- Monitor 27"
(3, 15, 1, 49.99),   -- Desk Lamp

-- Order 4 (Bob - processing)
(4, 14, 1, 599.99),  -- Standing Desk
(4, 9, 1, 79.99),    -- Webcam

-- Order 5 (Alice - delivered)
(5, 3, 1, 1899.99),  -- Gaming Laptop

-- Order 6 (Jane - shipped)
(6, 2, 1, 999.99),   -- Laptop Air

-- Order 7 (Charlie - processing)
(7, 1, 1, 1299.99),  -- Laptop Pro
(7, 7, 1, 149.99),   -- Keyboard
(7, 6, 1, 29.99),    -- Mouse
(7, 8, 3, 19.99),    -- Cables (x3)

-- Order 8 (David - pending)
(8, 10, 3, 249.99);  -- Headphones (x3)

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Count records in each table
SELECT 'categories' AS table_name, COUNT(*) AS record_count FROM categories
UNION ALL
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items;

-- Show sample data
SELECT 'Sample Users:' AS info;
SELECT user_id, username, email FROM users LIMIT 3;

SELECT 'Sample Products:' AS info;
SELECT product_id, name, price, category FROM products LIMIT 5;

SELECT 'Sample Orders:' AS info;
SELECT order_id, user_id, order_date, total_amount, status FROM orders LIMIT 3;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Sample data loaded successfully!';
    RAISE NOTICE '8 categories, 8 users, 17 products, 8 orders, 15 order items';
END $$;