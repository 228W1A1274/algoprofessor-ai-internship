-- ============================================
-- VERIFICATION & TESTING QUERIES
-- ============================================

-- Check database connection
SELECT current_database() AS database_name,
       current_user AS connected_user,
       version() AS postgres_version;

-- List all tables
SELECT 
    table_name,
    table_type
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Count records in all tables
SELECT 'categories' AS table_name, COUNT(*) AS records FROM categories
UNION ALL
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items;

-- List all views
SELECT table_name AS view_name
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;

-- List all functions
SELECT 
    routine_name AS function_name,
    routine_type
FROM information_schema.routines
WHERE routine_schema = 'public'
ORDER BY routine_name;

-- List all triggers
SELECT 
    trigger_name,
    event_manipulation AS event,
    event_object_table AS table_name
FROM information_schema.triggers
WHERE trigger_schema = 'public'
ORDER BY event_object_table, trigger_name;

-- List all indexes
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- Check constraints
SELECT 
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
WHERE tc.table_schema = 'public'
ORDER BY tc.table_name, tc.constraint_type;

-- Test views
SELECT * FROM order_summary LIMIT 5;
SELECT * FROM inventory_status LIMIT 5;
SELECT * FROM customer_lifetime_value LIMIT 5;

-- Database summary
SELECT 
    'Database Setup Complete!' AS status,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE') AS total_tables,
    (SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public') AS total_views,
    (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public') AS total_indexes,
    (SELECT COUNT(*) FROM information_schema.triggers WHERE trigger_schema = 'public') AS total_triggers,
    (SELECT SUM(n_tup_ins) FROM pg_stat_user_tables) AS total_records_inserted;