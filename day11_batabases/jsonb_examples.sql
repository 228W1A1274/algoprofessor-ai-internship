-- ============================================
-- BASIC JSONB - PostgreSQL (BEGINNER)
-- ============================================

-- JSONB allows storing JSON data in database
-- Useful when structure is flexible

-- ============================================
-- PART 1: CREATE TABLE
-- ============================================

CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_name VARCHAR(50),
    preferences JSONB
);

-- ============================================
-- PART 2: INSERT JSON DATA
-- ============================================

INSERT INTO user_preferences (user_name, preferences)
VALUES
('Shanmukha', '{"theme": "dark", "language": "en"}'),

('Ravi', '{"theme": "light", "language": "te"}'),

('Priya', '{"theme": "dark", "language": "hi"}');

-- ============================================
-- PART 3: VIEW FULL JSON DATA
-- ============================================

SELECT * FROM user_preferences;

-- Output example:
-- id | user_name | preferences
-- 1  | Shanmukha | {"theme":"dark","language":"en"}

-- ============================================
-- PART 4: GET VALUE FROM JSON
-- ============================================

-- ->> gets value as TEXT

SELECT 
    user_name,
    preferences->>'theme' AS theme
FROM user_preferences;

-- Output:
-- Shanmukha | dark
-- Ravi      | light

-- ============================================
-- PART 5: GET MULTIPLE VALUES
-- ============================================

SELECT
    user_name,
    preferences->>'theme' AS theme,
    preferences->>'language' AS language
FROM user_preferences;

-- ============================================
-- PART 6: FILTER USING JSON VALUE
-- ============================================

-- Find users with dark theme

SELECT *
FROM user_preferences
WHERE preferences->>'theme' = 'dark';

-- ============================================
-- PART 7: ADD NEW VALUE TO JSON
-- ============================================

UPDATE user_preferences
SET preferences = preferences || '{"notifications": "enabled"}'
WHERE user_name = 'Shanmukha';

-- Check result
SELECT * FROM user_preferences;

-- ============================================
-- PART 8: UPDATE JSON VALUE
-- ============================================

UPDATE user_preferences
SET preferences = jsonb_set(
    preferences,
    '{theme}',
    '"light"'
)
WHERE user_name = 'Shanmukha';

-- ============================================
-- PART 9: REMOVE VALUE FROM JSON
-- ============================================

UPDATE user_preferences
SET preferences = preferences - 'notifications'
WHERE user_name = 'Shanmukha';

-- ============================================
-- PART 10: CREATE INDEX ON JSON FIELD
-- ============================================

CREATE INDEX idx_theme
ON user_preferences ((preferences->>'theme'));

-- ============================================
-- PART 11: SEARCH USING INDEX
-- ============================================

SELECT *
FROM user_preferences
WHERE preferences->>'theme' = 'dark';

-- ============================================
-- SUMMARY
-- ============================================

SELECT 'Basic JSONB learning completed!' AS message;
