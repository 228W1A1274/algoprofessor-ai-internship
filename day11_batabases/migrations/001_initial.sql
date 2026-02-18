-- ============================================
-- MIGRATION 001: Initial Schema
-- Created: 2026-02-16
-- ============================================

-- This file contains the initial database schema
-- It's a copy of db_schema.sql for migration tracking

\echo 'Running Migration 001: Initial Schema...'

-- Execute the initial schema
\i '../db_schema.sql'

\echo 'Migration 001 completed successfully!'