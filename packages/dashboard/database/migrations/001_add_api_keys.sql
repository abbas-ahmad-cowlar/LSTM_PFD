-- Migration: 001_add_api_keys.sql
-- Feature #1: API Keys & Rate Limiting
-- Description: Add api_keys and api_usage tables for programmatic API access
-- Date: 2025-01-15

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,  -- bcrypt hash of the key
    name VARCHAR(100) NOT NULL,              -- User-provided name (e.g., "CI/CD Pipeline")
    prefix VARCHAR(20) NOT NULL,             -- First 20 chars for display (e.g., "sk_live_abc")
    scopes TEXT[] DEFAULT ARRAY['read', 'write'],  -- Permissions array
    rate_limit INTEGER DEFAULT 1000,         -- Requests per hour
    last_used_at TIMESTAMP,                  -- Last successful authentication
    is_active BOOLEAN DEFAULT TRUE,          -- Whether key is active (revoked keys are inactive)
    expires_at TIMESTAMP,                    -- NULL = never expires
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(prefix);  -- For fast lookup
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;

-- Create api_usage table for tracking API calls
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,          -- API endpoint called
    method VARCHAR(10) NOT NULL,             -- HTTP method (GET, POST, etc.)
    status_code INTEGER NOT NULL,            -- HTTP status code returned
    response_time_ms INTEGER,                -- Response time in milliseconds
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Create indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_api_usage_key_timestamp ON api_usage(api_key_id, timestamp DESC);
-- Partial index for recent data (last 30 days)
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp)
    WHERE timestamp > NOW() - INTERVAL '30 days';

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comment to tables
COMMENT ON TABLE api_keys IS 'API keys for programmatic access to the platform';
COMMENT ON TABLE api_usage IS 'API usage tracking for analytics and abuse detection';

-- ============================================================================
-- DOWN MIGRATION (ROLLBACK)
-- ============================================================================

-- To rollback this migration, run:
/*
DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP TABLE IF EXISTS api_usage CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
*/

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Verify tables were created
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_keys') THEN
        RAISE NOTICE 'Migration successful: api_keys table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: api_keys table not created';
    END IF;

    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_usage') THEN
        RAISE NOTICE 'Migration successful: api_usage table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: api_usage table not created';
    END IF;
END $$;
