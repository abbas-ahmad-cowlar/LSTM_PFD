-- Migration: 010_add_2fa_sessions.sql
-- Feature #8: 2FA & Sessions
-- Description: Add 2FA, session tracking, login history, and backup codes
-- Date: 2025-01-22

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Add 2FA columns to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret VARCHAR(32);
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_enabled BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN users.totp_secret IS 'Base32-encoded TOTP secret for 2FA';
COMMENT ON COLUMN users.totp_enabled IS 'Whether 2FA is enabled for this user';

-- Create index for fast 2FA lookups
CREATE INDEX IF NOT EXISTS idx_users_totp_enabled ON users(totp_enabled) WHERE totp_enabled = TRUE;

-- ============================================================================
-- Session Tracking
-- ============================================================================

-- Create session_logs table
CREATE TABLE IF NOT EXISTS session_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    ip_address VARCHAR(45),                    -- Supports IPv4 and IPv6
    user_agent VARCHAR(500),
    device_type VARCHAR(50),                   -- desktop, mobile, tablet
    browser VARCHAR(100),
    location VARCHAR(200),                      -- City, Country
    last_active TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    logged_out_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_session_logs_user_id ON session_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_session_logs_session_token ON session_logs(session_token);
CREATE INDEX IF NOT EXISTS idx_session_logs_is_active ON session_logs(is_active) WHERE is_active = TRUE;
-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_session_logs_user_active ON session_logs(user_id, is_active) WHERE is_active = TRUE;

COMMENT ON TABLE session_logs IS 'User session tracking for security monitoring';
COMMENT ON COLUMN session_logs.session_token IS 'Cryptographically secure session identifier';
COMMENT ON COLUMN session_logs.is_active IS 'Whether session is currently active';

-- ============================================================================
-- Login History
-- ============================================================================

-- Create login_history table
CREATE TABLE IF NOT EXISTS login_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address VARCHAR(45),                    -- Supports IPv4 and IPv6
    user_agent VARCHAR(500),
    location VARCHAR(200),                      -- City, Country
    login_method VARCHAR(50),                   -- password, 2fa, oauth, api_key
    success BOOLEAN DEFAULT TRUE,
    failure_reason VARCHAR(200),
    timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_login_history_user_id ON login_history(user_id);
CREATE INDEX IF NOT EXISTS idx_login_history_timestamp ON login_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_login_history_success ON login_history(success);
-- Composite index for analytics queries
CREATE INDEX IF NOT EXISTS idx_login_history_user_success_time ON login_history(user_id, success, timestamp DESC);

COMMENT ON TABLE login_history IS 'Complete audit trail of all login attempts';
COMMENT ON COLUMN login_history.success IS 'Whether login attempt was successful';
COMMENT ON COLUMN login_history.failure_reason IS 'Reason for failed login (if applicable)';

-- ============================================================================
-- Backup Codes
-- ============================================================================

-- Create backup_codes table
CREATE TABLE IF NOT EXISTS backup_codes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code_hash VARCHAR(255) NOT NULL UNIQUE,    -- bcrypt hash of backup code
    is_used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMP,
    ip_address VARCHAR(45),                     -- IP where code was used
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_backup_codes_user_id ON backup_codes(user_id);
CREATE INDEX IF NOT EXISTS idx_backup_codes_is_used ON backup_codes(is_used);
-- Composite index for finding unused codes
CREATE INDEX IF NOT EXISTS idx_backup_codes_user_unused ON backup_codes(user_id, is_used) WHERE is_used = FALSE;

COMMENT ON TABLE backup_codes IS '2FA backup codes for account recovery';
COMMENT ON COLUMN backup_codes.code_hash IS 'bcrypt hash of backup code (never store plaintext)';
COMMENT ON COLUMN backup_codes.is_used IS 'Whether this code has been used (one-time use only)';

-- ============================================================================
-- Triggers for updated_at
-- ============================================================================

-- Create or replace function for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for all new tables
DROP TRIGGER IF EXISTS update_session_logs_updated_at ON session_logs;
CREATE TRIGGER update_session_logs_updated_at
    BEFORE UPDATE ON session_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_login_history_updated_at ON login_history;
CREATE TRIGGER update_login_history_updated_at
    BEFORE UPDATE ON login_history
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_backup_codes_updated_at ON backup_codes;
CREATE TRIGGER update_backup_codes_updated_at
    BEFORE UPDATE ON backup_codes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Constraints and Validation
-- ============================================================================

-- Add CHECK constraints for data validation
ALTER TABLE session_logs
    ADD CONSTRAINT chk_session_token_length
    CHECK (LENGTH(session_token) >= 16);

ALTER TABLE login_history
    ADD CONSTRAINT chk_login_method
    CHECK (login_method IN ('password', '2fa', 'oauth', 'api_key'));

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Create a sample session for testing (optional, can be removed)
-- DO $$
-- BEGIN
--     IF EXISTS (SELECT 1 FROM users LIMIT 1) THEN
--         INSERT INTO session_logs (user_id, session_token, ip_address, device_type, browser, location)
--         SELECT
--             id,
--             encode(gen_random_bytes(32), 'base64'),
--             '127.0.0.1',
--             'desktop',
--             'Chrome 120',
--             'Local'
--         FROM users LIMIT 1;
--     END IF;
-- END $$;

-- ============================================================================
-- DOWN MIGRATION (ROLLBACK)
-- ============================================================================

-- To rollback this migration, run:
/*
-- Drop triggers
DROP TRIGGER IF EXISTS update_session_logs_updated_at ON session_logs;
DROP TRIGGER IF EXISTS update_login_history_updated_at ON login_history;
DROP TRIGGER IF EXISTS update_backup_codes_updated_at ON backup_codes;

-- Drop tables
DROP TABLE IF EXISTS backup_codes CASCADE;
DROP TABLE IF EXISTS login_history CASCADE;
DROP TABLE IF EXISTS session_logs CASCADE;

-- Remove columns from users table
ALTER TABLE users DROP COLUMN IF EXISTS totp_secret;
ALTER TABLE users DROP COLUMN IF EXISTS totp_enabled;

-- Drop indexes (done automatically with tables, but listed for reference)
-- DROP INDEX IF EXISTS idx_users_totp_enabled;
*/

-- ============================================================================
-- MIGRATION VERIFICATION
-- ============================================================================

-- Verify all tables and columns were created successfully
DO $$
BEGIN
    -- Check users table columns
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'totp_secret'
    ) THEN
        RAISE NOTICE 'Migration successful: totp_secret column added to users';
    ELSE
        RAISE EXCEPTION 'Migration failed: totp_secret column not added';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'totp_enabled'
    ) THEN
        RAISE NOTICE 'Migration successful: totp_enabled column added to users';
    ELSE
        RAISE EXCEPTION 'Migration failed: totp_enabled column not added';
    END IF;

    -- Check session_logs table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'session_logs') THEN
        RAISE NOTICE 'Migration successful: session_logs table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: session_logs table not created';
    END IF;

    -- Check login_history table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'login_history') THEN
        RAISE NOTICE 'Migration successful: login_history table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: login_history table not created';
    END IF;

    -- Check backup_codes table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'backup_codes') THEN
        RAISE NOTICE 'Migration successful: backup_codes table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: backup_codes table not created';
    END IF;

    -- Count indexes
    RAISE NOTICE 'Migration verification complete. All tables and columns created successfully.';
END $$;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Display summary
DO $$
DECLARE
    session_count INTEGER;
    history_count INTEGER;
    backup_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO session_count FROM session_logs;
    SELECT COUNT(*) INTO history_count FROM login_history;
    SELECT COUNT(*) INTO backup_count FROM backup_codes;

    RAISE NOTICE '========================================';
    RAISE NOTICE 'Migration 010: 2FA & Sessions Complete';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Session logs: % records', session_count;
    RAISE NOTICE 'Login history: % records', history_count;
    RAISE NOTICE 'Backup codes: % records', backup_count;
    RAISE NOTICE '========================================';
END $$;
