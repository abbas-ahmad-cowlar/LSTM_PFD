-- Migration 004: Add email digest queue table
-- Feature #3: Email Notifications
-- Created: 2025-01-21

-- Create email_digest_queue table for daily/weekly digest emails
CREATE TABLE IF NOT EXISTS email_digest_queue (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    scheduled_for TIMESTAMP NOT NULL,
    included_in_digest BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_digest_queue_user ON email_digest_queue(user_id);
CREATE INDEX idx_digest_queue_scheduled ON email_digest_queue(scheduled_for);
CREATE INDEX idx_digest_queue_pending ON email_digest_queue(included_in_digest)
    WHERE included_in_digest = FALSE;

-- Add comment
COMMENT ON TABLE email_digest_queue IS 'Queue for batching notifications into digest emails';
