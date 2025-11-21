-- Migration 003: Add email logs table
-- Feature #3: Email Notifications
-- Created: 2025-01-21

-- Create email_logs table for audit trail and debugging
CREATE TABLE IF NOT EXISTS email_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    recipient_email VARCHAR(255) NOT NULL,

    event_type VARCHAR(50) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    template_name VARCHAR(100) NOT NULL,

    -- Metadata
    event_data JSONB,

    -- Sending details
    provider VARCHAR(50),
    message_id VARCHAR(255),

    status VARCHAR(20) NOT NULL,
    error_message TEXT,

    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    opened_at TIMESTAMP,
    clicked_at TIMESTAMP,

    retry_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_email_logs_user ON email_logs(user_id);
CREATE INDEX idx_email_logs_status ON email_logs(status);
CREATE INDEX idx_email_logs_sent_at ON email_logs(sent_at DESC);
CREATE INDEX idx_email_logs_event ON email_logs(event_type);

-- Add comment
COMMENT ON TABLE email_logs IS 'Email notification logs for audit trail and debugging';
