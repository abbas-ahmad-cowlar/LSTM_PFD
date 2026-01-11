-- Migration 002: Add notification preferences table
-- Feature #3: Email Notifications
-- Created: 2025-01-21

-- Create notification_preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,

    -- Channel preferences (true = enabled)
    email_enabled BOOLEAN DEFAULT TRUE,
    in_app_enabled BOOLEAN DEFAULT TRUE,
    slack_enabled BOOLEAN DEFAULT FALSE,
    webhook_enabled BOOLEAN DEFAULT FALSE,

    -- Email-specific settings
    email_frequency VARCHAR(20) DEFAULT 'immediate',

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    UNIQUE(user_id, event_type)
);

-- Create indexes
CREATE INDEX idx_notif_prefs_user ON notification_preferences(user_id);
CREATE INDEX idx_notif_prefs_event ON notification_preferences(event_type);

-- Add comment
COMMENT ON TABLE notification_preferences IS 'User notification preferences for different event types';
