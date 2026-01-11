-- Migration 006: Add webhook logs table
-- Feature #4: Slack/Teams Webhook Integration
-- Created: 2025-01-21

-- Create webhook_logs table for tracking webhook delivery
CREATE TABLE IF NOT EXISTS webhook_logs (
    id SERIAL PRIMARY KEY,
    webhook_config_id INTEGER NOT NULL REFERENCES webhook_configurations(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,

    event_type VARCHAR(50) NOT NULL,
    provider_type VARCHAR(50) NOT NULL,

    -- Request details
    webhook_url TEXT NOT NULL,
    payload JSONB,

    -- Response details
    status VARCHAR(20) NOT NULL,  -- 'sent', 'failed', 'rate_limited', 'timeout'
    http_status_code INTEGER,
    response_body TEXT,
    error_message TEXT,

    retry_count INTEGER DEFAULT 0,
    sent_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

-- Create indexes
CREATE INDEX idx_webhook_logs_config ON webhook_logs(webhook_config_id);
CREATE INDEX idx_webhook_logs_user ON webhook_logs(user_id);
CREATE INDEX idx_webhook_logs_status ON webhook_logs(status);
CREATE INDEX idx_webhook_logs_created ON webhook_logs(created_at DESC);

-- Add comment
COMMENT ON TABLE webhook_logs IS 'Webhook delivery logs for audit trail and debugging (Feature #4)';
