-- Migration 005: Add webhook configurations table
-- Feature #4: Slack/Teams Webhook Integration
-- Created: 2025-01-21

-- Create webhook_configurations table for Slack/Teams/custom webhooks
CREATE TABLE IF NOT EXISTS webhook_configurations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Provider info
    provider_type VARCHAR(50) NOT NULL,  -- 'slack', 'teams', 'webhook'
    webhook_url TEXT NOT NULL,

    -- User-provided metadata
    name VARCHAR(200),
    description TEXT,

    -- Configuration
    is_active BOOLEAN DEFAULT TRUE,

    -- Event routing (JSON array of enabled events)
    enabled_events JSONB DEFAULT '[]'::jsonb NOT NULL,

    -- Provider-specific settings (flexible JSON)
    settings JSONB DEFAULT '{}'::jsonb NOT NULL,

    -- Status tracking
    last_used_at TIMESTAMP,
    last_error TEXT,
    consecutive_failures INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL,

    -- Constraints - prevent duplicate webhooks
    CONSTRAINT uq_user_webhook_url UNIQUE(user_id, webhook_url)
);

-- Create indexes
CREATE INDEX idx_webhook_configs_user ON webhook_configurations(user_id);
CREATE INDEX idx_webhook_configs_provider ON webhook_configurations(provider_type);
CREATE INDEX idx_webhook_configs_active ON webhook_configurations(is_active) WHERE is_active = TRUE;

-- Add comment
COMMENT ON TABLE webhook_configurations IS 'Webhook configurations for Slack, Teams, and custom webhook integrations (Feature #4)';
