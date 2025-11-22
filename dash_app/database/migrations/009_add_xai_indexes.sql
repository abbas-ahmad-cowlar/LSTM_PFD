-- Migration: Add indexes for XAI explanations table
-- Description: Performance indexes for explanation caching and retrieval
-- Date: 2024-11-22

-- Add composite index for cache lookups
CREATE INDEX IF NOT EXISTS idx_explanations_exp_signal_method
ON explanations(experiment_id, signal_id, method);

-- Add index for recent explanations queries
CREATE INDEX IF NOT EXISTS idx_explanations_created_at_desc
ON explanations(created_at DESC);

-- Add index for experiment-based filtering
CREATE INDEX IF NOT EXISTS idx_explanations_experiment_id
ON explanations(experiment_id);

-- Add index for method-based statistics
CREATE INDEX IF NOT EXISTS idx_explanations_method
ON explanations(method);

-- Add partial index for recent explanations (last 30 days)
CREATE INDEX IF NOT EXISTS idx_explanations_recent
ON explanations(created_at)
WHERE created_at > CURRENT_DATE - INTERVAL '30 days';
