# Database Indexes for Query Optimization

## Overview
This document lists the database indexes that should be created to support the query patterns used throughout the application, particularly after the N+1 query fixes and pagination improvements.

## Critical Indexes

### 1. Experiments Table

```sql
-- Primary key (already exists)
CREATE INDEX idx_experiments_pkey ON experiments(id);

-- Status + created_at for filtered listings
CREATE INDEX idx_experiments_status_created ON experiments(status, created_at DESC);

-- Model type filtering
CREATE INDEX idx_experiments_model_type ON experiments(model_type);

-- User's experiments
CREATE INDEX idx_experiments_user_id ON experiments(created_by);

-- HPO campaign experiments
CREATE INDEX idx_experiments_hpo_campaign ON experiments(hpo_campaign_id) WHERE hpo_campaign_id IS NOT NULL;

-- Completed experiments (common query)
CREATE INDEX idx_experiments_completed ON experiments(created_at DESC) WHERE status = 'completed';
```

### 2. ExperimentTag Table (Many-to-Many)

```sql
-- Composite index for finding tags by experiment (CRITICAL for bulk loading)
CREATE INDEX idx_experiment_tags_exp_id ON experiment_tags(experiment_id);

-- Composite index for finding experiments by tag
CREATE INDEX idx_experiment_tags_tag_id ON experiment_tags(tag_id);

-- Covering index for bulk tag queries (includes tag relationship)
CREATE INDEX idx_experiment_tags_covering ON experiment_tags(experiment_id, tag_id);

-- Unique constraint (likely already exists)
CREATE UNIQUE INDEX idx_experiment_tags_unique ON experiment_tags(experiment_id, tag_id);
```

### 3. Tags Table

```sql
-- Name lookup (unique index likely already exists)
CREATE UNIQUE INDEX idx_tags_name ON tags(name);

-- Slug lookup
CREATE UNIQUE INDEX idx_tags_slug ON tags(slug);

-- Popular tags (ordered by usage)
CREATE INDEX idx_tags_usage_count ON tags(usage_count DESC);
```

### 4. TrainingRun Table

```sql
-- Find training runs by experiment (CRITICAL for eager loading)
CREATE INDEX idx_training_runs_experiment ON training_runs(experiment_id, epoch);

-- Composite index for efficient ordering
CREATE INDEX idx_training_runs_exp_epoch ON training_runs(experiment_id, epoch ASC);
```

### 5. WebhookConfiguration Table

```sql
-- User's active webhooks
CREATE INDEX idx_webhooks_user_active ON webhook_configurations(user_id, is_active);

-- Created date for pagination
CREATE INDEX idx_webhooks_created ON webhook_configurations(created_at DESC);
```

### 6. WebhookLog Table

```sql
-- Find logs by webhook config
CREATE INDEX idx_webhook_logs_config ON webhook_logs(webhook_config_id, created_at DESC);

-- Find logs by user
CREATE INDEX idx_webhook_logs_user ON webhook_logs(user_id, created_at DESC);

-- Event type filtering
CREATE INDEX idx_webhook_logs_event ON webhook_logs(event_type);

-- Status filtering
CREATE INDEX idx_webhook_logs_status ON webhook_logs(status);
```

### 7. APIKey Table

```sql
-- User's API keys
CREATE INDEX idx_api_keys_user ON api_keys(user_id, created_at DESC);

-- Active keys filtering
CREATE INDEX idx_api_keys_active ON api_keys(user_id, is_active);

-- Prefix lookup for verification
CREATE INDEX idx_api_keys_prefix ON api_keys(prefix);
```

### 8. HPOCampaign Table

```sql
-- Campaign listing
CREATE INDEX idx_hpo_campaigns_created ON hpo_campaigns(created_at DESC);

-- Status filtering
CREATE INDEX idx_hpo_campaigns_status ON hpo_campaigns(status);
```

### 9. NASTrial Table

```sql
-- Find trials by campaign
CREATE INDEX idx_nas_trials_campaign ON nas_trials(campaign_id, trial_number ASC);

-- Status filtering within campaign
CREATE INDEX idx_nas_trials_campaign_status ON nas_trials(campaign_id, status);
```

### 10. Dataset Table

```sql
-- Dataset listing with pagination
CREATE INDEX idx_datasets_created ON datasets(created_at DESC);

-- User's datasets
CREATE INDEX idx_datasets_user ON datasets(created_by);
```

## Index Verification Queries

Use these queries to verify indexes exist:

```sql
-- PostgreSQL
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Performance Monitoring

### Query Performance Analysis

```sql
-- Find slow queries (PostgreSQL)
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
WHERE mean_time > 100  -- queries taking > 100ms on average
ORDER BY mean_time DESC
LIMIT 20;
```

### Index Bloat Check

```sql
-- Check for bloated indexes that need REINDEX
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Migration Script

Create a migration file to add all indexes:

```python
# migrations/add_performance_indexes.py
from alembic import op

def upgrade():
    # Experiments table indexes
    op.create_index('idx_experiments_status_created', 'experiments',
                    ['status', op.text('created_at DESC')])
    op.create_index('idx_experiments_model_type', 'experiments', ['model_type'])
    op.create_index('idx_experiments_user_id', 'experiments', ['created_by'])

    # ExperimentTag table indexes (CRITICAL!)
    op.create_index('idx_experiment_tags_exp_id', 'experiment_tags', ['experiment_id'])
    op.create_index('idx_experiment_tags_tag_id', 'experiment_tags', ['tag_id'])

    # Tags table indexes
    op.create_index('idx_tags_usage_count', 'tags', [op.text('usage_count DESC')])

    # TrainingRun table indexes (CRITICAL!)
    op.create_index('idx_training_runs_exp_epoch', 'training_runs',
                    ['experiment_id', 'epoch'])

    # Webhook indexes
    op.create_index('idx_webhooks_user_active', 'webhook_configurations',
                    ['user_id', 'is_active'])
    op.create_index('idx_webhook_logs_config', 'webhook_logs',
                    ['webhook_config_id', op.text('created_at DESC')])

    # API Key indexes
    op.create_index('idx_api_keys_user', 'api_keys',
                    ['user_id', op.text('created_at DESC')])
    op.create_index('idx_api_keys_prefix', 'api_keys', ['prefix'])

    # Additional indexes for pagination
    op.create_index('idx_hpo_campaigns_created', 'hpo_campaigns',
                    [op.text('created_at DESC')])
    op.create_index('idx_nas_trials_campaign', 'nas_trials',
                    ['campaign_id', 'trial_number'])
    op.create_index('idx_datasets_created', 'datasets',
                    [op.text('created_at DESC')])


def downgrade():
    # Drop indexes in reverse order
    op.drop_index('idx_datasets_created')
    op.drop_index('idx_nas_trials_campaign')
    op.drop_index('idx_hpo_campaigns_created')
    op.drop_index('idx_api_keys_prefix')
    op.drop_index('idx_api_keys_user')
    op.drop_index('idx_webhook_logs_config')
    op.drop_index('idx_webhooks_user_active')
    op.drop_index('idx_training_runs_exp_epoch')
    op.drop_index('idx_tags_usage_count')
    op.drop_index('idx_experiment_tags_tag_id')
    op.drop_index('idx_experiment_tags_exp_id')
    op.drop_index('idx_experiments_user_id')
    op.drop_index('idx_experiments_model_type')
    op.drop_index('idx_experiments_status_created')
```

## Best Practices

1. **Always index foreign keys** - They're used in JOINs and `.filter()` clauses
2. **Index commonly filtered columns** - `status`, `is_active`, `user_id`, etc.
3. **Composite indexes for sorting** - Include ORDER BY columns
4. **Partial indexes for common WHERE clauses** - e.g., WHERE status = 'completed'
5. **Monitor index usage** - Drop unused indexes to save space and write performance
6. **REINDEX regularly** - Prevent index bloat on heavily updated tables

## Impact on Write Performance

⚠️ **Important**: While indexes speed up reads, they slow down writes (INSERT/UPDATE/DELETE).

- Each additional index adds overhead to write operations
- For tables with heavy writes, balance read vs write performance
- Consider using partial indexes to reduce overhead
- Monitor write performance after adding indexes

## Estimated Performance Gains

| Query Type | Before | After | Improvement |
|------------|--------|-------|-------------|
| Load 500 experiments with tags | 501 queries | 2 queries | **99.6% reduction** |
| Load experiment with training runs | N+1 queries | 1 query | **~100 fewer queries** |
| Filter experiments by status | Full table scan | Index scan | **10-100x faster** |
| Tag autocomplete | Sequential scan | Index scan | **5-50x faster** |
| Webhook lookup by user | Sequential scan | Index scan | **10-100x faster** |

## Next Steps

1. Run the migration to add indexes
2. Analyze query plans with `EXPLAIN ANALYZE`
3. Monitor slow query log
4. Adjust indexes based on actual usage patterns
5. Set up automated index bloat monitoring
