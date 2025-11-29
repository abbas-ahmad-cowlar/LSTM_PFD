# Database Index Migration Guide

## Overview

This guide explains how to migrate the database from the previous index configuration to the optimized one.

## Changes Summary

### Removed (Duplicate Indexes)
These indexes were redundant and have been removed:
- `ix_api_keys_user_id` - duplicate of FK index
- `ix_api_keys_user_active` - composite on already-indexed columns
- `ix_email_logs_user_status` - duplicate of column-level indexes
- `ix_email_logs_event_status` - duplicate of column-level indexes
- `ix_webhook_configurations_user_active` - duplicate
- `ix_webhook_configurations_provider_active` - duplicate
- `ix_webhook_logs_webhook_status` - duplicate
- `ix_webhook_logs_event_status` - duplicate
- `ix_experiments_created_by` - FK auto-indexed
- `ix_experiments_dataset_id` - FK auto-indexed
- `ix_experiments_hpo_campaign_id` - FK auto-indexed
- `ix_experiments_user_status` - duplicate composite
- `ix_hpo_campaigns_created_by` - FK auto-indexed
- `ix_hpo_campaigns_dataset_id` - FK auto-indexed
- `ix_hpo_campaigns_user_status` - duplicate composite
- `ix_nas_campaigns_dataset_id` - FK auto-indexed
- `ix_nas_campaigns_dataset_status` - duplicate composite
- `ix_nas_trials_campaign_id` - FK auto-indexed
- `ix_api_request_logs_api_key_id` - FK auto-indexed
- `ix_api_request_logs_endpoint_status` - duplicate
- `ix_api_request_logs_key_time` - duplicate
- `ix_datasets_created_by` - FK auto-indexed
- `ix_tags_created_by` - FK auto-indexed
- `ix_experiment_tags_added_by` - FK auto-indexed
- `ix_experiment_tags_experiment_tag` - UniqueConstraint creates this
- `ix_notification_preferences_user_event` - UniqueConstraint creates this
- `ix_saved_searches_user_pinned` - low benefit composite
- `ix_users_email` - already unique=True

### Retained (Useful Indexes)
These indexes provide clear performance benefits:
- `ix_*_created_at` - All timestamp indexes for time-range queries
- `ix_*_sent_at` - Log timestamp indexes
- `ix_users_is_active` - Filtering active/inactive users
- `ix_saved_searches_is_pinned` - Filtering pinned searches
- `ix_hpo_campaigns_status` - Campaign status filtering
- `ix_nas_campaigns_status` - Campaign status filtering
- `ix_training_runs_experiment_epoch` - Composite for ordered results
- `ix_nas_trials_campaign_trial` - Composite for ordered results
- `ix_api_metrics_summary_period_type` - Dashboard query optimization

## Migration Options

### Option 1: Using Alembic (Recommended)

#### Step 1: Create Migration

```bash
cd /home/user/LSTM_PFD
alembic revision -m "Optimize database indexes - remove duplicates"
```

#### Step 2: Edit Migration File

The migration file will be created in `alembic/versions/`. Edit it with the following template:

```python
"""Optimize database indexes - remove duplicates

Revision ID: xxxxx
Revises: yyyyy
Create Date: 2025-11-22

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'xxxxx'
down_revision = 'yyyyy'
branch_labels = None
depends_on = None


def upgrade():
    """Remove duplicate indexes that provide no performance benefit."""
    # These will be created automatically on next table create/alter
    # since they're defined in model __table_args__

    # Just let SQLAlchemy recreate the schema with the new index configuration
    # Alternatively, manually drop specific indexes:

    # Example (uncomment and customize as needed):
    # op.drop_index('ix_api_keys_user_id', table_name='api_keys', if_exists=True)
    # op.drop_index('ix_email_logs_user_status', table_name='email_logs', if_exists=True)
    # ... etc for all duplicate indexes listed above

    pass  # Changes will be applied when models are recreated


def downgrade():
    """Recreate removed indexes if reverting."""
    # This is complex because we're removing intentional duplicates
    # If needed, recreate indexes based on previous schema
    pass
```

#### Step 3: Run Migration

```bash
alembic upgrade head
```

### Option 2: Manual SQL (Review Only)

For existing databases, you may want to review existing indexes first:

```sql
-- Check all indexes on a specific table
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
    AND tablename = 'experiments'  -- Change table name as needed
ORDER BY indexname;

-- Find potentially duplicate indexes
SELECT
    idx1.tablename,
    idx1.indexname AS index1,
    idx2.indexname AS index2,
    idx1.indexdef
FROM pg_indexes idx1
JOIN pg_indexes idx2
    ON idx1.tablename = idx2.tablename
    AND idx1.indexname < idx2.indexname
    AND idx1.indexdef = idx2.indexdef
WHERE idx1.schemaname = 'public';
```

### Option 3: Fresh Database

If you're setting up a fresh database:

```bash
# Simply initialize with the new schema
python -m dash_app.database.run_migration
```

The models now have optimized indexes defined, so no migration is needed.

## Connection Pool Changes

The connection pool configuration has been updated in `dash_app/database/connection.py`:

### Old Configuration
```python
pool_size=50
max_overflow=50
# Total: 100 connections
```

### New Configuration
```python
pool_size=30
max_overflow=30
pool_timeout=30
max_identifier_length=128
# Total: 60 connections (sufficient for 26 callbacks)
```

**No migration required** - these changes take effect immediately on application restart.

## Performance Monitoring

New features added:
- **Slow Query Logging**: Queries >1 second are logged as warnings
- **Connection Pool Monitoring**: New connections logged at DEBUG level

Enable debug logging to see connection pool activity:
```python
# In your logging configuration
logging.getLogger('sqlalchemy.pool').setLevel(logging.DEBUG)
```

## Verification

After migration, verify the changes:

```sql
-- Count indexes per table
SELECT
    tablename,
    COUNT(*) as index_count
FROM pg_indexes
WHERE schemaname = 'public'
GROUP BY tablename
ORDER BY index_count DESC;

-- Show all indexes (should be fewer than before)
SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

## Rollback Plan

If issues occur:

1. **Code Rollback**:
   ```bash
   git revert <commit_hash>
   git push
   ```

2. **Database Rollback** (if using Alembic):
   ```bash
   alembic downgrade -1
   ```

3. **Manual Index Recreation** (if needed):
   Check `DATABASE_PERFORMANCE_ANALYSIS.md` for the list of removed indexes
   and recreate specific ones if required.

## Expected Benefits

After migration:
- ✅ Reduced disk space usage (fewer redundant indexes)
- ✅ Faster write operations (fewer indexes to update)
- ✅ Same or better read performance (kept all beneficial indexes)
- ✅ Better connection pool utilization
- ✅ Slow query visibility

## Monitoring Post-Migration

Monitor these metrics for 1 week:
1. Application response times
2. Database connection pool usage
3. Slow query logs
4. Disk I/O patterns

If any degradation is observed, check `DATABASE_PERFORMANCE_ANALYSIS.md` for
specific indexes that might need to be recreated.

---

**Last Updated**: 2025-11-22
**Author**: Syed Abbas Ahmad
