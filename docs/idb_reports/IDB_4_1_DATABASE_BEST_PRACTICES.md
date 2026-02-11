# IDB 4.1: Database Best Practices

**IDB ID:** 4.1  
**Domain:** Infrastructure  
**Curator Date:** 2026-01-23

---

## 1. Model Definition Conventions

### 1.1 Base Model Inheritance

```python
# Always extend BaseModel for consistent timestamps
class BaseModel(Base):
    """Abstract base model with common fields."""
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
```

### 1.2 Column Definition Standards

```python
class Experiment(BaseModel):
    __tablename__ = 'experiments'

    # ✅ String columns: Always specify max length
    name = Column(String(255), unique=True, nullable=False, index=True)

    # ✅ Enums: Use SQLAlchemy Enum with Python enum
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING, index=True)

    # ✅ Foreign keys: Specify ondelete behavior
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)

    # ✅ JSON: Use for flexible/nested data
    config = Column(JSON, nullable=False)

    # ✅ Database-agnostic variants
    scopes = Column(ARRAY(String).with_variant(JSON, 'sqlite'))
```

### 1.3 Index Strategy

```python
__table_args__ = (
    # ✅ Time-based indexes for common queries
    Index('ix_experiments_created_at', 'created_at'),

    # ✅ Composite indexes only for proven query patterns
    Index('ix_training_runs_experiment_epoch', 'experiment_id', 'epoch'),

    # ⚠️ Avoid duplicate indexes on:
    # - ForeignKey columns (auto-indexed in PostgreSQL)
    # - Columns with unique=True (auto-indexed)
)
```

---

## 2. Relationship Patterns

### 2.1 Bidirectional Relationships

```python
# Parent model
class User(BaseModel):
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

# Child model
class APIKey(Base):
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    user = relationship("User", back_populates="api_keys")
```

### 2.2 Many-to-Many with Junction Table

```python
class ExperimentTag(BaseModel):
    __tablename__ = 'experiment_tags'

    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    tag_id = Column(Integer, ForeignKey('tags.id', ondelete='CASCADE'), nullable=False)
    added_by = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))  # Audit trail

    __table_args__ = (
        UniqueConstraint('experiment_id', 'tag_id', name='uq_experiment_tag'),
    )
```

### 2.3 Self-Referential Relationships

```python
# For NAS campaigns with best_trial reference
best_trial_id = Column(Integer, ForeignKey('nas_trials.id', use_alter=True), nullable=True)
```

### 2.4 Cascade Delete Patterns

| Pattern               | Use Case                              |
| --------------------- | ------------------------------------- |
| `ondelete='CASCADE'`  | Child data meaningless without parent |
| `ondelete='SET NULL'` | Audit trails, logs, history           |
| `ondelete='RESTRICT'` | Prevent deletion if children exist    |

---

## 3. Migration Patterns

### 3.1 Migration File Structure

```sql
-- Migration: 001_add_feature.sql
-- Feature: Brief description
-- Date: YYYY-MM-DD

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS new_table (...);
CREATE INDEX IF NOT EXISTS idx_name ON table(column);

-- ============================================================================
-- DOWN MIGRATION (ROLLBACK)
-- ============================================================================
/*
DROP TABLE IF EXISTS new_table CASCADE;
*/

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'new_table') THEN
        RAISE NOTICE 'Migration successful: new_table created';
    ELSE
        RAISE EXCEPTION 'Migration failed: new_table not created';
    END IF;
END $$;
```

### 3.2 Idempotent Migrations

```sql
-- ✅ Always use IF NOT EXISTS / IF EXISTS
CREATE TABLE IF NOT EXISTS api_keys (...);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_enabled BOOLEAN DEFAULT FALSE;

-- ✅ Drop triggers before recreating
DROP TRIGGER IF EXISTS update_timestamp ON table;
CREATE TRIGGER update_timestamp ...;
```

### 3.3 Updated_at Triggers

```sql
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
```

---

## 4. Query Optimization Patterns

### 4.1 Slow Query Logging

```python
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > 1.0:  # Threshold: 1 second
        logger.warning(f"Slow query detected ({total:.2f}s): {statement[:200]}...")
```

### 4.2 Partial Indexes (PostgreSQL)

```sql
-- Only index active records (common query pattern)
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_session_logs_active ON session_logs(user_id, is_active) WHERE is_active = TRUE;

-- Only index recent data
CREATE INDEX idx_api_usage_recent ON api_usage(timestamp)
    WHERE timestamp > NOW() - INTERVAL '30 days';
```

### 4.3 Pre-computed Aggregations

```python
class APIMetricsSummary(BaseModel):
    """Pre-computed statistics for faster dashboard loading."""
    period_start = Column(DateTime, nullable=False, index=True)
    aggregation_type = Column(String(20))  # 'hourly', 'daily', 'weekly'

    total_requests = Column(Integer, default=0)
    avg_response_time_ms = Column(Float)
    p95_response_time_ms = Column(Float)
    p99_response_time_ms = Column(Float)
```

---

## 5. Session Management Conventions

### 5.1 Context Manager Pattern

```python
@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with get_db_session() as session:
    user = session.query(User).filter_by(id=user_id).first()
```

### 5.2 Generator for Dependency Injection

```python
def get_db():
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Usage with FastAPI
@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).get(user_id)
```

### 5.3 Connection Pool Configuration

```python
# PostgreSQL production settings
engine_kwargs = {
    "pool_pre_ping": True,       # Validate connections before use
    "pool_recycle": 3600,        # Recycle connections after 1 hour
    "pool_size": 30,             # Base pool size
    "max_overflow": 30,          # Burst capacity
    "pool_timeout": 30,          # Wait timeout for connection
    "echo": False,               # Disable SQL logging in production
}

# Apply only for PostgreSQL
if str(DATABASE_URL).startswith("postgresql"):
    engine_kwargs.update({...})
```

### 5.4 Scoped Sessions for Thread Safety

```python
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionScoped = scoped_session(SessionLocal)  # Thread-local sessions
```

---

## Quick Reference

| Pattern        | Example                                              |
| -------------- | ---------------------------------------------------- |
| Base model     | `class MyModel(BaseModel)`                           |
| Timestamps     | Auto via `BaseModel` inheritance                     |
| FK cascade     | `ForeignKey('table.id', ondelete='CASCADE')`         |
| Bidirectional  | `back_populates="related_name"`                      |
| Orphan cleanup | `cascade="all, delete-orphan"`                       |
| Session usage  | `with get_db_session() as session:`                  |
| Slow queries   | `@event.listens_for(engine, "after_cursor_execute")` |
| Migrations     | `CREATE TABLE IF NOT EXISTS` + verification          |
