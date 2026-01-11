"""
Database connection and session management.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import Pool
from contextlib import contextmanager
import time

from dashboard_config import DATABASE_URL
from models.base import Base
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)

# Create engine with optimized connection pool
# Configuration for 26+ concurrent callbacks (Dash/Plotly architecture)
# Estimated concurrent connections: ~16 peak, +14 headroom = 30 pool_size
# Create engine with optimized connection pool
engine_kwargs = {
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": False,
}

# Add PostgreSQL-specific arguments
if str(DATABASE_URL).startswith("postgresql"):
    engine_kwargs.update({
        "pool_size": 30,
        "max_overflow": 30,
        "pool_timeout": 30,
        "max_identifier_length": 128,
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)

# Performance monitoring - log slow queries
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record query start time for performance monitoring."""
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries that exceed 1 second execution time."""
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > 1.0:  # Log slow queries (>1 second)
        logger.warning(
            f"Slow query detected ({total:.2f}s): {statement[:200]}..."
            if len(statement) > 200 else f"Slow query detected ({total:.2f}s): {statement}"
        )

# Connection pool monitoring
@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new database connections."""
    logger.debug("New database connection established")

@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Monitor connection pool utilization."""
    # This can be extended to track pool metrics
    pass

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionScoped = scoped_session(SessionLocal)


def init_database():
    """Initialize database tables."""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            session.query(Model).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    Dependency for getting database sessions.
    Yields a database session and closes it after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
