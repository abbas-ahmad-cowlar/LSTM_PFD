"""
Database migration runner.
Executes SQL migration files in order.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import engine
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_migration(migration_file: str):
    """
    Run a SQL migration file.

    Args:
        migration_file: Path to migration SQL file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(migration_file, 'r') as f:
            sql = f.read()

        # Execute SQL
        with engine.connect() as conn:
            # Split on semicolon and execute each statement
            for statement in sql.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('/*') and not statement.startswith('--'):
                    conn.execute(statement)
            conn.commit()

        logger.info(f"Migration successful: {migration_file}")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {migration_file} - {e}")
        return False


def run_all_migrations():
    """Run all migrations in order."""
    migrations_dir = Path(__file__).parent / 'migrations'

    if not migrations_dir.exists():
        logger.warning("No migrations directory found")
        return

    # Get all .sql files sorted by name
    migration_files = sorted(migrations_dir.glob('*.sql'))

    if not migration_files:
        logger.info("No migration files found")
        return

    logger.info(f"Found {len(migration_files)} migration(s)")

    for migration_file in migration_files:
        logger.info(f"Running migration: {migration_file.name}")
        if not run_migration(str(migration_file)):
            logger.error(f"Migration failed, stopping: {migration_file.name}")
            return False

    logger.info("All migrations completed successfully")
    return True


if __name__ == '__main__':
    import argparse
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument(
        '--migration',
        type=str,
        help='Specific migration file to run (e.g., 001_add_api_keys.sql)'
    )

    args = parser.parse_args()

    if args.migration:
        # Run specific migration
        migration_path = Path(__file__).parent / 'migrations' / args.migration
        if not migration_path.exists():
            logger.error(f"Migration file not found: {args.migration}")
            sys.exit(1)

        success = run_migration(str(migration_path))
        sys.exit(0 if success else 1)
    else:
        # Run all migrations
        success = run_all_migrations()
        sys.exit(0 if success else 1)
