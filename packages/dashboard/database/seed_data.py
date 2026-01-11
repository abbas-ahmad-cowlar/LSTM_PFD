"""
Seed initial data into the database.
"""
from database.connection import get_db_session
from models.user import User
from models.dataset import Dataset
from utils.logger import setup_logger
from dashboard_config import FAULT_CLASSES, PHASE_0_CACHE_PATH
import os

logger = setup_logger(__name__)


def seed_initial_data():
    """Seed initial data (fault classes, default user, sample dataset)."""
    with get_db_session() as session:
        # Create default user
        default_user = session.query(User).filter_by(username='admin').first()
        if not default_user:
            default_user = User(
                username='admin',
                email='admin@localhost',
                role='admin'
            )
            session.add(default_user)
            logger.info("Created default admin user")

        # Check if default dataset exists
        if os.path.exists(PHASE_0_CACHE_PATH):
            default_dataset = session.query(Dataset).filter_by(name='Default').first()
            if not default_dataset:
                default_dataset = Dataset(
                    name='Default',
                    description='Default dataset from Phase 0',
                    num_signals=1430,
                    fault_types=FAULT_CLASSES,
                    severity_levels=['incipient', 'mild', 'moderate', 'severe'],
                    file_path=str(PHASE_0_CACHE_PATH),
                    created_by=default_user.id
                )
                session.add(default_dataset)
                logger.info("Created default dataset")

        session.commit()
        logger.info("Seed data inserted successfully")


if __name__ == '__main__':
    from database.connection import init_database
    init_database()
    seed_initial_data()
