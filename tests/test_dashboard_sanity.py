
import unittest
import sys
import os
from pathlib import Path

# Add project root and dashboard directory to path to simulate app environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / 'packages' / 'dashboard'

# Mock environment variables for config validation
os.environ['DATABASE_URL'] = "sqlite:///:memory:"
os.environ['SECRET_KEY'] = "test_key"
os.environ['SKIP_CONFIG_VALIDATION'] = "True"  # heuristic to skip strict checks

sys.path.insert(0, str(DASHBOARD_DIR))
sys.path.append(str(PROJECT_ROOT))

class TestDashboardSanity(unittest.TestCase):
    """Test Dashboard application integrity."""

    def test_app_import(self):
        """Test that app.py can be imported without errors."""
        try:
            from packages.dashboard.app import app, server
            self.assertIsNotNone(app.layout)
            self.assertIsNotNone(server)
        except ImportError as e:
            self.fail(f"Failed to import app: {e}")
        except Exception as e:
            self.fail(f"App initialization failed: {e}")

    def test_layout_structure(self):
        """Test basic layout components present."""
        from packages.dashboard.app import app
        layout = app.layout
        
        # Dash Layout is a tree of components.
        # Check for key IDs
        # We can traverse the component tree/dictionary
        
        def find_id(component, target_id):
            if hasattr(component, 'id') and component.id == target_id:
                return True
            if hasattr(component, 'children'):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        if find_id(child, target_id):
                            return True
                elif children:
                    if find_id(children, target_id):
                        return True
            return False

        # Check for core stores and containers
        self.assertTrue(find_id(layout, 'session-store'), "Session store not found in layout")
        self.assertTrue(find_id(layout, 'url'), "URL component not found in layout")
        self.assertTrue(find_id(layout, 'page-content'), "Page content container not found")

