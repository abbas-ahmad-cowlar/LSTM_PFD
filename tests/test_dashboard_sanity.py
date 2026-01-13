
import unittest
import sys
import os
from pathlib import Path

# Add project root and dashboard directory to path to simulate app environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / 'packages' / 'dashboard'

# Mock environment variables for config validation BEFORE any imports
os.environ['DATABASE_URL'] = "sqlite:///:memory:"
os.environ['SECRET_KEY'] = "test_secret_key_for_testing_purposes_1234567890"
os.environ['JWT_SECRET_KEY'] = "test_jwt_secret_key_for_testing_purposes_1234567890"
os.environ['JWT_ACCESS_TOKEN_EXPIRES'] = "3600"
os.environ['SKIP_CONFIG_VALIDATION'] = "True"
os.environ['ENV'] = "test"
os.environ['DEBUG'] = "True"

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
        
        def find_id(component, target_id, visited=None):
            """Recursively find a component by ID in the layout tree."""
            if visited is None:
                visited = set()
            
            # Avoid infinite recursion with object identity
            comp_id = id(component)
            if comp_id in visited:
                return False
            visited.add(comp_id)
            
            # Check if this component has the target ID
            if hasattr(component, 'id') and component.id == target_id:
                return True
            
            # Check children attribute (standard Dash)
            if hasattr(component, 'children'):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        if find_id(child, target_id, visited):
                            return True
                elif children is not None:
                    if find_id(children, target_id, visited):
                        return True
            
            # For dbc components, they may have different attribute names
            # Try to iterate over common attribute names
            for attr_name in ['children', 'content', 'items']:
                if hasattr(component, attr_name):
                    attr = getattr(component, attr_name)
                    if isinstance(attr, list):
                        for item in attr:
                            if find_id(item, target_id, visited):
                                return True
                    elif attr is not None and hasattr(attr, 'id'):
                        if find_id(attr, target_id, visited):
                            return True
            
            return False

        # Check for core stores and containers
        self.assertTrue(find_id(layout, 'session-store'), "Session store not found in layout")
        self.assertTrue(find_id(layout, 'url'), "URL component not found in layout")
        self.assertTrue(find_id(layout, 'page-content'), "Page content container not found")

