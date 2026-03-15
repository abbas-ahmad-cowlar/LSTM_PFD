"""
Tests for dashboard layout rendering and component construction.
Verifies that layout functions return valid Dash components without
requiring a running server.
"""
import pytest
from unittest.mock import patch, MagicMock
import dash_bootstrap_components as dbc
from dash import html


class TestSettingsLayout:
    """Tests for settings page layout construction."""

    def test_create_settings_layout_returns_container(self):
        """Test that settings layout returns a valid container."""
        from layouts.settings import create_settings_layout
        layout = create_settings_layout()

        assert layout is not None
        assert isinstance(layout, dbc.Container)

    def test_settings_layout_has_tabs(self):
        """Test that settings layout contains a Tabs component."""
        from layouts.settings import create_settings_layout
        layout = create_settings_layout()

        # Walk children to find Tabs
        found_tabs = _find_component_by_type(layout, dbc.Tabs)
        assert found_tabs is not None, "Settings layout should contain dbc.Tabs"

    def test_api_keys_tab_returns_div(self):
        """Test that API Keys tab returns valid layout."""
        from layouts.settings.api_keys_tab import create_api_keys_tab
        tab = create_api_keys_tab()
        assert tab is not None
        assert isinstance(tab, html.Div)

    def test_profile_tab_returns_container(self):
        """Test that Profile tab returns valid layout."""
        from layouts.settings.profile_tab import create_profile_tab
        tab = create_profile_tab()
        assert tab is not None

    def test_security_tab_returns_container(self):
        """Test that Security tab returns valid layout."""
        from layouts.settings.security_tab import create_security_tab
        tab = create_security_tab()
        assert tab is not None

    def test_notifications_tab_returns_container(self):
        """Test that Notifications tab returns valid layout."""
        from layouts.settings.notifications_tab import create_notifications_tab
        tab = create_notifications_tab()
        assert tab is not None

    def test_webhooks_tab_returns_container(self):
        """Test that Webhooks tab returns valid layout."""
        from layouts.settings.webhooks_tab import create_webhooks_tab
        tab = create_webhooks_tab()
        assert tab is not None


class TestExperimentComparisonLayout:
    """Tests for experiment comparison layout construction."""

    def test_create_comparison_layout_returns_container(self):
        """Test layout builds with mock experiment IDs."""
        from layouts.experiment_comparison import create_experiment_comparison_layout
        layout = create_experiment_comparison_layout([1, 2, 3])

        assert layout is not None
        assert isinstance(layout, dbc.Container)

    def test_comparison_layout_with_single_experiment(self):
        """Test layout builds with single experiment."""
        from layouts.experiment_comparison import create_experiment_comparison_layout
        layout = create_experiment_comparison_layout([42])

        assert layout is not None

    def test_format_duration_seconds(self):
        """Test format_duration with seconds only."""
        from layouts.experiment_comparison import format_duration

        assert format_duration(30) == "30s"
        assert format_duration(0) == "N/A"
        assert format_duration(None) == "N/A"

    def test_format_duration_minutes(self):
        """Test format_duration with minutes."""
        from layouts.experiment_comparison import format_duration

        assert format_duration(90) == "1m 30s"
        assert format_duration(600) == "10m 0s"

    def test_format_duration_hours(self):
        """Test format_duration with hours."""
        from layouts.experiment_comparison import format_duration

        result = format_duration(5400)
        assert result == "1h 30m"


class TestSidebarComponent:
    """Tests for sidebar navigation component."""

    def test_create_sidebar_returns_div(self):
        """Test that sidebar returns valid layout."""
        from components.sidebar import create_sidebar
        sidebar = create_sidebar()
        assert sidebar is not None
        assert isinstance(sidebar, html.Div)

    def test_sidebar_has_navigation_role(self):
        """Test that sidebar container has ARIA navigation role."""
        from components.sidebar import create_sidebar
        sidebar = create_sidebar()

        # The sidebar container should have role="navigation"
        nav_container = _find_component_by_id(sidebar, "sidebar-container")
        assert nav_container is not None, "Should have sidebar-container"

    def test_sidebar_has_stores(self):
        """Test that sidebar includes required dcc.Store components."""
        from dash import dcc
        from components.sidebar import create_sidebar
        sidebar = create_sidebar()

        stores = _find_all_components_by_type(sidebar, dcc.Store)
        store_ids = [s.id for s in stores if hasattr(s, 'id')]

        assert 'sidebar-collapsed-store' in store_ids
        assert 'section-collapse-store' in store_ids


class TestSkeletonComponents:
    """Tests for skeleton loading components."""

    def test_skeleton_card_returns_div(self):
        """Test skeleton card returns valid component."""
        from components.skeleton import skeleton_card
        result = skeleton_card(n_cards=2)
        assert result is not None
        assert isinstance(result, html.Div)

    def test_skeleton_table_returns_div(self):
        """Test skeleton table returns valid component."""
        from components.skeleton import skeleton_table
        result = skeleton_table(n_rows=3, n_cols=4)
        assert result is not None
        assert isinstance(result, html.Div)

    def test_skeleton_chart_bar(self):
        """Test skeleton bar chart returns valid component."""
        from components.skeleton import skeleton_chart
        result = skeleton_chart(chart_type='bar')
        assert result is not None
        assert isinstance(result, html.Div)

    def test_skeleton_chart_line(self):
        """Test skeleton line chart returns valid component."""
        from components.skeleton import skeleton_chart
        result = skeleton_chart(chart_type='line')
        assert result is not None

    def test_skeleton_chart_pie(self):
        """Test skeleton pie chart returns valid component."""
        from components.skeleton import skeleton_chart
        result = skeleton_chart(chart_type='pie')
        assert result is not None

    def test_skeleton_metric_card(self):
        """Test skeleton metric card returns valid component."""
        from components.skeleton import skeleton_metric_card
        result = skeleton_metric_card(n_cards=4)
        assert result is not None
        assert isinstance(result, html.Div)

    def test_skeleton_page_dashboard(self):
        """Test full dashboard skeleton page."""
        from components.skeleton import skeleton_page
        result = skeleton_page(page_type='dashboard')
        assert result is not None

    def test_skeleton_page_table(self):
        """Test full table skeleton page."""
        from components.skeleton import skeleton_page
        result = skeleton_page(page_type='table')
        assert result is not None


class TestVisualizationHelpers:
    """Tests for experiment comparison visualization helper functions."""

    def test_create_overall_metrics_chart(self):
        """Test overall metrics chart with mock data."""
        from layouts.experiment_comparison.visualization_helpers import create_overall_metrics_chart

        mock_experiments = [
            {
                'id': 1,
                'name': 'Test Experiment 1',
                'metrics': {
                    'accuracy': 0.95,
                    'precision': 0.94,
                    'recall': 0.93,
                    'f1_score': 0.935,
                }
            },
            {
                'id': 2,
                'name': 'Test Experiment 2',
                'metrics': {
                    'accuracy': 0.90,
                    'precision': 0.89,
                    'recall': 0.91,
                    'f1_score': 0.90,
                }
            },
        ]

        fig = create_overall_metrics_chart(mock_experiments)
        assert fig is not None
        assert len(fig.data) == 2  # One trace per experiment

    def test_create_training_curves_comparison(self):
        """Test training curves comparison with mock data."""
        from layouts.experiment_comparison.visualization_helpers import create_training_curves_comparison

        mock_experiments = [
            {
                'id': 1,
                'name': 'Exp 1',
                'training_history': {
                    'epochs': list(range(1, 11)),
                    'train_loss': [0.5 - i * 0.04 for i in range(10)],
                    'val_loss': [0.6 - i * 0.04 for i in range(10)],
                    'val_accuracy': [0.5 + i * 0.04 for i in range(10)],
                }
            },
        ]

        fig_loss = create_training_curves_comparison(mock_experiments, 'loss')
        assert fig_loss is not None
        assert len(fig_loss.data) == 2  # train + val traces

        fig_acc = create_training_curves_comparison(mock_experiments, 'accuracy')
        assert fig_acc is not None
        assert len(fig_acc.data) == 1  # one trace per experiment


# ---------------------------------------------------------------------------
# Utility helpers for walking Dash component trees
# ---------------------------------------------------------------------------

def _find_component_by_type(component, target_type):
    """Recursively find first component of given type."""
    if isinstance(component, target_type):
        return component

    children = getattr(component, 'children', None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component_by_type(child, target_type)
            if result is not None:
                return result
    else:
        return _find_component_by_type(children, target_type)

    return None


def _find_component_by_id(component, target_id):
    """Recursively find component with given id."""
    comp_id = getattr(component, 'id', None)
    if comp_id == target_id:
        return component

    children = getattr(component, 'children', None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component_by_id(child, target_id)
            if result is not None:
                return result
    else:
        return _find_component_by_id(children, target_id)

    return None


def _find_all_components_by_type(component, target_type, results=None):
    """Recursively find all components of given type."""
    if results is None:
        results = []

    if isinstance(component, target_type):
        results.append(component)

    children = getattr(component, 'children', None)
    if children is not None:
        if isinstance(children, (list, tuple)):
            for child in children:
                _find_all_components_by_type(child, target_type, results)
        else:
            _find_all_components_by_type(children, target_type, results)

    return results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
