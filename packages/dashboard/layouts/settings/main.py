"""
Settings page — main layout orchestrator.
Assembles the tab-based settings page from individual tab modules.
"""
from dash import html
import dash_bootstrap_components as dbc

from layouts.settings.api_keys_tab import create_api_keys_tab
from layouts.settings.profile_tab import create_profile_tab
from layouts.settings.security_tab import create_security_tab
from layouts.settings.notifications_tab import create_notifications_tab
from layouts.settings.webhooks_tab import create_webhooks_tab
from layouts.email_digest_management import create_email_digest_tab


def create_settings_layout():
    """
    Create the settings page layout.

    Features:
        - API Keys management tab
        - User profile tab
        - Security settings tab
        - Notifications tab
        - Webhooks tab
        - Email Digest tab
    """
    return dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2("⚙️ Settings", className="mb-3"),
                html.P(
                    "Manage your API keys, profile, and security settings.",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),

        # Tabs
        dbc.Tabs([
            dbc.Tab(
                label="🔑 API Keys",
                tab_id="api-keys",
                children=create_api_keys_tab()
            ),
            dbc.Tab(
                label="👤 Profile",
                tab_id="profile",
                children=create_profile_tab()
            ),
            dbc.Tab(
                label="🔒 Security",
                tab_id="security",
                children=create_security_tab()
            ),
            dbc.Tab(
                label="🔔 Notifications",
                tab_id="notifications",
                children=create_notifications_tab()
            ),
            dbc.Tab(
                label="🔗 Webhooks",
                tab_id="webhooks",
                children=create_webhooks_tab()
            ),
            dbc.Tab(
                label="📧 Email Digests",
                tab_id="email-digests",
                children=create_email_digest_tab()
            ),
        ], id='settings-tabs', active_tab='api-keys',
           role="tablist"),

    ], fluid=True, className="py-4")
