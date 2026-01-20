"""
Left sidebar navigation component with collapsible state and icons.

Enhanced with:
- Collapsed state showing icons only
- Hover tooltips for collapsed state
- Smooth width transition
- Toggle button
- Collapsible section groups
- Scrollable navigation area

Addresses Deficiency #29 (Priority 40): Sidebar Icons with collapsed state
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, clientside_callback, ALL, MATCH
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


# Navigation items with icons and routes
# Items can have 'admin_only': True to indicate they require admin role
NAV_ITEMS = {
    'main': [
        {'id': 'nav-home', 'icon': 'fas fa-home', 'label': 'Home', 'href': '/'},
    ],
    'Data': [
        {'id': 'nav-data-gen', 'icon': 'fas fa-cog', 'label': 'Generate Data', 'href': '/data-generation'},
        {'id': 'nav-data-explorer', 'icon': 'fas fa-database', 'label': 'Data Explorer', 'href': '/data-explorer'},
        {'id': 'nav-signal-viewer', 'icon': 'fas fa-signal', 'label': 'Signal Viewer', 'href': '/signal-viewer'},
        {'id': 'nav-datasets', 'icon': 'fas fa-folder', 'label': 'Datasets', 'href': '/datasets'},
        {'id': 'nav-features', 'icon': 'fas fa-project-diagram', 'label': 'Feature Engineering', 'href': '/feature-engineering'},
    ],
    'Training': [
        {'id': 'nav-new-exp', 'icon': 'fas fa-flask', 'label': 'New Experiment', 'href': '/experiment/new'},
        {'id': 'nav-experiments', 'icon': 'fas fa-list', 'label': 'Experiments', 'href': '/experiments'},
        {'id': 'nav-hpo', 'icon': 'fas fa-search', 'label': 'HPO Campaigns', 'href': '/hpo/campaigns'},
        {'id': 'nav-nas', 'icon': 'fas fa-sitemap', 'label': 'NAS', 'href': '/nas'},
    ],
    'Evaluation': [
        {'id': 'nav-eval', 'icon': 'fas fa-chart-bar', 'label': 'Evaluation', 'href': '/evaluation'},
        {'id': 'nav-xai', 'icon': 'fas fa-brain', 'label': 'XAI Dashboard', 'href': '/xai'},
        {'id': 'nav-viz', 'icon': 'fas fa-chart-area', 'label': 'Visualizations', 'href': '/visualization'},
    ],
    'Production': [
        {'id': 'nav-deploy', 'icon': 'fas fa-rocket', 'label': 'Deployment', 'href': '/deployment'},
        {'id': 'nav-api', 'icon': 'fas fa-server', 'label': 'API Monitoring', 'href': '/api-monitoring', 'admin_only': True},
        {'id': 'nav-testing', 'icon': 'fas fa-vial', 'label': 'Testing & QA', 'href': '/testing'},
    ],
    'System': [
        {'id': 'nav-health', 'icon': 'fas fa-heartbeat', 'label': 'System Health', 'href': '/system-health', 'admin_only': True},
        {'id': 'nav-settings', 'icon': 'fas fa-cog', 'label': 'Settings', 'href': '/settings'},
    ],
}


def create_nav_link(item: dict, collapsed: bool = False, is_admin: bool = True):
    """
    Create a single navigation link with icon and optional tooltip.
    
    Args:
        item: Navigation item dict
        collapsed: Whether sidebar is collapsed
        is_admin: Whether current user has admin role
    """
    # Skip admin-only items for non-admin users
    if item.get('admin_only', False) and not is_admin:
        return None  # Will be filtered out
    
    icon = html.I(className=f"{item['icon']} nav-icon")
    label = html.Span(item['label'], className="nav-label")
    
    # Add lock icon for admin-only items
    if item.get('admin_only', False):
        admin_badge = html.I(className="fas fa-lock ms-2 admin-badge", 
                             style={"fontSize": "0.7rem", "opacity": "0.6"})
        children = [icon, label, admin_badge]
    else:
        children = [icon, label]
    
    return dbc.NavLink(
        children,
        id=item['id'],
        href=item['href'],
        active="exact",
        className="sidebar-nav-link"
    )


def create_section(section_name: str, items: list, is_admin: bool = True):
    """
    Create a navigation section with collapsible header and links.
    
    Args:
        section_name: Section name for header
        items: List of nav items
        is_admin: Whether current user has admin role
    """
    if section_name == 'main':
        # No header for main section, just return links
        links = [create_nav_link(item, is_admin=is_admin) for item in items]
        return [link for link in links if link is not None]  # Filter None items
    
    # Create section ID for collapse functionality
    section_id = section_name.lower().replace(' ', '-')
    
    # Clickable header that toggles collapse
    header = html.Div(
        [
            html.I(className="fas fa-chevron-down section-chevron me-2", id=f"chevron-{section_id}"),
            html.Span(section_name, className="section-label")
        ],
        id={'type': 'section-header', 'section': section_id},
        className="sidebar-section-header",
        n_clicks=0
    )
    
    # Filter links (remove admin-only for non-admin users)
    nav_links = [create_nav_link(item, is_admin=is_admin) for item in items]
    nav_links = [link for link in nav_links if link is not None]  # Filter None
    
    # If no links visible after filtering, skip entire section
    if not nav_links:
        return []
    
    # Links container (collapsible)
    links = html.Div(
        nav_links,
        id={'type': 'section-content', 'section': section_id},
        className="sidebar-section-content"
    )
    
    return [
        html.Hr(className="sidebar-divider"),
        header,
        links
    ]


def create_sidebar():
    """Create enhanced sidebar navigation with collapse support and scrolling."""
    
    # Build navigation elements
    nav_elements = []
    for section_name, items in NAV_ITEMS.items():
        nav_elements.extend(create_section(section_name, items))
    
    # Sidebar header with logo and toggle
    sidebar_header = html.Div([
        html.Div([
            html.I(className="fas fa-bearings sidebar-logo-icon"),
            html.Span("LSTM-PFD", className="sidebar-logo-text")
        ], className="sidebar-logo"),
        html.Button(
            html.I(className="fas fa-chevron-left", id="toggle-icon"),
            id="sidebar-toggle-btn",
            className="sidebar-toggle-btn",
            n_clicks=0
        )
    ], className="sidebar-header")
    
    # Navigation container with scrollable area
    nav_container = html.Div(
        nav_elements,
        className="sidebar-nav"
    )
    
    # Sidebar footer
    sidebar_footer = html.Div([
        html.Div([
            html.I(className="fas fa-info-circle me-2"),
            html.Span("v2.0.0", className="version-text")
        ], className="sidebar-version")
    ], className="sidebar-footer")
    
    return html.Div([
        # Store for collapsed state
        dcc.Store(id='sidebar-collapsed-store', data=False),
        # Store for section collapse states (all sections expanded by default)
        dcc.Store(id='section-collapse-store', data={}),
        
        # Sidebar container
        html.Div([
            sidebar_header,
            nav_container,
            sidebar_footer
        ], id="sidebar-container", className="sidebar")
    ])


def create_sidebar_styles():
    """Return CSS styles for the enhanced sidebar."""
    return """
    /* Enhanced Sidebar Styles */
    .sidebar {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 260px;
        background: linear-gradient(180deg, #1a1f2e 0%, #0d1117 100%);
        padding: 0;
        display: flex;
        flex-direction: column;
        transition: width 0.3s ease-in-out;
        z-index: 1000;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }
    
    .sidebar.collapsed {
        width: 70px;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 60px;
        flex-shrink: 0;
    }
    
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        overflow: hidden;
    }
    
    .sidebar-logo-icon {
        font-size: 1.5rem;
        color: #3b82f6;
        flex-shrink: 0;
    }
    
    .sidebar-logo-text {
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
        white-space: nowrap;
        opacity: 1;
        transition: opacity 0.2s ease;
    }
    
    .sidebar.collapsed .sidebar-logo-text {
        opacity: 0;
        width: 0;
    }
    
    .sidebar-toggle-btn {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        border-radius: 8px;
        color: #94a3b8;
        padding: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        flex-shrink: 0;
    }
    
    .sidebar-toggle-btn:hover {
        background: rgba(59, 130, 246, 0.3);
        color: #3b82f6;
    }
    
    .sidebar.collapsed .sidebar-toggle-btn {
        transform: rotate(180deg);
    }
    
    /* Navigation - SCROLLABLE */
    .sidebar-nav {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 0.5rem;
        min-height: 0;
    }
    
    /* Custom scrollbar for sidebar */
    .sidebar-nav::-webkit-scrollbar {
        width: 6px;
    }
    
    .sidebar-nav::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }
    
    .sidebar-nav::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    .sidebar-nav::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    .sidebar-divider {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Collapsible Section Header */
    .sidebar-section-header {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.5rem 0.75rem;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        cursor: pointer;
        display: flex;
        align-items: center;
        border-radius: 6px;
        transition: all 0.2s ease;
        user-select: none;
    }
    
    .sidebar-section-header:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #94a3b8;
    }
    
    .sidebar-section-header .section-chevron {
        transition: transform 0.25s ease;
        font-size: 0.65rem;
    }
    
    .sidebar-section-header.collapsed .section-chevron {
        transform: rotate(-90deg);
    }
    
    /* Section Content (collapsible) */
    .sidebar-section-content {
        overflow: hidden;
        max-height: 500px;
        transition: max-height 0.3s ease, opacity 0.2s ease;
        opacity: 1;
    }
    
    .sidebar-section-content.collapsed {
        max-height: 0;
        opacity: 0;
    }
    
    .sidebar.collapsed .sidebar-section-header {
        text-align: center;
        justify-content: center;
    }
    
    .sidebar.collapsed .section-label {
        display: none;
    }
    
    .sidebar.collapsed .section-chevron {
        display: none;
    }
    
    /* Navigation Links */
    .sidebar-nav-link {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        color: #94a3b8 !important;
        text-decoration: none;
        transition: all 0.2s ease;
        margin-bottom: 0.25rem;
        position: relative;
        white-space: nowrap;
        overflow: hidden;
    }
    
    .sidebar-nav-link:hover {
        background: rgba(59, 130, 246, 0.15);
        color: #e2e8f0 !important;
    }
    
    .sidebar-nav-link.active {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }
    
    .nav-icon {
        font-size: 1.1rem;
        width: 1.25rem;
        text-align: center;
        flex-shrink: 0;
    }
    
    .nav-label {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 1;
        transition: opacity 0.2s ease;
    }
    
    .sidebar.collapsed .nav-label {
        opacity: 0;
        width: 0;
        overflow: hidden;
    }
    
    /* Collapsed state tooltip */
    .sidebar.collapsed .sidebar-nav-link:hover::after {
        content: attr(title);
        position: absolute;
        left: 70px;
        top: 50%;
        transform: translateY(-50%);
        background: #1e293b;
        color: #ffffff;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        white-space: nowrap;
        z-index: 1001;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        animation: tooltipFadeIn 0.2s ease;
    }
    
    @keyframes tooltipFadeIn {
        from { opacity: 0; transform: translateY(-50%) translateX(-10px); }
        to { opacity: 1; transform: translateY(-50%) translateX(0); }
    }
    
    /* Footer */
    .sidebar-footer {
        padding: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        flex-shrink: 0;
    }
    
    .sidebar-version {
        display: flex;
        align-items: center;
        color: #64748b;
        font-size: 0.75rem;
        padding: 0.5rem;
        justify-content: center;
    }
    
    .sidebar.collapsed .version-text {
        display: none;
    }
    
    /* Main content adjustment */
    .main-content {
        margin-left: 260px;
        transition: margin-left 0.3s ease-in-out;
    }
    
    .main-content.sidebar-collapsed {
        margin-left: 70px;
    }
    
    /* Responsive - Mobile */
    @media (max-width: 768px) {
        .sidebar {
            transform: translateX(-100%);
            width: 260px;
        }
        
        .sidebar.mobile-open {
            transform: translateX(0);
        }
        
        .main-content {
            margin-left: 0;
        }
    }
    """


# Clientside callback for toggle functionality
clientside_callback(
    """
    function(n_clicks, is_collapsed) {
        if (n_clicks === undefined || n_clicks === 0) {
            return [is_collapsed, {}];
        }
        
        const new_collapsed = !is_collapsed;
        const sidebar = document.getElementById('sidebar-container');
        const body = document.body;
        
        if (sidebar) {
            if (new_collapsed) {
                sidebar.classList.add('collapsed');
            } else {
                sidebar.classList.remove('collapsed');
            }
        }
        
        // Toggle body class for dynamic layout (header & content margins)
        if (body) {
            if (new_collapsed) {
                body.classList.add('sidebar-collapsed');
            } else {
                body.classList.remove('sidebar-collapsed');
            }
        }
        
        return [new_collapsed, {}];
    }
    """,
    [Output('sidebar-collapsed-store', 'data'),
     Output('sidebar-toggle-btn', 'style')],
    [Input('sidebar-toggle-btn', 'n_clicks')],
    [State('sidebar-collapsed-store', 'data')],
    prevent_initial_call=True
)


# Clientside callback for section collapse toggle
clientside_callback(
    """
    function(n_clicks_list, section_states) {
        // Ensure section_states is an object
        if (!section_states) {
            section_states = {};
        }
        
        // Get the triggered element
        const ctx = dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {
            return dash_clientside.no_update;
        }
        
        const triggeredItem = ctx.triggered[0];
        if (!triggeredItem || !triggeredItem.value) {
            return dash_clientside.no_update;
        }
        
        // Parse the triggered prop_id to extract section name
        const propId = triggeredItem.prop_id;
        let sectionId;
        try {
            // Remove .n_clicks suffix and parse JSON
            const jsonStr = propId.split('.n_clicks')[0];
            const parsed = JSON.parse(jsonStr);
            sectionId = parsed.section;
        } catch (e) {
            console.error('Failed to parse section ID:', e);
            return dash_clientside.no_update;
        }
        
        if (!sectionId) {
            return dash_clientside.no_update;
        }
        
        // Toggle the section state
        const newStates = Object.assign({}, section_states);
        newStates[sectionId] = !newStates[sectionId];
        
        // Update DOM classes for all headers and contents
        document.querySelectorAll('.sidebar-section-content').forEach(function(el) {
            try {
                const elId = JSON.parse(el.id);
                if (elId.section === sectionId) {
                    if (newStates[sectionId]) {
                        el.classList.add('collapsed');
                    } else {
                        el.classList.remove('collapsed');
                    }
                }
            } catch (e) {}
        });
        
        // Update header chevron
        document.querySelectorAll('.sidebar-section-header').forEach(function(el) {
            try {
                const elId = JSON.parse(el.id);
                if (elId.section === sectionId) {
                    if (newStates[sectionId]) {
                        el.classList.add('collapsed');
                    } else {
                        el.classList.remove('collapsed');
                    }
                }
            } catch (e) {}
        });
        
        return newStates;
    }
    """,
    Output('section-collapse-store', 'data'),
    [Input({'type': 'section-header', 'section': ALL}, 'n_clicks')],
    [State('section-collapse-store', 'data')],
    prevent_initial_call=True
)


