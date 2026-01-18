"""
Skeleton Loading Components

Modern skeleton loaders to replace generic spinners with smooth,
content-aware placeholder animations.

Addresses Deficiency #33 (Priority 32): Generic Loading States

Components:
- SkeletonText: Animated text placeholder lines
- SkeletonCard: Card-shaped skeleton with title and content
- SkeletonTable: Table skeleton with rows and columns
- SkeletonChart: Chart placeholder with axes
- SkeletonMetricCard: Dashboard metric card skeleton

Usage:
    from components.skeleton import skeleton_card, skeleton_table
    
    # Show skeleton while loading
    loading_state = skeleton_card(n_cards=3)
"""
import dash_bootstrap_components as dbc
from dash import html


def skeleton_base_styles():
    """Return base CSS for skeleton animations."""
    return """
    /* Skeleton Base Styles */
    .skeleton {
        background: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.05) 25%,
            rgba(255, 255, 255, 0.1) 50%,
            rgba(255, 255, 255, 0.05) 75%
        );
        background-size: 200% 100%;
        animation: skeleton-shimmer 1.5s ease-in-out infinite;
        border-radius: 4px;
    }
    
    .skeleton-light {
        background: linear-gradient(
            90deg,
            #e2e8f0 25%,
            #f1f5f9 50%,
            #e2e8f0 75%
        );
        background-size: 200% 100%;
        animation: skeleton-shimmer 1.5s ease-in-out infinite;
        border-radius: 4px;
    }
    
    @keyframes skeleton-shimmer {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    /* Skeleton Text */
    .skeleton-text {
        height: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .skeleton-text.short {
        width: 40%;
    }
    
    .skeleton-text.medium {
        width: 70%;
    }
    
    .skeleton-text.long {
        width: 90%;
    }
    
    .skeleton-text.title {
        height: 1.5rem;
        width: 50%;
        margin-bottom: 1rem;
    }
    
    /* Skeleton Card */
    .skeleton-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .skeleton-card-light {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Skeleton Table */
    .skeleton-table {
        width: 100%;
    }
    
    .skeleton-table-header {
        display: flex;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .skeleton-table-row {
        display: flex;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .skeleton-table-cell {
        flex: 1;
        height: 1rem;
    }
    
    /* Skeleton Chart */
    .skeleton-chart {
        position: relative;
        height: 200px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .skeleton-chart-bar {
        position: absolute;
        bottom: 20px;
        width: 8%;
        background: linear-gradient(
            90deg,
            rgba(59, 130, 246, 0.2) 25%,
            rgba(59, 130, 246, 0.4) 50%,
            rgba(59, 130, 246, 0.2) 75%
        );
        background-size: 200% 100%;
        animation: skeleton-shimmer 1.5s ease-in-out infinite;
        border-radius: 4px 4px 0 0;
    }
    
    .skeleton-chart-axis-x {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 20px;
        background: rgba(255, 255, 255, 0.02);
    }
    
    .skeleton-chart-axis-y {
        position: absolute;
        top: 0;
        bottom: 20px;
        left: 0;
        width: 40px;
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Skeleton Metric Card */
    .skeleton-metric {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .skeleton-metric-value {
        height: 2.5rem;
        width: 60%;
    }
    
    .skeleton-metric-label {
        height: 0.875rem;
        width: 80%;
    }
    
    .skeleton-metric-change {
        height: 0.75rem;
        width: 40%;
        margin-top: 0.5rem;
    }
    
    /* Skeleton Avatar */
    .skeleton-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .skeleton-avatar.large {
        width: 64px;
        height: 64px;
    }
    
    /* Skeleton Button */
    .skeleton-button {
        height: 38px;
        width: 120px;
        border-radius: 6px;
    }
    
    /* Skeleton Image */
    .skeleton-image {
        width: 100%;
        height: 150px;
        border-radius: 8px;
    }
    
    /* Loading Pulse Effect */
    .loading-pulse {
        animation: loading-pulse 2s ease-in-out infinite;
    }
    
    @keyframes loading-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    """


def skeleton_text(width: str = 'long', is_title: bool = False, dark_mode: bool = True):
    """Create a skeleton text line.
    
    Args:
        width: 'short' (40%), 'medium' (70%), or 'long' (90%)
        is_title: If True, renders as a larger title
        dark_mode: Use dark or light theme
    """
    classes = ['skeleton' if dark_mode else 'skeleton-light', 'skeleton-text']
    
    if is_title:
        classes.append('title')
    else:
        classes.append(width)
    
    return html.Div(className=' '.join(classes))


def skeleton_card(n_cards: int = 1, n_lines: int = 3, show_title: bool = True, 
                  dark_mode: bool = True):
    """Create skeleton card(s) with text lines.
    
    Args:
        n_cards: Number of cards to render
        n_lines: Number of text lines per card
        show_title: Include a title skeleton
        dark_mode: Use dark or light theme
    """
    cards = []
    base_class = 'skeleton' if dark_mode else 'skeleton-light'
    card_class = 'skeleton-card' if dark_mode else 'skeleton-card-light'
    
    for _ in range(n_cards):
        content = []
        
        if show_title:
            content.append(html.Div(className=f'{base_class} skeleton-text title'))
        
        widths = ['long', 'medium', 'short']
        for i in range(n_lines):
            width = widths[i % len(widths)]
            content.append(html.Div(className=f'{base_class} skeleton-text {width}'))
        
        cards.append(html.Div(content, className=card_class))
    
    return html.Div(cards)


def skeleton_table(n_rows: int = 5, n_cols: int = 4, dark_mode: bool = True):
    """Create a skeleton table.
    
    Args:
        n_rows: Number of data rows
        n_cols: Number of columns
        dark_mode: Use dark or light theme
    """
    base_class = 'skeleton' if dark_mode else 'skeleton-light'
    
    # Header row
    header_cells = [
        html.Div(className=f'{base_class} skeleton-table-cell')
        for _ in range(n_cols)
    ]
    header = html.Div(header_cells, className='skeleton-table-header')
    
    # Data rows
    rows = []
    for _ in range(n_rows):
        cells = [
            html.Div(className=f'{base_class} skeleton-table-cell')
            for _ in range(n_cols)
        ]
        rows.append(html.Div(cells, className='skeleton-table-row'))
    
    return html.Div([header] + rows, className='skeleton-table')


def skeleton_chart(chart_type: str = 'bar', height: int = 200, dark_mode: bool = True):
    """Create a skeleton chart placeholder.
    
    Args:
        chart_type: 'bar', 'line', or 'pie'
        height: Chart height in pixels
        dark_mode: Use dark or light theme
    """
    base_class = 'skeleton' if dark_mode else 'skeleton-light'
    
    if chart_type == 'bar':
        # Create bar chart skeleton
        bars = []
        heights = [70, 45, 85, 55, 90, 40, 75, 60]
        for i, h in enumerate(heights):
            bars.append(html.Div(
                className='skeleton-chart-bar',
                style={
                    'left': f'{10 + i * 11}%',
                    'height': f'{h}%',
                    'animationDelay': f'{i * 0.1}s'
                }
            ))
        
        chart_content = [
            html.Div(className='skeleton-chart-axis-y'),
            html.Div(className='skeleton-chart-axis-x'),
            *bars
        ]
    elif chart_type == 'line':
        # Line chart skeleton with wave effect
        chart_content = [
            html.Div(className='skeleton-chart-axis-y'),
            html.Div(className='skeleton-chart-axis-x'),
            html.Div(
                className=f'{base_class}',
                style={
                    'position': 'absolute',
                    'top': '30%',
                    'left': '50px',
                    'right': '10px',
                    'height': '2px'
                }
            )
        ]
    else:  # pie
        chart_content = [
            html.Div(
                className=f'{base_class}',
                style={
                    'width': f'{height - 40}px',
                    'height': f'{height - 40}px',
                    'borderRadius': '50%',
                    'margin': '20px auto'
                }
            )
        ]
    
    return html.Div(
        chart_content,
        className='skeleton-chart',
        style={'height': f'{height}px'}
    )


def skeleton_metric_card(n_cards: int = 1, dark_mode: bool = True):
    """Create skeleton metric cards for dashboards.
    
    Args:
        n_cards: Number of metric cards
        dark_mode: Use dark or light theme
    """
    base_class = 'skeleton' if dark_mode else 'skeleton-light'
    card_class = 'skeleton-card' if dark_mode else 'skeleton-card-light'
    
    cards = []
    for _ in range(n_cards):
        content = html.Div([
            html.Div(className=f'{base_class} skeleton-metric-label'),
            html.Div(className=f'{base_class} skeleton-metric-value'),
            html.Div(className=f'{base_class} skeleton-metric-change'),
        ], className='skeleton-metric')
        
        cards.append(html.Div(content, className=card_class))
    
    return html.Div(cards, className='d-flex gap-3 flex-wrap')


def skeleton_list_item(n_items: int = 5, with_avatar: bool = True, dark_mode: bool = True):
    """Create skeleton list items.
    
    Args:
        n_items: Number of list items
        with_avatar: Include avatar placeholder
        dark_mode: Use dark or light theme
    """
    base_class = 'skeleton' if dark_mode else 'skeleton-light'
    
    items = []
    for _ in range(n_items):
        content = []
        
        if with_avatar:
            content.append(html.Div(className=f'{base_class} skeleton-avatar'))
        
        text_content = html.Div([
            html.Div(className=f'{base_class} skeleton-text medium'),
            html.Div(className=f'{base_class} skeleton-text short'),
        ], style={'flex': '1'})
        content.append(text_content)
        
        items.append(html.Div(
            content,
            className='d-flex gap-3 align-items-center mb-3'
        ))
    
    return html.Div(items)


def skeleton_page(page_type: str = 'dashboard', dark_mode: bool = True):
    """Create a full page skeleton layout.
    
    Args:
        page_type: 'dashboard', 'table', 'form', or 'detail'
        dark_mode: Use dark or light theme
    """
    if page_type == 'dashboard':
        return html.Div([
            # Top metrics row
            dbc.Row([
                dbc.Col(skeleton_metric_card(1, dark_mode), md=3),
                dbc.Col(skeleton_metric_card(1, dark_mode), md=3),
                dbc.Col(skeleton_metric_card(1, dark_mode), md=3),
                dbc.Col(skeleton_metric_card(1, dark_mode), md=3),
            ], className='mb-4'),
            
            # Charts row
            dbc.Row([
                dbc.Col(skeleton_chart('bar', 250, dark_mode), md=6),
                dbc.Col(skeleton_chart('line', 250, dark_mode), md=6),
            ], className='mb-4'),
            
            # Table
            skeleton_table(5, 5, dark_mode)
        ])
    
    elif page_type == 'table':
        return html.Div([
            # Search/filter bar
            html.Div([
                html.Div(className=f'skeleton skeleton-button'),
                html.Div(className=f'skeleton skeleton-button'),
            ], className='d-flex gap-2 mb-3'),
            
            # Table
            skeleton_table(10, 6, dark_mode)
        ])
    
    elif page_type == 'form':
        return html.Div([
            skeleton_card(1, 0, True, dark_mode),
            html.Div([
                skeleton_text('medium', False, dark_mode),
                html.Div(className=f'skeleton' if dark_mode else 'skeleton-light',
                         style={'height': '38px', 'width': '100%', 'marginBottom': '1rem'}),
            ] * 4, className='mb-3'),
            html.Div(className=f'skeleton skeleton-button')
        ])
    
    else:  # detail
        return html.Div([
            html.Div([
                html.Div(className=f'skeleton skeleton-avatar large'),
                html.Div([
                    skeleton_text('medium', True, dark_mode),
                    skeleton_text('short', False, dark_mode),
                ], style={'flex': '1'})
            ], className='d-flex gap-3 mb-4'),
            
            skeleton_card(1, 5, False, dark_mode),
            skeleton_chart('bar', 200, dark_mode)
        ])


# Export loading wrapper component
def loading_wrapper(loading_component, content_component, is_loading: bool = True):
    """Wrapper to switch between skeleton and actual content.
    
    Args:
        loading_component: Skeleton component to show while loading
        content_component: Actual content to show when loaded
        is_loading: Whether currently loading
    """
    if is_loading:
        return loading_component
    return content_component
