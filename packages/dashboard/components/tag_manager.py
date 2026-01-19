"""
Tag Manager Component for Experiment Tagging

Deficiency #48: Tagging UI Polish

Features:
- Tag autocomplete dropdown
- Color picker for tag customization
- Removable tags with X button
- Bulk tagging support

Usage:
    from components.tag_manager import create_tag_input, create_tag_display

    # In layout:
    create_tag_input(id="experiment-tags", existing_tags=["training", "cnn"])
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
from typing import List, Optional


# Available tag colors
TAG_COLORS = [
    {'name': 'blue', 'hex': '#3b82f6'},
    {'name': 'green', 'hex': '#22c55e'},
    {'name': 'yellow', 'hex': '#ca8a04'},
    {'name': 'red', 'hex': '#ef4444'},
    {'name': 'purple', 'hex': '#a855f7'},
    {'name': 'pink', 'hex': '#ec4899'},
    {'name': 'orange', 'hex': '#f97316'},
    {'name': 'cyan', 'hex': '#06b6d4'},
]

# Default suggestions for tag autocomplete
DEFAULT_TAG_SUGGESTIONS = [
    'training', 'evaluation', 'production', 'experiment',
    'cnn', 'transformer', 'pinn', 'ensemble',
    'high-accuracy', 'low-latency', 'baseline',
    'phase-1', 'phase-2', 'phase-3',
    'urgent', 'review', 'archived'
]


def create_single_tag(
    text: str,
    color: str = 'blue',
    removable: bool = False,
    tag_id: Optional[str] = None
) -> html.Span:
    """
    Create a single tag element.
    
    Args:
        text: Tag text
        color: Color variant (blue, green, yellow, red, purple, pink, orange, cyan)
        removable: Show remove X button
        tag_id: Optional ID for callback targeting
    
    Returns:
        html.Span element
    """
    children = [text]
    class_name = f"tag tag-{color}"
    
    if removable:
        class_name += " tag-removable"
        children.append(
            html.Button(
                html.I(className="fas fa-times"),
                className="tag-remove",
                id={"type": "tag-remove", "tag": text} if tag_id is None else tag_id
            )
        )
    
    return html.Span(children, className=class_name, id=tag_id)


def create_tag_display(
    tags: List[dict],
    removable: bool = False
) -> html.Div:
    """
    Create a display of multiple tags.
    
    Args:
        tags: List of tag dicts with 'name' and optional 'color' keys
        removable: Allow removing tags
    
    Returns:
        html.Div containing tags
    """
    tag_elements = []
    for tag in tags:
        name = tag.get('name', tag) if isinstance(tag, dict) else tag
        color = tag.get('color', 'blue') if isinstance(tag, dict) else 'blue'
        tag_elements.append(create_single_tag(name, color, removable))
    
    return html.Div(tag_elements, className="d-flex flex-wrap gap-1")


def create_tag_input(
    input_id: str = "tag-input",
    existing_tags: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    placeholder: str = "Type to add tags..."
) -> html.Div:
    """
    Create a tag input with autocomplete.
    
    Args:
        input_id: ID for the input component
        existing_tags: List of already-selected tags
        suggestions: List of tag suggestions for autocomplete
        placeholder: Placeholder text
    
    Returns:
        html.Div with tag input container
    """
    existing_tags = existing_tags or []
    suggestions = suggestions or DEFAULT_TAG_SUGGESTIONS
    
    # Create tag elements for existing tags
    tag_elements = [
        create_single_tag(tag, 'blue', removable=True)
        for tag in existing_tags
    ]
    
    return html.Div([
        # Hidden store for tag data
        dcc.Store(id=f"{input_id}-store", data=existing_tags),
        dcc.Store(id=f"{input_id}-suggestions", data=suggestions),
        
        # Tag input container
        html.Div([
            # Existing tags
            html.Div(tag_elements, id=f"{input_id}-tags", className="d-flex flex-wrap gap-1"),
            
            # Text input
            dcc.Input(
                id=input_id,
                type="text",
                placeholder=placeholder,
                className="tag-input",
                debounce=True
            ),
        ], className="tag-input-container", id=f"{input_id}-container"),
        
        # Autocomplete dropdown (hidden by default)
        html.Div(
            id=f"{input_id}-autocomplete",
            className="tag-autocomplete",
            style={"display": "none"}
        )
    ], style={"position": "relative"})


def create_color_picker(
    picker_id: str = "tag-color-picker",
    selected_color: str = "blue"
) -> html.Div:
    """
    Create a color picker for tags.
    
    Args:
        picker_id: ID for the picker component
        selected_color: Initially selected color
    
    Returns:
        html.Div with color picker
    """
    color_options = []
    for color in TAG_COLORS:
        is_selected = color['name'] == selected_color
        color_options.append(
            html.Div(
                id={"type": "color-option", "color": color['name']},
                className=f"tag-color-option {'selected' if is_selected else ''}",
                style={"backgroundColor": color['hex']},
                n_clicks=0
            )
        )
    
    return html.Div([
        html.Label("Tag Color", className="form-label small text-muted"),
        html.Div(color_options, className="tag-color-picker"),
        dcc.Store(id=f"{picker_id}-store", data=selected_color)
    ])


def create_tag_filter(
    filter_id: str = "tag-filter",
    available_tags: Optional[List[dict]] = None
) -> html.Div:
    """
    Create a tag filter bar for filtering experiments.
    
    Args:
        filter_id: ID for the filter component
        available_tags: List of available tags to filter by
    
    Returns:
        html.Div with tag filter pills
    """
    available_tags = available_tags or []
    
    if not available_tags:
        # Default sample tags
        available_tags = [
            {'name': 'training', 'color': 'blue'},
            {'name': 'evaluation', 'color': 'green'},
            {'name': 'production', 'color': 'purple'},
        ]
    
    filter_pills = [
        html.Span(
            tag['name'],
            className=f"tag tag-{tag.get('color', 'blue')} tag-filter",
            id={"type": "tag-filter", "tag": tag['name']},
            n_clicks=0
        )
        for tag in available_tags
    ]
    
    return html.Div([
        html.Label("Filter by Tags:", className="me-2 text-muted small"),
        html.Div(filter_pills, className="tag-filter-container")
    ], className="d-flex align-items-center flex-wrap")


def create_bulk_tag_bar(
    bar_id: str = "bulk-tag-bar",
    selected_count: int = 0
) -> html.Div:
    """
    Create a bulk tagging action bar.
    
    Args:
        bar_id: ID for the bar component
        selected_count: Number of selected items
    
    Returns:
        html.Div with bulk tag controls
    """
    return html.Div([
        html.Span([
            html.Strong(str(selected_count), className="selected-count"),
            " experiments selected"
        ]),
        
        # Tag dropdown for bulk add
        dbc.DropdownMenu(
            label="Add Tag",
            children=[
                dbc.DropdownMenuItem(tag, id={"type": "bulk-add-tag", "tag": tag})
                for tag in DEFAULT_TAG_SUGGESTIONS[:8]
            ],
            size="sm",
            color="primary"
        ),
        
        # Remove tag dropdown
        dbc.DropdownMenu(
            label="Remove Tag",
            children=[
                dbc.DropdownMenuItem(tag, id={"type": "bulk-remove-tag", "tag": tag})
                for tag in DEFAULT_TAG_SUGGESTIONS[:8]
            ],
            size="sm",
            color="secondary"
        ),
        
        # Clear selection
        dbc.Button(
            "Clear Selection",
            id=f"{bar_id}-clear",
            size="sm",
            outline=True,
            color="secondary"
        )
    ], id=bar_id, className="bulk-tag-bar", 
       style={"display": "flex" if selected_count > 0 else "none"})
