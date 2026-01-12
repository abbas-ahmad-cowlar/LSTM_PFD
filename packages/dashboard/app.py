"""
Main Dash application entry point.
Initializes the Dash app, registers callbacks, and starts the server.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from flask import Flask

from dashboard_config import APP_HOST, APP_PORT, DEBUG
from utils.logger import setup_logger
from database.connection import init_database

# Initialize logger
logger = setup_logger(__name__)

# Initialize Flask server
server = Flask(__name__)

# Configure Flask server for sessions (Phase 11D - Authentication)
from dashboard_config import SECRET_KEY
server.secret_key = SECRET_KEY
server.config['SESSION_COOKIE_HTTPONLY'] = True
server.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# For production with HTTPS, also set:
# server.config['SESSION_COOKIE_SECURE'] = True

# Register API blueprints (Feature #1, #5)
from api.routes import api_bp
from api.api_keys import api_keys_bp
from api.tags import tags_bp
from api.search import search_bp

server.register_blueprint(api_bp)
server.register_blueprint(api_keys_bp)
server.register_blueprint(tags_bp)
server.register_blueprint(search_bp)

# Initialize Dash app
# serve_locally=True serves assets from local packages instead of CDN
# This fixes the issue where CDN is blocked or timing out
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="LSTM PFD Dashboard",
    update_title="Loading...",
    serve_locally=True,  # Critical: serve assets locally to avoid CDN timeouts
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)


# App layout
from components.header import create_header
from components.sidebar import create_sidebar
from components.footer import create_footer

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session-store', storage_type='session'),
    dcc.Store(id='comparison-cart', storage_type='session', data=[]),
    dcc.Interval(id='refresh-interval', interval=5000, n_intervals=0),
    dcc.Interval(id='system-health-interval', interval=5000, n_intervals=0),

    create_header(),

    dbc.Row([
        dbc.Col(create_sidebar(), width=2, className="bg-light"),
        dbc.Col(html.Div(id='page-content'), width=10)
    ]),

    create_footer(),

    # Toast notifications container
    html.Div(id='toast-container', className='position-fixed top-0 end-0 p-3', style={'zIndex': 9999})
], fluid=True)

# Register all callbacks
from callbacks import register_all_callbacks
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
register_all_callbacks(app)

if __name__ == '__main__':
    logger.info("Initializing database...")
    init_database()

    logger.info(f"Starting Dash app on {APP_HOST}:{APP_PORT}")
    # Enable threaded=True to handle concurrent browser requests
    # Without this, the browser's fetch() calls timeout waiting for server
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG, threaded=True)

