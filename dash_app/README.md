# LSTM PFD Dashboard

Enterprise-grade Plotly Dash application for the LSTM PFD bearing fault diagnosis system.

## Features

### Phase 11A: Foundation & Data Exploration ✅
- **PostgreSQL database** for experiment metadata
- **Redis caching** for performance optimization
- **Celery task queue** for background jobs
- **5 core pages**:
  - Home Dashboard
  - Data Explorer (dataset visualization & filtering)
  - Signal Viewer (time/frequency/spectrogram analysis)
  - Dataset Manager
  - System Health monitoring

### Phase 11B: ML Pipeline Orchestration ✅
- **Training experiment configuration wizard** - Multi-step wizard for model selection, hyperparameters, and training options
- **Real-time training progress monitoring** - Live updates with loss/accuracy curves, LR schedule, and logs
- **Comprehensive results visualization** - Confusion matrix, per-class metrics, and training history
- **Experiment history and comparison** - Searchable, filterable table with multi-select comparison
- **Integration with Phases 1-8 training code** - Full adapters for classical ML, CNNs, Transformers, PINN, Ensemble

### Phase 11C: Advanced Analytics & XAI ✅
- **Explainable AI integration** - Interactive dashboard with SHAP, LIME, Integrated Gradients, and Grad-CAM
- **Hyperparameter optimization campaigns** - Bayesian optimization, random search, grid search, and Hyperband
- **Statistical model comparison** - Side-by-side metrics, significance tests, and per-class radar charts
- **Multi-signal comparison tools** - Batch signal analysis and comparison
- **Advanced analytics dashboard** - Comprehensive analytics with export options

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (recommended)
- PostgreSQL 15+ (if not using Docker)
- Redis 7+ (if not using Docker)

### Installation

#### Option 1: Docker Compose (Recommended)

```bash
cd dash_app
cp .env.example .env
# Edit .env if needed
docker-compose up
```

Access the dashboard at http://localhost:8050

#### Option 2: Local Development

```bash
# 1. Install dependencies
cd dash_app
pip install -r requirements.txt

# 2. Start PostgreSQL and Redis
# (Install and start separately)

# 3. Set environment variables
cp .env.example .env
source .env

# 4. Initialize database
python -c "from database.connection import init_database; from database.seed_data import seed_initial_data; init_database(); seed_initial_data()"

# 5. Start Celery worker (in separate terminal)
celery -A tasks.celery_app worker --loglevel=info

# 6. Start Dash app
python app.py
```

## Architecture

```
dash_app/
├── app.py                 # Main application entry point
├── config.py              # Configuration management
├── assets/                # Static CSS/JS assets
├── components/            # Reusable UI components
│   ├── header.py
│   ├── sidebar.py
│   └── cards.py
├── layouts/               # Page layouts
│   ├── home.py
│   ├── data_explorer.py
│   ├── signal_viewer.py
│   └── experiments.py
├── callbacks/             # Dash callbacks (event handlers)
│   ├── __init__.py
│   ├── data_explorer_callbacks.py
│   └── signal_viewer_callbacks.py
├── services/              # Business logic layer
│   ├── cache_service.py
│   ├── data_service.py
│   ├── signal_service.py
│   └── notification_service.py
├── models/                # SQLAlchemy database models
│   ├── base.py
│   ├── dataset.py
│   ├── experiment.py
│   └── ...
├── database/              # Database connection & migrations
│   ├── connection.py
│   └── seed_data.py
├── integrations/          # Phase 0-10 integration adapters
│   └── phase0_adapter.py
├── tasks/                 # Celery background tasks
│   ├── __init__.py
│   └── training_tasks.py
├── api/                   # REST API endpoints
│   └── routes.py
└── utils/                 # Utilities
    ├── logger.py
    ├── constants.py
    ├── formatting.py
    └── plotting.py
```

## Key Design Principles

### 1. Three-Layer Architecture
- **Presentation Layer**: Dash layouts & callbacks (UI only)
- **Service Layer**: Business logic (testable independently)
- **Data Layer**: Database & file storage

### 2. Phase 0-10 Integration
- **Adapter Pattern**: Wraps existing code without modification
- **No Duplication**: Dashboard orchestrates, doesn't reimplement
- **Progress Callbacks**: Injected for real-time monitoring

### 3. Caching Strategy
- **Redis**: Expensive computations (t-SNE, spectrograms)
- **Session Storage**: User preferences, filter selections
- **Database**: Persistent experiment data

### 4. Asynchronous Tasks
- **Celery**: Background training jobs (15-45 minutes)
- **Progress Tracking**: Real-time updates via Redis
- **Graceful Handling**: User can close browser, training continues

## Database Schema

Key tables:
- `datasets`: Dataset metadata
- `signals`: Signal records
- `experiments`: Training experiments
- `training_runs`: Per-epoch metrics
- `hpo_campaigns`: Hyperparameter optimization campaigns
- `explanations`: Cached XAI explanations

## Environment Variables

See `.env.example` for all configuration options:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `DEBUG`: Enable debug mode
- `SECRET_KEY`: Flask secret key (change in production!)

## Development

### Running Tests

```bash
pytest -v
pytest --cov=. --cov-report=html  # With coverage
```

### Adding a New Page

1. Create layout in `layouts/my_page.py`:
```python
def create_my_page_layout():
    return html.Div([...])
```

2. Create callbacks in `callbacks/my_page_callbacks.py`:
```python
def register_my_page_callbacks(app):
    @app.callback(...)
    def my_callback(...):
        ...
```

3. Register in `callbacks/__init__.py`:
```python
from callbacks.my_page_callbacks import register_my_page_callbacks
register_my_page_callbacks(app)
```

4. Add route in `callbacks/__init__.py` display_page function:
```python
elif pathname == '/my-page':
    from layouts.my_page import create_my_page_layout
    return create_my_page_layout()
```

## Production Deployment

### Performance Targets
- Page load: <2 seconds
- Filter response: <500ms
- Signal load: <1 second
- Training task spawn: <1 second

### Scaling
- **Horizontal**: Run multiple Dash app instances behind load balancer
- **Celery Workers**: Scale based on GPU availability
- **Database**: Connection pooling (10 connections per app instance)
- **Redis**: Single instance sufficient for <100 users

### Security Checklist
- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS (reverse proxy with nginx/Caddy)
- [ ] Enable CORS restrictions
- [ ] Add authentication (Phase 11D)
- [ ] Sanitize user inputs
- [ ] Rate limiting on API endpoints

## Troubleshooting

**Database connection errors:**
```bash
# Check PostgreSQL is running
docker-compose ps
# View logs
docker-compose logs postgres
```

**Redis connection errors:**
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG
```

**Celery tasks not running:**
```bash
# Check Celery worker is running
docker-compose logs celery_worker
# Restart worker
docker-compose restart celery_worker
```

**Page not loading:**
- Check browser console for errors
- Check `app.log` for backend errors
- Verify all callbacks are registered

## Contributing

See `../CONTRIBUTING.md` for development guidelines.

## License

MIT License - see `../LICENSE`

## Status

- ✅ Phase 11A: Foundation & Data Exploration (Complete)
- ✅ Phase 11B: ML Pipeline Orchestration (Complete)
- ✅ Phase 11C: Advanced Analytics & XAI (Complete)
- ✅ Phase 11D: Production Hardening (Complete)

**Last Updated**: November 2025

**🎉 ALL PHASES COMPLETE - PRODUCTION READY!**
