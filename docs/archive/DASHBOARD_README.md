# LSTM PFD Dashboard

**Enterprise-grade Plotly Dash application for the LSTM PFD bearing fault diagnosis system.**

![Dashboard](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/Plotly-Dash-purple)
![Coverage](https://img.shields.io/badge/Coverage-90%25%2B-brightgreen)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Pages & Functionality](#-pages--functionality)
- [Development](#-development)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)

---

## 🎯 Overview

The **LSTM PFD Dashboard** is a comprehensive web application that provides an intuitive, production-ready interface for the entire LSTM PFD bearing fault diagnosis system. Built with **Plotly Dash**, it integrates all phases (0-10) into a unified platform for:

- 🔬 **Interactive Data Exploration** - Visualize and filter bearing fault datasets
- 🚀 **ML Experiment Management** - Configure, launch, and monitor training experiments
- 📊 **Real-time Monitoring** - Live training progress with loss/accuracy curves
- 🔍 **Explainable AI** - SHAP, LIME, Integrated Gradients, Grad-CAM explanations
- 🎯 **Hyperparameter Optimization** - Automated HPO campaigns with Bayesian optimization
- 📈 **Advanced Analytics** - Statistical model comparison and per-class analysis
- 🔐 **Production Security** - JWT authentication, rate limiting, security headers
- 📡 **System Monitoring** - Real-time resource usage and health monitoring

### Why Use the Dashboard?

- **No Code Required**: Run experiments without writing Python scripts
- **Real-time Feedback**: Monitor training progress live without SSH/terminal access
- **Explainability**: Understand why models make predictions
- **Comparison**: Easily compare multiple models side-by-side
- **Reproducibility**: All experiments tracked with full configuration history
- **Production Ready**: Scales horizontally, includes auth, monitoring, and alerts

---

## ✨ Features

### Phase 11A: Foundation & Data Exploration ✅

**Core Infrastructure:**
- **PostgreSQL database** - Persistent storage for experiments, datasets, and metadata
- **Redis caching** - Performance optimization for expensive computations (t-SNE, spectrograms)
- **Celery task queue** - Background job processing for long-running training tasks
- **RESTful API** - `/api/*` endpoints for programmatic access

**5 Core Pages:**
1. **Home Dashboard** - System overview, quick stats, recent experiments, health gauges
2. **Data Explorer** - Dataset visualization, filtering, t-SNE embeddings, export
3. **Signal Viewer** - Time/frequency/spectrogram analysis with interactive plots
4. **Dataset Manager** - Upload, version, and manage datasets
5. **System Health** - Real-time CPU/memory/disk monitoring with alerts

### Phase 11B: ML Pipeline Orchestration ✅

**Experiment Workflow:**
- **Configuration Wizard** - Multi-step wizard for model selection and hyperparameters
  - 20+ model architectures (Classical ML, 1D CNN, ResNet, EfficientNet, Transformer, PINN, Ensemble)
  - Dataset selection and train/val/test splits
  - Optimizer, scheduler, and augmentation options
  - Early stopping and checkpointing configuration

- **Real-time Training Monitoring** - Live updates without page refresh
  - Progress bars with ETA
  - Loss and accuracy curves (train/val)
  - Learning rate schedule visualization
  - Gradient norms monitoring
  - Scrollable training logs
  - Pause/resume/stop controls

- **Results Visualization** - Comprehensive post-training analysis
  - Confusion matrix heatmap
  - Per-class precision, recall, F1-score table
  - Training history plots
  - Model configuration and hyperparameters
  - Export options (PDF report, model weights, logs)

- **Experiment History** - Searchable, filterable experiment database
  - Sort by accuracy, duration, timestamp
  - Multi-select for comparison (2-5 experiments)
  - Status indicators (running, completed, failed)
  - Tag-based organization

- **Model Comparison** - Side-by-side statistical comparison
  - Accuracy and training time charts
  - Statistical significance tests (t-test, ANOVA)
  - Per-class radar charts
  - Export comparison reports

### Phase 11C: Advanced Analytics & XAI ✅

**Explainable AI Dashboard:**
- **SHAP Explanations** - Feature attribution using game theory
  - Signal attribution overlay
  - Feature importance bar charts
  - Summary statistics (mean |SHAP value|, max contribution)

- **LIME Explanations** - Local interpretable explanations
  - Ranked feature importance table
  - Local linear approximation
  - Confidence scores (R²)

- **Integrated Gradients** - Neural network attribution
  - Gradient-based attribution maps
  - Baseline comparison (zero signal)
  - Target class visualization

- **Grad-CAM** - CNN activation visualization
  - Activation heatmaps overlaid on signals
  - Layer-specific analysis
  - Class activation contributions

**Hyperparameter Optimization:**
- **Multiple Strategies** - Bayesian optimization, random search, grid search, Hyperband
- **Search Space Definition** - Flexible parameter ranges and distributions
- **Campaign Monitoring** - Real-time progress and best score tracking
- **Results Analysis** - Parallel coordinates plot, parameter importance, export

**Statistical Analysis:**
- Multi-model comparison with significance tests
- Per-class performance breakdown
- Confidence intervals and error bars
- Export-ready visualizations

### Phase 11D: Production Hardening ✅

**Security:**
- **JWT Authentication** - Secure token-based auth (24-hour expiry)
- **Rate Limiting** - 60 requests/minute per IP, customizable per endpoint
- **Security Headers** - XSS, CSP, HSTS, clickjacking protection
- **Input Sanitization** - Prevents injection attacks
- **File Upload Validation** - Extension whitelist, path traversal prevention

**Monitoring:**
- **System Metrics** - CPU, memory, disk usage with alerts
- **Application Metrics** - Experiment counts, failure rates
- **Alert System** - Threshold-based alerts with deduplication
- **Health Check Endpoint** - `/api/health` for load balancers

**Testing:**
- **90%+ Test Coverage** - Comprehensive unit and integration tests
- **40+ Tests** - Authentication, rate limiting, database, experiments, security
- **CI/CD Integration** - Automated testing on push/PR
- **Benchmark Suite** - Performance regression tests

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
packages/dashboard/
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

## 📚 Documentation

### 🎯 Choose Your Path

**For GUI Users (No Coding Required):**
- **[GUI Quick Start Guide](../GUI_QUICKSTART.md)** - 30-minute tutorial to train your first 96% accurate model using only the web dashboard

**For Dashboard Power Users:**
- **[Phase 11 Complete Usage Guide](../docs/USAGE_PHASE_11.md)** - Comprehensive 850+ line guide covering all dashboard features

**For Understanding Feature Coverage:**
- **[Dashboard Gaps Analysis](../docs/DASHBOARD_GAPS.md)** - Complete analysis of what's accessible via GUI vs CLI (50% feature coverage)

### Complete Usage Guide Details

The **[Phase 11 Usage Guide](../docs/USAGE_PHASE_11.md)** covers:
- Detailed setup instructions for all deployment scenarios
- Complete walkthrough of every page and feature
- Data generation (synthetic + MAT file import)
- XAI dashboard tutorials (SHAP, LIME, IG, Grad-CAM)
- Feature status matrix (what's available vs what's planned)
- Troubleshooting and performance optimization

### Additional Resources

- **[Main Project README](../README.md)** - Overall LSTM PFD system documentation
- **[CLI Quick Start](../QUICKSTART.md)** - Command-line workflow guide
- **[API Reference](./api/README.md)** - REST API endpoints documentation (if available)
- **[Database Schema](./database/README.md)** - PostgreSQL database structure (if available)
- **[Contributing Guide](../CONTRIBUTING.md)** - Development guidelines (if available)

---

## 📊 Pages & Functionality

### 1. Home Dashboard (`/`)
- System overview with quick stats
- Recent experiments timeline
- System health gauges (CPU, memory, disk)
- Dataset distribution visualization
- Quick action buttons to key features

### 2. Data Explorer (`/data-explorer`)
- Interactive dataset browser
- Filter by fault type, severity, conditions
- t-SNE visualization of signal embeddings
- Statistical summaries per class
- Export filtered datasets

### 3. Signal Viewer (`/signal-viewer`)
- Time-domain waveform plotting
- FFT frequency spectrum analysis
- Spectrogram generation (STFT, CWT, WVD)
- Feature extraction and display
- Zoom, pan, and export capabilities

### 4. Experiments (`/experiments`)
- **Experiment List** - View all experiments with search/filter
- **New Experiment** (`/experiment/new`) - Configuration wizard
- **Monitor Training** (`/experiment/<id>/monitor`) - Real-time progress
- **View Results** (`/experiment/<id>/results`) - Post-training analysis
- **Compare Models** - Multi-experiment comparison

### 5. XAI Dashboard (`/xai`)
- Select model and signal
- Generate SHAP, LIME, IG, or Grad-CAM explanations
- Interactive visualizations
- Export explanations as images/JSON

### 6. HPO Campaigns (`/hpo`)
- Create optimization campaigns
- Monitor trial progress
- View best hyperparameters
- Parallel coordinates visualization

### 7. System Health (`/health`)
- Real-time resource monitoring
- Alert history and status
- Database connection status
- Redis cache statistics
- Active Celery tasks

---

## 🎯 Use Cases

### For Data Scientists
- **Rapid Experimentation**: Test 20+ models without writing code
- **Hyperparameter Tuning**: Automated Bayesian optimization
- **Model Interpretation**: SHAP and LIME explanations
- **Result Tracking**: All experiments logged with full configuration

### For ML Engineers
- **Production Deployment**: Docker-ready with auth and monitoring
- **Scalability**: Horizontal scaling with load balancers
- **Monitoring**: Real-time system health and alerts
- **API Access**: REST endpoints for programmatic control

### For Domain Experts
- **No Coding Required**: Web UI for all operations
- **Explainable Predictions**: Understand why models make decisions
- **Signal Analysis**: Interactive time/frequency domain tools
- **Report Generation**: Export PDF reports for stakeholders

### For Researchers
- **Reproducibility**: Full experiment configuration tracking
- **Comparison**: Statistical significance testing across models
- **Visualization**: Publication-ready plots
- **Export**: Results in CSV/JSON/PDF formats

---

## 🔧 Configuration

### Environment Variables

**Required:**
```bash
DATABASE_URL=postgresql://user:password@host:5432/dbname
REDIS_URL=redis://host:6379/0
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
```

**Optional:**
```bash
DEBUG=False                    # Enable debug mode (default: False)
APP_HOST=0.0.0.0              # Host to bind (default: 0.0.0.0)
APP_PORT=8050                  # Port to bind (default: 8050)
LOG_LEVEL=INFO                 # Logging level (default: INFO)
MAX_UPLOAD_SIZE=100            # Max upload size in MB (default: 100)
RATE_LIMIT_PER_MINUTE=60       # API rate limit (default: 60)
```

### Database Configuration

**Connection Pooling:**
```python
# In config.py
DATABASE_CONFIG = {
    'pool_size': 10,           # Base connection pool size
    'max_overflow': 20,        # Max extra connections
    'pool_timeout': 30,        # Seconds to wait for connection
    'pool_recycle': 3600,      # Recycle connections after 1 hour
}
```

### Redis Configuration

**Cache TTLs:**
```python
CACHE_CONFIG = {
    'tsne_embeddings': 3600,    # 1 hour
    'spectrograms': 21600,      # 6 hours
    'shap_explanations': 86400, # 24 hours
    'session_data': 3600,       # 1 hour
}
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html

# Run specific test file
pytest tests/test_integration.py -v

# Run tests matching pattern
pytest -k "test_auth" -v
```

### Test Structure

```
tests/
├── test_auth.py              # Authentication tests (8 tests)
├── test_rate_limiting.py     # Rate limiting tests (4 tests)
├── test_database.py          # Database model tests (10 tests)
├── test_experiments.py       # Experiment workflow tests (6 tests)
├── test_monitoring.py        # Monitoring service tests (5 tests)
├── test_security.py          # Security middleware tests (7 tests)
└── test_integration.py       # End-to-end integration tests
```

### Coverage Report

```
Name                          Stmts   Miss  Cover
-------------------------------------------------
api/routes.py                   124      8    94%
callbacks/__init__.py           89      5    94%
database/connection.py          45      2    96%
middleware/auth.py              112      9    92%
middleware/security.py          78      6    92%
services/monitoring_service.py  95      8    92%
models/experiment.py            67      3    96%
-------------------------------------------------
TOTAL                          1247    89    93%
```

---

## 🚨 Troubleshooting

### Common Issues

**Issue: Database connection errors**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection manually
psql -h localhost -U lstm_user -d lstm_pfd
```

**Issue: Redis connection errors**
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG

# Check Redis memory usage
docker-compose exec redis redis-cli info memory
```

**Issue: Celery tasks not starting**
```bash
# Check worker status
docker-compose logs celery_worker

# Restart worker
docker-compose restart celery_worker

# Purge all queued tasks
celery -A tasks.celery_app purge
```

**Issue: Page loading slowly**
- Clear Redis cache: `docker-compose exec redis redis-cli FLUSHALL`
- Check database query performance: `SELECT * FROM pg_stat_statements`
- Enable query logging in PostgreSQL
- Check CPU/memory usage on server

**Issue: Training stuck at 0%**
- Verify dataset exists: Check `/data-explorer`
- Check Celery worker logs: `docker-compose logs celery_worker`
- Verify GPU is available: `docker-compose exec dash_app python -c "import torch; print(torch.cuda.is_available())"`

---

## 📈 Performance Optimization

### Database Optimization

**Indexes:**
```sql
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_model_type ON experiments(model_type);
CREATE INDEX idx_training_runs_experiment_id ON training_runs(experiment_id);
```

**Vacuuming:**
```bash
docker-compose exec postgres psql -U lstm_user -d lstm_pfd -c "VACUUM ANALYZE;"
```

### Redis Optimization

**Memory Configuration:**
```bash
# Set max memory in docker-compose.yml
redis:
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### Application Optimization

**Dash Performance:**
```python
# In app.py
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# Enable compression
server.config['COMPRESS_MIMETYPES'] = [
    'text/html', 'text/css', 'application/javascript', 'application/json'
]
```

---

## 🌐 API Reference

### Authentication

**POST `/api/auth/login`**
```bash
curl -X POST http://localhost:8050/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

**Response:**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400
}
```

### Experiments

**GET `/api/experiments`** - List all experiments
**POST `/api/experiments`** - Create new experiment
**GET `/api/experiments/<id>`** - Get experiment details
**GET `/api/experiments/<id>/progress`** - Get training progress
**POST `/api/experiments/<id>/stop`** - Stop training

### Health

**GET `/api/health`** - System health check

---

## 🎨 Customization

### Adding Custom Pages

1. Create layout in `layouts/my_page.py`
2. Create callbacks in `callbacks/my_page_callbacks.py`
3. Register in `callbacks/__init__.py`
4. Add navigation link in `components/sidebar.py`

### Adding Custom Models

1. Implement model in `../models/`
2. Create adapter in `integrations/my_model_adapter.py`
3. Register in model factory
4. Add to experiment wizard dropdown

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abbas-ahmad-cowlar/LSTM_PFD/discussions)
- **Documentation**: [Complete Usage Guide](../docs/USAGE_PHASE_11.md)

---

## 📝 Status

- ✅ Phase 11A: Foundation & Data Exploration (Complete)
- ✅ Phase 11B: ML Pipeline Orchestration (Complete)
- ✅ Phase 11C: Advanced Analytics & XAI (Complete)
- ✅ Phase 11D: Production Hardening (Complete)

**Development Status**: 🎉 **PRODUCTION READY**

**Last Updated**: November 2025

**Test Coverage**: 93%

**Lines of Code**: 8,500+

---

<div align="center">

### 🚀 Ready to Get Started?

[Quick Start](#-quick-start) | [GUI Quick Start](../GUI_QUICKSTART.md) | [Complete Usage Guide](../docs/USAGE_PHASE_11.md) | [Main Project](../README.md)

**Built with ❤️ for Enterprise ML Operations**

</div>
