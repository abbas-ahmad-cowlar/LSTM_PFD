# Implementation Plan: API Monitoring, Enhanced Evaluation, Testing & QA

**Date**: 2025-11-22
**Phase**: Production Completeness (Phase 2)
**Estimated Total**: 7 days

---

## 📊 Codebase Analysis Summary

### Existing Components Identified

#### 1. API Infrastructure (api/)
- ✅ **FastAPI Application** (`api/main.py`) - Complete REST API with endpoints
- ✅ **Prediction Endpoints**: `/predict`, `/predict/batch`
- ✅ **Health Check**: `/health`
- ✅ **Model Info**: `/model/info`
- ✅ **Logging**: Python logging to file
- ❌ **No request/response logging to database**
- ❌ **No metrics tracking (latency, throughput, errors)**
- ❌ **No API key management UI**

#### 2. Evaluation Tools (evaluation/)
- ✅ **ROC Analysis** (`roc_analyzer.py`) - ROC curves, AUC scores per class
- ✅ **Error Analysis** (`error_analysis.py`) - Misclassification analysis, confusion patterns
- ✅ **Architecture Comparison** (`architecture_comparison.py`) - FLOPs, params, Pareto frontier
- ✅ **Ensemble Evaluator** (`ensemble_evaluator.py`) - Ensemble metrics
- ✅ **Robustness Tester** (`robustness_tester.py`) - Noise/adversarial testing
- ❌ **No dashboard integration**

#### 3. Testing Infrastructure (tests/)
- ✅ **Unit Tests** (`tests/unit/`) - Feature extraction, deployment, API tests
- ✅ **Integration Tests** (`tests/integration/`) - Pipeline tests
- ✅ **Benchmark Suite** (`tests/benchmarks/benchmark_suite.py`) - Performance benchmarks
- ✅ **pytest framework** with fixtures
- ❌ **No test execution UI**
- ❌ **No coverage reporting UI**

### Database Models Available
- ✅ `api_key.py` - API key model exists but not used
- ✅ `system_log.py` - Can store API logs
- ✅ `experiment.py` - Has metrics, can be used for evaluation
- ✅ `training_run.py` - Epoch-level metrics

---

## 🎯 Feature 1: API Monitoring Dashboard (2 days)

### Architecture Design

#### Database Model (New)
Create `packages/dashboard/models/api_request_log.py`:
```python
class APIRequestLog(BaseModel):
    endpoint: str  # /predict, /predict/batch
    method: str  # GET, POST
    status_code: int  # 200, 400, 500
    request_time: DateTime
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    ip_address: str
    user_agent: str
    api_key_id: int (FK to api_keys)
    error_message: str (nullable)
    request_payload: JSON (sample)
    response_payload: JSON (sample)
```

#### Service Layer
`packages/dashboard/services/api_monitoring_service.py`:
- `log_api_request()` - Log request/response
- `get_request_stats()` - Aggregate statistics (last hour, day, week)
- `get_endpoint_metrics()` - Per-endpoint breakdown
- `get_error_rate()` - Error rate over time
- `get_latency_percentiles()` - P50, P95, P99
- `get_top_api_keys()` - Most active API keys

#### API Middleware (Modify api/main.py)
Add middleware to log all requests:
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Log to database (async)
    await log_api_request_to_db(request, response, duration_ms)

    return response
```

#### Layout (`packages/dashboard/layouts/api_dashboard.py`)
**Components**:
1. **Overview Cards** - Total requests, avg latency, error rate, active API keys
2. **Request Timeline** - Requests per minute (last 24h)
3. **Endpoint Breakdown** - Table with stats per endpoint
4. **Latency Distribution** - Histogram + percentiles
5. **Error Log** - Recent errors with details
6. **API Key Management** - Create, revoke, view usage

#### Callbacks (`packages/dashboard/callbacks/api_callbacks.py`)
- `update_api_metrics()` - Real-time metrics (auto-refresh 10s)
- `update_request_timeline()` - Timeline chart
- `update_endpoint_table()` - Endpoint statistics
- `show_error_details()` - Modal with error info
- `create_api_key()` - Generate new API key
- `revoke_api_key()` - Deactivate API key

### Implementation Steps
1. Create database model + migration
2. Add logging middleware to API
3. Create service layer
4. Create layout with charts
5. Wire up callbacks
6. Add route `/api-monitoring`
7. Test with sample requests

---

## 🎯 Feature 2: Enhanced Evaluation Dashboard (3 days)

### Architecture Design

#### Service Layer
`packages/dashboard/services/evaluation_service.py`:
- `generate_roc_curves()` - Generate ROC data for experiment
- `compute_error_analysis()` - Confusion patterns, misclassified samples
- `compare_architectures()` - FLOPs vs Accuracy Pareto frontier
- `test_robustness()` - Noise/adversarial evaluation
- `cache_evaluation_results()` - Cache expensive computations

#### Layout (`packages/dashboard/layouts/evaluation_dashboard.py`)
**Tabs**:
1. **ROC Analysis Tab**
   - Multi-class ROC curves (one-vs-rest)
   - AUC scores table
   - Macro/Micro averaged metrics
   - Interactive class selection

2. **Error Analysis Tab**
   - Confusion heatmap (enhanced)
   - Top confused pairs table
   - Misclassification samples viewer
   - Confidence distribution for errors

3. **Architecture Comparison Tab**
   - Accuracy vs FLOPs scatter plot
   - Accuracy vs Parameters scatter plot
   - Pareto frontier highlighting
   - Model selection table with metrics

4. **Robustness Testing Tab**
   - SNR vs Accuracy plot
   - Adversarial attack results
   - Robustness score per class

#### Enhance Existing Results Page
Modify `packages/dashboard/layouts/experiment_results.py`:
- Add "Advanced Evaluation" button
- Link to evaluation dashboard with experiment_id

#### Callbacks (`packages/dashboard/callbacks/evaluation_callbacks.py`)
- `generate_roc_curves_callback()` - Trigger ROC generation (Celery task)
- `display_roc_curves()` - Show ROC plots
- `analyze_errors_callback()` - Trigger error analysis (Celery task)
- `display_error_analysis()` - Show confusion patterns
- `compare_models_callback()` - Load multiple experiments for comparison
- `display_pareto_frontier()` - Architecture comparison plot

#### Celery Tasks (`packages/dashboard/tasks/evaluation_tasks.py`)
- `generate_roc_task()` - Compute ROC curves (CPU intensive)
- `error_analysis_task()` - Deep error analysis
- `robustness_test_task()` - Run robustness tests
- `architecture_comparison_task()` - Compare multiple architectures

### Implementation Steps
1. Create service layer integrating evaluation/
2. Create Celery tasks for heavy computations
3. Create layouts with interactive charts
4. Wire up callbacks
5. Add route `/evaluation`
6. Enhance experiment results page
7. Test with real experiments

---

## 🎯 Feature 3: Testing & QA Dashboard (2 days)

### Architecture Design

#### Service Layer
`packages/dashboard/services/testing_service.py`:
- `run_tests()` - Execute pytest via subprocess
- `get_test_results()` - Parse pytest output
- `get_coverage_report()` - Parse coverage.py XML
- `get_benchmark_results()` - Load benchmark JSON
- `compare_benchmark_history()` - Track performance over time

#### Database Model (Optional)
`packages/dashboard/models/test_run.py`:
```python
class TestRun(BaseModel):
    run_id: str
    timestamp: DateTime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    coverage_percent: float
    test_results: JSON  # Detailed results
```

#### Layout (`packages/dashboard/layouts/testing_dashboard.py`)
**Sections**:
1. **Test Execution Panel**
   - Test suite selector (unit/integration/all)
   - Run Tests button
   - Real-time output stream
   - Progress indicator

2. **Test Results**
   - Summary cards (passed, failed, skipped)
   - Test results table with filters
   - Failed test details with traceback

3. **Coverage Report**
   - Coverage percentage gauge
   - Coverage by module (bar chart)
   - Uncovered lines report

4. **Benchmark Results**
   - Benchmark results table
   - Performance trends (line chart)
   - Comparison with baseline

#### Callbacks (`packages/dashboard/callbacks/testing_callbacks.py`)
- `run_tests_callback()` - Execute tests (Celery task)
- `stream_test_output()` - Show live test output
- `display_test_results()` - Show results table
- `display_coverage()` - Coverage visualization
- `display_benchmarks()` - Benchmark charts

#### Celery Tasks (`packages/dashboard/tasks/testing_tasks.py`)
- `run_pytest_task()` - Execute pytest with coverage
- `run_benchmark_task()` - Execute benchmark suite
- Stream output to Redis for live updates

### Implementation Steps
1. Create service layer for test execution
2. Create Celery tasks with output streaming
3. Create layout with test controls
4. Wire up callbacks for real-time updates
5. Add route `/testing`
6. Test execution and result display
7. Add coverage parsing

---

## 🔧 Technical Implementation Details

### Safety Measures

#### 1. Database Migrations
- Create migrations for new models
- Test rollback procedures
- No data loss on existing tables

#### 2. Backward Compatibility
- All new features are additive
- Existing routes unchanged
- Optional middleware (can be disabled)

#### 3. Error Handling
```python
# Pattern for all callbacks
@app.callback(...)
def callback(...):
    try:
        # Implementation
        return success_component
    except Exception as e:
        logger.error(f"Callback failed: {e}", exc_info=True)
        return dbc.Alert(f"Error: {str(e)}", color="danger")
```

#### 4. Performance
- Cache evaluation results (Redis)
- Pagination for large datasets
- Lazy loading for charts
- Background tasks for heavy computations

#### 5. Testing Before Commit
- Import all new modules
- Test each route
- Verify database migrations
- Check callback registration

---

## 📅 Implementation Timeline

### Day 1-2: API Monitoring
- **Day 1 Morning**: Database model, migration, middleware
- **Day 1 Afternoon**: Service layer, basic layout
- **Day 2 Morning**: Callbacks, charts, API key management
- **Day 2 Afternoon**: Testing, integration, polish

### Day 3-5: Enhanced Evaluation
- **Day 3 Morning**: Service layer, ROC integration
- **Day 3 Afternoon**: ROC layout and callbacks
- **Day 4 Morning**: Error analysis integration
- **Day 4 Afternoon**: Architecture comparison
- **Day 5 Morning**: Robustness testing, polish
- **Day 5 Afternoon**: Testing, integration

### Day 6-7: Testing & QA
- **Day 6 Morning**: Test execution service, tasks
- **Day 6 Afternoon**: Layout, test runner
- **Day 7 Morning**: Coverage integration, benchmarks
- **Day 7 Afternoon**: Testing, polish, commit

---

## 🎨 UI/UX Design Principles

### Consistent Patterns
- Use existing card/alert components
- Match XAI dashboard style
- Bootstrap color scheme
- FontAwesome icons

### Real-time Updates
- Auto-refresh intervals (10-30s)
- Celery task progress bars
- Live streaming for test output

### Professional Quality
- Loading spinners during async operations
- Graceful error messages
- Tooltips for metrics
- Export capabilities (CSV, JSON, PDF)

---

## ✅ Quality Checklist

### Before Commit
- [ ] All imports work
- [ ] All routes registered
- [ ] All callbacks registered
- [ ] Database migrations created and tested
- [ ] No existing functionality broken
- [ ] Error handling in all callbacks
- [ ] Logging added for debugging
- [ ] Comments and docstrings
- [ ] Consistent code style
- [ ] Git commit with detailed message

### Testing
- [ ] Navigate to each new route
- [ ] Trigger each callback
- [ ] Test error scenarios
- [ ] Check database writes
- [ ] Verify caching works
- [ ] Test with real data

---

## 🚀 Ready to Implement

This plan ensures:
- ✅ Professional, production-ready code
- ✅ No breaking changes
- ✅ Comprehensive error handling
- ✅ Real-time, responsive UIs
- ✅ Integration with existing codebase
- ✅ Scalable architecture

Starting implementation now...
