"""
Testing & QA Dashboard callbacks.
Handle test execution, coverage, and benchmarks.
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from tasks.testing_tasks import (
    run_tests_task,
    run_coverage_task,
    run_benchmarks_task,
    run_quality_checks_task
)
from utils.logger import setup_logger
import plotly.graph_objs as go

logger = setup_logger(__name__)


def register_testing_callbacks(app):
    """Register testing & QA callbacks."""

    @app.callback(
        [Output('test-results-content', 'children'),
         Output('test-output-text', 'children')],
        Input('run-all-tests-btn', 'n_clicks'),
        [State('test-path-input', 'value'),
         State('test-markers-input', 'value')],
        prevent_initial_call=True
    )
    def run_tests(n_clicks, test_path, markers):
        """Run tests."""
        if not n_clicks:
            raise PreventUpdate

        try:
            logger.info(f"Launching test execution: path={test_path}, markers={markers}")

            # Launch Celery task
            task = run_tests_task.delay(test_path=test_path, markers=markers)

            return (
                dbc.Alert([
                    html.H5("Test Execution Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("Running tests..."),
                    dbc.Spinner(size="sm")
                ], color="info"),
                "Waiting for test results..."
            )

        except Exception as e:
            logger.error(f"Failed to start tests: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color="danger"),
                ""
            )

    @app.callback(
        [Output('coverage-results-content', 'children'),
         Output('coverage-chart', 'figure')],
        Input('run-coverage-btn', 'n_clicks'),
        [State('coverage-test-path', 'value'),
         State('coverage-source-path', 'value'),
         State('coverage-threshold', 'value')],
        prevent_initial_call=True
    )
    def run_coverage(n_clicks, test_path, source_path, threshold):
        """Run coverage analysis."""
        if not n_clicks:
            raise PreventUpdate

        try:
            logger.info(f"Launching coverage analysis: test_path={test_path}, source={source_path}")

            # Launch Celery task
            task = run_coverage_task.delay(
                test_path=test_path,
                source_path=source_path,
                min_coverage=float(threshold)
            )

            # Placeholder chart
            fig = go.Figure()
            fig.update_layout(
                title="Coverage Report",
                xaxis_title="",
                yaxis_title="Coverage %"
            )

            return (
                dbc.Alert([
                    html.H5("Coverage Analysis Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("Analyzing code coverage..."),
                    dbc.Spinner(size="sm")
                ], color="info"),
                fig
            )

        except Exception as e:
            logger.error(f"Failed to start coverage: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color="danger"),
                go.Figure()
            )

    @app.callback(
        [Output('benchmark-results-content', 'children'),
         Output('benchmark-latency-chart', 'figure'),
         Output('benchmark-throughput-chart', 'figure')],
        Input('run-benchmarks-btn', 'n_clicks'),
        [State('benchmark-model-path', 'value'),
         State('benchmark-api-url', 'value')],
        prevent_initial_call=True
    )
    def run_benchmarks(n_clicks, model_path, api_url):
        """Run performance benchmarks."""
        if not n_clicks:
            raise PreventUpdate

        try:
            logger.info(f"Launching benchmarks: model={model_path}, api={api_url}")

            # Launch Celery task
            task = run_benchmarks_task.delay(
                model_path=model_path if model_path else None,
                api_url=api_url if api_url else None,
                num_samples=100
            )

            # Placeholder charts
            latency_fig = go.Figure()
            latency_fig.update_layout(title="Latency Benchmarks")

            throughput_fig = go.Figure()
            throughput_fig.update_layout(title="Throughput Benchmarks")

            return (
                dbc.Alert([
                    html.H5("Benchmarks Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("Running performance benchmarks..."),
                    dbc.Spinner(size="sm")
                ], color="info"),
                latency_fig,
                throughput_fig
            )

        except Exception as e:
            logger.error(f"Failed to start benchmarks: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color="danger"),
                go.Figure(),
                go.Figure()
            )

    @app.callback(
        Output('quality-results-content', 'children'),
        Input('run-quality-btn', 'n_clicks'),
        State('quality-path', 'value'),
        prevent_initial_call=True
    )
    def run_quality_checks(n_clicks, path):
        """Run code quality checks."""
        if not n_clicks:
            raise PreventUpdate

        try:
            logger.info(f"Launching quality checks: path={path}")

            # Launch Celery task
            task = run_quality_checks_task.delay(path=path)

            return dbc.Alert([
                html.H5("Quality Checks Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P("Running code quality analysis..."),
                dbc.Spinner(size="sm")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start quality checks: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_test_results_display(results: dict) -> html.Div:
    """
    Create test results display.

    Args:
        results: Test results dictionary

    Returns:
        Dash component
    """
    stats = results.get('stats', {})
    passed = stats.get('passed', 0)
    failed = stats.get('failed', 0)
    skipped = stats.get('skipped', 0)
    duration = stats.get('duration', 0)

    total = passed + failed + skipped
    pass_rate = (passed / total * 100) if total > 0 else 0

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Passed", className="text-muted"),
                    html.H3(str(passed), className="text-success")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Failed", className="text-muted"),
                    html.H3(str(failed), className="text-danger")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Skipped", className="text-muted"),
                    html.H3(str(skipped), className="text-warning")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Pass Rate", className="text-muted"),
                    html.H3(f"{pass_rate:.1f}%")
                ])
            ])
        ], width=3),
    ])


def create_coverage_chart(coverage_data: dict) -> go.Figure:
    """
    Create coverage chart.

    Args:
        coverage_data: Coverage data dictionary

    Returns:
        Plotly figure
    """
    files = coverage_data.get('files', {})

    if not files:
        return go.Figure()

    # Extract file coverage
    file_names = []
    file_coverage = []

    for file_path, file_data in files.items():
        file_names.append(file_path.split('/')[-1])  # Just filename
        summary = file_data.get('summary', {})
        percent = summary.get('percent_covered', 0)
        file_coverage.append(percent)

    # Sort by coverage
    sorted_data = sorted(zip(file_names, file_coverage), key=lambda x: x[1])
    file_names, file_coverage = zip(*sorted_data) if sorted_data else ([], [])

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(file_coverage),
            y=list(file_names),
            orientation='h',
            marker=dict(
                color=file_coverage,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100
            )
        )
    ])

    fig.update_layout(
        title="File Coverage",
        xaxis_title="Coverage %",
        yaxis_title="File",
        height=max(400, len(file_names) * 20)
    )

    return fig


def create_benchmark_charts(benchmark_data: dict) -> tuple:
    """
    Create benchmark charts.

    Args:
        benchmark_data: Benchmark results

    Returns:
        Tuple of (latency_fig, throughput_fig)
    """
    benchmarks = benchmark_data.get('benchmarks', {})

    # Latency chart
    latency_data = []
    labels = []

    if 'feature_extraction' in benchmarks:
        fe = benchmarks['feature_extraction']
        latency_data.append(fe.get('time_per_sample_ms', 0))
        labels.append('Feature Extraction')

    if 'model_inference' in benchmarks:
        mi = benchmarks['model_inference']
        latency_data.append(mi.get('time_per_sample_ms', 0))
        labels.append('Model Inference')

    if 'api_latency' in benchmarks:
        api = benchmarks['api_latency']
        latency_data.append(api.get('mean_latency_ms', 0))
        labels.append('API Latency')

    latency_fig = go.Figure(data=[
        go.Bar(x=labels, y=latency_data)
    ])
    latency_fig.update_layout(
        title="Latency Comparison",
        xaxis_title="Benchmark",
        yaxis_title="Time (ms)"
    )

    # Throughput chart
    throughput_data = []
    throughput_labels = []

    if 'feature_extraction' in benchmarks:
        fe = benchmarks['feature_extraction']
        throughput_data.append(fe.get('throughput_samples_per_sec', 0))
        throughput_labels.append('Feature Extraction')

    if 'model_inference' in benchmarks:
        mi = benchmarks['model_inference']
        throughput_data.append(mi.get('throughput_samples_per_sec', 0))
        throughput_labels.append('Model Inference')

    throughput_fig = go.Figure(data=[
        go.Bar(x=throughput_labels, y=throughput_data)
    ])
    throughput_fig.update_layout(
        title="Throughput Comparison",
        xaxis_title="Benchmark",
        yaxis_title="Samples/sec"
    )

    return latency_fig, throughput_fig
