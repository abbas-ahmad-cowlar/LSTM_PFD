"""
Experiment wizard callbacks (Phase 11B).
Handles multi-step wizard navigation and experiment launch.
"""
import time
import json
from dash import Input, Output, State, ALL, callback_context, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from layouts.experiment_wizard import (
    create_step1_model_selection,
    create_step2_dataset_hyperparams,
    create_step3_training_options,
    create_step4_review_launch
)
from services.data_service import DataService
from database.connection import get_db_session
from models.dataset import Dataset
from models.experiment import Experiment, ExperimentStatus
from tasks.training_tasks import train_model_task
from utils.logger import setup_logger
from utils.constants import (
    NUM_CLASSES,
    SIGNAL_LENGTH,
    SAMPLING_RATE,
    RF_N_ESTIMATORS_MIN,
    RF_N_ESTIMATORS_MAX,
    RF_N_ESTIMATORS_STEP,
    RF_MAX_DEPTH_MIN,
    RF_MAX_DEPTH_MAX,
    RF_MAX_DEPTH_STEP,
    SVM_GAMMA_MIN,
    SVM_GAMMA_MAX,
    SVM_GAMMA_STEP,
    NN_FILTERS_MIN,
    NN_FILTERS_MAX,
    NN_FILTERS_STEP,
    TRANSFORMER_D_MODEL_MIN,
    TRANSFORMER_D_MODEL_MAX,
    TRANSFORMER_D_MODEL_STEP,
    DEFAULT_EPOCHS_FALLBACK,
    DEFAULT_LEARNING_RATE_FALLBACK,
    PROGRESSIVE_START_SIZE_DEFAULT,
    PROGRESSIVE_END_SIZE_DEFAULT,
    PERCENT_MULTIPLIER,
)

logger = setup_logger(__name__)


# Model hyperparameter templates
MODEL_HYPERPARAMS = {
    "rf": {
        "n_estimators": (RF_N_ESTIMATORS_MIN, RF_N_ESTIMATORS_MAX, RF_N_ESTIMATORS_STEP),
        "max_depth": (RF_MAX_DEPTH_MIN, RF_MAX_DEPTH_MAX, RF_MAX_DEPTH_STEP),
        "min_samples_split": (2, 20, 2),
    },
    "svm": {
        "C": (0.1, 10.0, 0.1),
        "gamma": (SVM_GAMMA_MIN, SVM_GAMMA_MAX, SVM_GAMMA_STEP),
    },
    "cnn1d": {
        "num_filters": (NN_FILTERS_MIN, NN_FILTERS_MAX, NN_FILTERS_STEP),
        "dropout": (0.0, 0.5, 0.1),
    },
    "resnet18": {
        "dropout": (0.0, 0.5, 0.1),
    },
    "resnet34": {
        "dropout": (0.0, 0.5, 0.1),
    },
    "transformer": {
        "d_model": (TRANSFORMER_D_MODEL_MIN, TRANSFORMER_D_MODEL_MAX, TRANSFORMER_D_MODEL_STEP),
        "nhead": (4, 16, 4),
        "num_layers": (2, 12, 2),
        "dropout": (0.0, 0.5, 0.1),
    },
}


def register_experiment_wizard_callbacks(app):
    """Register all experiment wizard callbacks."""

    @app.callback(
        [Output("wizard-content", "children"),
         Output("wizard-progress", "value"),
         Output("wizard-prev-btn", "disabled"),
         Output("wizard-next-btn", "children"),
         Output("wizard-next-btn", "disabled"),
         Output("step1-nav", "active"),
         Output("step2-nav", "active"),
         Output("step3-nav", "active"),
         Output("step4-nav", "active"),
         Output("step1-nav", "disabled"),
         Output("step2-nav", "disabled"),
         Output("step3-nav", "disabled"),
         Output("step4-nav", "disabled")],
        [Input("wizard-step", "data"),
         Input("wizard-config", "data")]
    )
    def update_wizard_display(step, config):
        """Update wizard content based on current step."""
        if step == 1:
            content = create_step1_model_selection()
            progress = 25
            prev_disabled = True
            next_text = "Next"
            next_disabled = not config.get("model_type")
            nav_active = (True, False, False, False)
            nav_disabled = (False, True, True, True)
        elif step == 2:
            content = create_step2_dataset_hyperparams(config.get("model_type"))
            progress = 50
            prev_disabled = False
            next_text = "Next"
            next_disabled = not (config.get("dataset_id") and config.get("hyperparameters"))
            nav_active = (False, True, False, False)
            nav_disabled = (False, False, True, True)
        elif step == 3:
            content = create_step3_training_options()
            progress = 75
            prev_disabled = False
            next_text = "Review"
            next_disabled = False
            nav_active = (False, False, True, False)
            nav_disabled = (False, False, False, True)
        else:  # step == 4
            content = create_step4_review_launch(config)
            progress = PERCENT_MULTIPLIER
            prev_disabled = False
            next_text = "Launch"
            next_disabled = True  # Launch button is separate
            nav_active = (False, False, False, True)
            nav_disabled = (False, False, False, False)

        return (content, progress, prev_disabled, next_text, next_disabled,
                *nav_active, *nav_disabled)

    @app.callback(
        Output("wizard-step", "data"),
        [Input("wizard-next-btn", "n_clicks"),
         Input("wizard-prev-btn", "n_clicks")],
        [State("wizard-step", "data")]
    )
    def navigate_wizard(next_clicks, prev_clicks, current_step):
        """Handle wizard navigation."""
        if not callback_context.triggered:
            raise PreventUpdate

        button_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        if button_id == "wizard-next-btn":
            return min(current_step + 1, 4)
        elif button_id == "wizard-prev-btn":
            return max(current_step - 1, 1)

        raise PreventUpdate

    @app.callback(
        Output("wizard-config", "data", allow_duplicate=True),
        Input({"type": "model-select-btn", "index": ALL}, "n_clicks"),
        State("wizard-config", "data"),
        prevent_initial_call=True
    )
    def select_model(n_clicks, config):
        """Handle model selection."""
        if not callback_context.triggered or not any(n_clicks):
            raise PreventUpdate

        # Extract model_id from the button that was clicked
        button_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        button_data = json.loads(button_id)
        model_type = button_data["index"]

        config["model_type"] = model_type
        logger.info(f"Model selected: {model_type}")
        return config

    @app.callback(
        Output("selected-model-summary", "children"),
        Input("wizard-config", "data")
    )
    def display_selected_model(config):
        """Display selected model summary."""
        model_type = config.get("model_type")
        if not model_type:
            return dbc.Alert("No model selected", color="warning")

        return dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            html.Strong(f"Selected: {model_type.upper()}")
        ], color="success")

    @app.callback(
        Output("dataset-dropdown", "options"),
        Input("wizard-step", "data")
    )
    def load_datasets(step):
        """Load available datasets for dropdown."""
        if step != 2:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).limit(50).all()
                return [
                    {"label": f"{ds.name} ({ds.num_signals} signals)", "value": ds.id}
                    for ds in datasets
                ]
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    @app.callback(
        Output("hyperparameters-form", "children"),
        Input("wizard-config", "data")
    )
    def create_hyperparameter_form(config):
        """Dynamically create hyperparameter form based on model type."""
        model_type = config.get("model_type")
        if not model_type:
            return html.P("Select a model first", className="text-muted")

        hyperparam_template = MODEL_HYPERPARAMS.get(model_type, {})
        if not hyperparam_template:
            return dbc.Alert("No hyperparameters to configure for this model", color="info")

        form_fields = []
        for param_name, (min_val, max_val, step_val) in hyperparam_template.items():
            form_fields.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Label(param_name.replace("_", " ").title()),
                        dbc.Input(
                            type="number",
                            id={"type": "hyperparam", "name": param_name},
                            value=(min_val + max_val) // 2 if isinstance(min_val, int) else (min_val + max_val) / 2,
                            min=min_val,
                            max=max_val,
                            step=step_val,
                            className="mb-3"
                        ),
                    ])
                ])
            )

        return html.Div(form_fields)

    @app.callback(
        Output("advanced-options-collapse", "is_open"),
        Input("show-advanced-options", "value")
    )
    def toggle_advanced_options(show_advanced):
        """Toggle advanced options visibility."""
        return show_advanced

    @app.callback(
        [Output("review-model-config", "children"),
         Output("review-training-config", "children")],
        Input("wizard-config", "data")
    )
    def update_review_summary(config):
        """Update review summary on step 4."""
        if not config:
            return "No configuration", "No configuration"

        # Model config summary
        model_summary = html.Ul([
            html.Li(f"Model Type: {config.get('model_type', 'Not selected')}"),
            html.Li(f"Dataset: {config.get('dataset_id', 'Not selected')}"),
            html.Li(f"Hyperparameters: {len(config.get('hyperparameters', {}))} configured"),
        ])

        # Training config summary
        training_summary = html.Ul([
            html.Li(f"Epochs: {config.get('num_epochs', DEFAULT_EPOCHS_FALLBACK)}"),
            html.Li(f"Batch Size: {config.get('batch_size', 32)}"),
            html.Li(f"Optimizer: {config.get('optimizer', 'adam').upper()}"),
            html.Li(f"Learning Rate: {config.get('learning_rate', DEFAULT_LEARNING_RATE_FALLBACK)}"),
            html.Li(f"Augmentation: {len(config.get('augmentation', []))} enabled"),
        ])

        return model_summary, training_summary

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        [Input("launch-training-btn", "n_clicks")],
        [State("wizard-config", "data"),
         State("experiment-name", "value"),
         State("experiment-tags", "value"),
         State("experiment-notes", "value")],
        prevent_initial_call=True
    )
    def launch_training(n_clicks, config, name, tags, notes):
        """Launch training experiment."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Create experiment in database
            with get_db_session() as session:
                experiment = Experiment(
                    name=name or f"experiment_{config['model_type']}_{int(time.time())}",
                    model_type=config["model_type"],
                    dataset_id=config["dataset_id"],
                    status=ExperimentStatus.PENDING,
                    config=config,
                    hyperparameters=config.get("hyperparameters", {}),
                )
                session.add(experiment)
                session.flush()
                experiment_id = experiment.id
                session.commit()  # Commit to persist the experiment

            # Add experiment_id to config for the training task
            config["experiment_id"] = experiment_id

            # Launch Celery task
            task = train_model_task.delay(config)
            logger.info(f"Launched training task {task.id} for experiment {experiment_id}")

            # Redirect to training monitor page
            return f"/experiment/{experiment_id}/monitor"

        except Exception as e:
            logger.error(f"Failed to launch training: {e}")
            return "/experiments"  # Redirect to experiments list on error

    # Advanced Training Options Callbacks

    @app.callback(
        Output("distillation-config", "style"),
        Input("enable-distillation", "value")
    )
    def toggle_distillation_config(enable):
        """Show/hide distillation configuration."""
        if enable:
            return {"display": "block", "marginTop": "1rem"}
        return {"display": "none"}

    @app.callback(
        Output("advanced-aug-config", "style"),
        Input("enable-advanced-aug", "value")
    )
    def toggle_advanced_aug_config(enabled_augs):
        """Show/hide advanced augmentation configuration."""
        if enabled_augs and len(enabled_augs) > 0:
            return {"display": "block", "marginTop": "1rem"}
        return {"display": "none"}

    @app.callback(
        Output("progressive-config", "style"),
        Input("enable-progressive", "value")
    )
    def toggle_progressive_config(enable):
        """Show/hide progressive resizing configuration."""
        if enable:
            return {"display": "block", "marginTop": "1rem"}
        return {"display": "none"}

    @app.callback(
        Output("teacher-model-select", "options"),
        Input("enable-distillation", "value")
    )
    def load_teacher_models(enable):
        """Load available teacher models from completed experiments."""
        if not enable:
            return []

        try:
            with get_db_session() as session:
                # Get completed experiments
                completed_experiments = session.query(Experiment).filter(
                    Experiment.status == ExperimentStatus.COMPLETED
                ).order_by(Experiment.created_at.desc()).limit(20).all()

                return [
                    {
                        "label": f"{exp.name} ({exp.model_type}) - Acc: {exp.metrics.get('accuracy', 0):.2%}",
                        "value": exp.id
                    }
                    for exp in completed_experiments
                    if exp.metrics and 'accuracy' in exp.metrics
                ]
        except Exception as e:
            logger.error(f"Failed to load teacher models: {e}")
            return []

    @app.callback(
        Output("wizard-config", "data", allow_duplicate=True),
        [Input("dataset-dropdown", "value"),
         Input({"type": "hyperparam", "name": ALL}, "value"),
         Input("num-epochs", "value"),
         Input("batch-size-dropdown", "value"),
         Input("optimizer-dropdown", "value"),
         Input("learning-rate", "value"),
         Input("scheduler-dropdown", "value"),
         Input("augmentation-checklist", "value"),
         # Advanced options
         Input("enable-distillation", "value"),
         Input("teacher-model-select", "value"),
         Input("distillation-temperature", "value"),
         Input("distillation-alpha", "value"),
         Input("mixed-precision-mode", "value"),
         Input("enable-advanced-aug", "value"),
         Input("aug-magnitude", "value"),
         Input("aug-probability", "value"),
         Input("enable-progressive", "value"),
         Input("progressive-start-size", "value"),
         Input("progressive-end-size", "value")],
        [State("wizard-config", "data"),
         State({"type": "hyperparam", "name": ALL}, "id")],
        prevent_initial_call=True
    )
    def collect_training_config(
        dataset_id, hyperparam_values, num_epochs, batch_size, optimizer,
        learning_rate, scheduler, augmentation,
        enable_distillation, teacher_model_id, distill_temp, distill_alpha,
        mixed_precision, enable_adv_aug, aug_mag, aug_prob,
        enable_progressive, prog_start, prog_end,
        config, hyperparam_ids
    ):
        """Collect all training configuration including advanced options."""
        if not callback_context.triggered:
            raise PreventUpdate

        # Update basic training config
        config["dataset_id"] = dataset_id
        config["num_epochs"] = num_epochs or DEFAULT_EPOCHS_FALLBACK
        config["batch_size"] = batch_size or 32
        config["optimizer"] = optimizer or "adam"
        config["learning_rate"] = learning_rate or DEFAULT_LEARNING_RATE_FALLBACK
        config["scheduler"] = scheduler or "plateau"
        config["augmentation"] = augmentation or []

        # Collect hyperparameters
        hyperparameters = {}
        for param_id, value in zip(hyperparam_ids, hyperparam_values):
            param_name = param_id["name"]
            hyperparameters[param_name] = value
        config["hyperparameters"] = hyperparameters

        # Advanced options
        config["advanced_training"] = {}

        # Knowledge Distillation
        if enable_distillation and teacher_model_id:
            config["advanced_training"]["distillation"] = {
                "enabled": True,
                "teacher_model_id": teacher_model_id,
                "temperature": distill_temp or 3.0,
                "alpha": distill_alpha or 0.5
            }

        # Mixed Precision
        if mixed_precision and mixed_precision != "fp32":
            config["advanced_training"]["mixed_precision"] = {
                "enabled": True,
                "dtype": mixed_precision
            }

        # Advanced Augmentation
        if enable_adv_aug and len(enable_adv_aug) > 0:
            config["advanced_training"]["advanced_augmentation"] = {
                "enabled": True,
                "methods": enable_adv_aug,
                "magnitude": aug_mag or 9,
                "probability": aug_prob or 0.5
            }

        # Progressive Resizing
        if enable_progressive:
            config["advanced_training"]["progressive_resizing"] = {
                "enabled": True,
                "start_size": prog_start or PROGRESSIVE_START_SIZE_DEFAULT,
                "end_size": prog_end or PROGRESSIVE_END_SIZE_DEFAULT
            }

        logger.info(f"Updated training config with advanced options: {config.get('advanced_training', {})}")
        return config
