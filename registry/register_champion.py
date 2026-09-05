import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "UAE-Retail-Demand-XGBoost"
EXPERIMENT_NAME = "UAE_Retail_Demand_Forecasting"

# ── Find your best run by MAE ────────────────────────────
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.MAE ASC"],
    max_results=1
)
best_run = runs[0]
best_run_id  = best_run.info.run_id
best_mae     = best_run.data.metrics['MAE']
best_mape    = best_run.data.metrics['MAPE']
best_model   = best_run.data.params['model']

print(f"Best run:  {best_run_id}")
print(f"Model:     {best_model}")
print(f"MAE:       {best_mae}")
print(f"MAPE:      {best_mape:.4f} ({best_mape*100:.2f}%)")

# ── Register it ─────────────────────────────────────────
model_uri = f"runs:/{best_run_id}/xgboost_model"
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)
print(f"\nRegistered as version: {model_version.version}")

# ── Add description ──────────────────────────────────────
client.update_model_version(
    name=MODEL_NAME,
    version=model_version.version,
    description=(
        f"XGBoost with Ramadan calendar features. "
        f"MAE: {best_mae:.2f} | MAPE: {best_mape*100:.2f}%. "
        f"Trained on Rossmann proxy data with Hijri calendar "
        f"feature engineering. Note: temporal collinearity "
        f"with German summer limits Ramadan feature impact — "
        f"see experiment notes for full analysis."
    )
)

# ── Promote to Production ────────────────────────────────
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Version {model_version.version} → Production ✅")

# ── Verify ───────────────────────────────────────────────
prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
print(f"\nProduction model:")
print(f"  Version:     {prod.version}")
print(f"  Run ID:      {prod.run_id}")
print(f"  Description: {prod.description}")