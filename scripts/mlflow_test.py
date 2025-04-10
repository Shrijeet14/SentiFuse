import mlflow
import random
import dagshub

dagshub.init(repo_owner='Shrijeet14', repo_name='SentiFuse', mlflow=True)
# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Shrijeet14/SentiFuse.mlflow")

# mlflow.set_experiment('dvc-pipeline-runs')

# Start an MLflow run
with mlflow.start_run():
    # Log some random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log some random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")