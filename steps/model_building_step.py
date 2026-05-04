import logging
import joblib, os
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType
from zenml.client import Client

from scripts.tune_hyperparams import tune_hyperparams


experiment_tracker = Client().active_stack.experiment_tracker
mlflow.set_tracking_uri(experiment_tracker.get_tracking_uri())

if experiment_tracker is None:
    raise RuntimeError("No experiment tracker in active stack. Register MLflow tracker first.")


from zenml import Model

model = Model(
    name="rainfall_forcast",
    version=None,
    description="Rainfall prediction model"
)


@step(enable_cache=True, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)]:
    """
    Build and train a model.

    Parameters:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.series): Training data labels.

    Return:
        Pipeline: The trained sklearn pipeline including preprocessing.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Hyperparameter tuning
    model = tune_hyperparams(X_train, y_train, X_test, y_test)

    # Create a pipeline
    pipeline = Pipeline([
        ('model', model)
    ])


    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        # Assign weights to monsoon months
        weights = X_train['month'].map({
            6: 3.0, 7: 4.0, 8: 4.0, 9: 3.0
        }).fillna(1.0)

        logging.info("Fitting the model...")
        pipeline.fit(X_train, y_train)
        logging.info("Model fitted successfully.")

        # Save the pipeline
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, 'models/trained_pipeline.pkl')

    except Exception as e:
        logging.error(f"Failed to fit the model: {str(e)}")
        raise

    finally:
        mlflow.end_run()


    return pipeline
