import logging

from typing import Tuple, Annotated
import pandas as pd

from src.preprocess import preprocess_new_data
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

from zenml import step
from sklearn.pipeline import Pipeline


@step
def model_evaluator_step(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> Tuple[
    Annotated[float, "mae"],
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
    Annotated[float, "r2"]
]:
    """
    Evaluate the model on the test data.

    Parameters:
        model (Pipeline): The trained model.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data labels.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the metrics (MAE, MSE, RMSE, R2).
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Evaluating the model...")
    evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    metrics = evaluator.evaluate(model, X_test, y_test.values)

    logging.info("Model evaluation completed.")

    # Validate metrics
    if not isinstance(metrics, tuple) or len(metrics) != 4:
        raise ValueError("Model evaluation failed to return a tuple of 4 metrics (MAE, MSE, RMSE, R2).")

    return metrics