import logging
from typing import Tuple

import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from abc import ABC, abstractmethod


# Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, float]:
        """
        Abstract method to evaluate the model.

        Parameters:
            model (RegressorMixin): The model to evaluate.
            X_test (pd.DataFrame): The test data features.
            y_test (pd.Series): The test data labels.

        Returns:
            Tuple: A tuple containing the metrics (MAE, MSE, RMSE, R2).
        """
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, float]:
        """
        Evaluates a regression model.

        Parameters:
            model (RegressorMixin): The trained regression model
            X_test (pd.DataFrame): The test data features
            y_test (pd.Series): The test data labels

        Returns:
            Tuple: A tuple containing the metrics (MAE, MSE, RMSE, R2).
        """
        logging.info("Predicting on test data...")
        y_pred = model.predict(X_test)

        # Save predictions
        pd.Series(y_pred).to_csv('data/predictions.csv', index=False)

        # Calculate metrics
        logging.info("Calculating metrics...")
        mbe = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"MAE: {mbe}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        return mbe, mse, rmse, r2


# Context class for model evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initialize the ModelEvaluator with a specific evaluation strategy.

        Parameters:
            strategy (ModelEvaluationStrategy): The evaluation strategy to use.
        """
        self.strategy = strategy


    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Set a new evaluation strategy.

        Parameters:
            strategy (ModelEvaluationStrategy): The new evaluation strategy.
        """
        self.strategy = strategy


    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the model using the current evaluation strategy.

        Parameters:
            model (RegressorMixin): The trained regression model
            X_test (pd.DataFrame): The test data features
            y_test (pd.Series): The test data labels

        Returns:
            Tuple: A tuple containing the metrics (MAE, MSE, RMSE, R2).
        """
        return self.strategy.evaluate(model, X_test, y_test)

