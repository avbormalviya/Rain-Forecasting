import optuna

import pandas as pd

from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Hyperparameter Tuning Constants
N_TRIALS = 50
STORAGE = "sqlite:///optuna.db"
DIRECTION = "maximize"


# Hyperparameter Tuning
def tune_hyperparams(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> dict:
    """
    Find the best hyperparameters for the model.

    Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data labels.

    Returns:
        dict: The best hyperparameters for the model.
    """
    # Objective function for hyperparameter optimization
    def objective(trial):
        # Model selection
        model_name = trial.suggest_categorical("model", ["lgbm", "xgboost"])

        # Model hyperparameters
        if model_name == "lgbm":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }

            model = LGBMRegressor(**params)

        elif model_name == "xgboost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            }

            model = XGBRegressor(**params)


        # Model training and prediction
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Return negative R2 score for maximization
        return r2_score(y_test, pred)


    # Create Optuna study
    study = optuna.create_study(
        study_name="rainfall_forcast",
        storage=STORAGE,
        direction=DIRECTION,
        load_if_exists=True
    )

    # Optimize hyperparameters
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    best_model_name = best_params.pop("model")

    # Return untrained model instance with best params
    if best_model_name == "lgbm":
        best_model = LGBMRegressor(**best_params)
    elif best_model_name == "xgboost":
        best_model = XGBRegressor(**best_params)
    
    return best_model

