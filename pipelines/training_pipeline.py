import logging

from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

from zenml import Model, pipeline, step

import pandas as pd


@pipeline(
    model=Model(
        name="rainfall_forcast"
    ),
)
def ml_pipeline():

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path=r"D:\WorkShop\3. MachineLearning\ML - Project\Rain_Forcasting_Project\extracted_data\daily-rainfall-at-state-level.csv"
    )

    # Handling Missing Values Step
    dropped_data = handle_missing_values_step(raw_data, strategy="drop", axis=1, threshold=0.8)
    filled_data = handle_missing_values_step(dropped_data, strategy="constant", fill_value=0.0)

    # Feature Engineering Step
    feature_engineered_data = feature_engineering_step(df=filled_data)

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(
        df=feature_engineered_data,
        target_column="actual"
    )

    # Model Building Step
    model = model_building_step(X_train, y_train, X_test, y_test)

    # Model Evaluation Step
    model_metrics = model_evaluator_step(model, X_test, y_test)

    return model, model_metrics


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()