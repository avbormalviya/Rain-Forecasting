import logging

from typing import Tuple
import pandas as pd

from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy

from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Split the input DataFrame into train and test datasets """
    splitter = DataSplitter(SimpleTrainTestSplitStrategy())

    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    # save test data
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test
