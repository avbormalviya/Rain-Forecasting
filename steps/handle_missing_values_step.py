import pandas as pd

from zenml import step
from src.handle_missing_values import MissingValueHandler, DropMissingValuesStrategy, FillMissingValuesStrategy


@step
def handle_missing_values_step(
        df: pd.DataFrame,
        axis: int = 0,
        threshold: float = 0.5,
        strategy: str = "constant",
        fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        axis (int): 0 to drop rows, 1 to drop columns
        threshold (float): The threshold for non-NA values. Rows/Columns with less than this number of non-NA values will be dropped.
        strategy (str): Strategy for handling missing values.
        fill_value (float): Value to fill in when using constant strategy.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=axis, threshold=threshold))
    elif strategy in ["mean", "median", "mode"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    elif strategy == "constant":
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy, fill_value=fill_value))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    df_cleaned = handler.handle_missing_values(df)
    return df_cleaned