import logging
from typing import Callable, Dict

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# Abstract class for missing values handling strategies
class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in a DataFrame

        Parameters:
            df (pd.DataFrame): The DataFrame containing missing values

        Returns:
            pd.DataFrame: The DataFrame with missing values handled
        """
        ...


# Concrete strategy class for Dropping missing values
class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, axis=0, threshold=None):
        """
        Initialize the strategy with axis and threshold

        Parameters:
            axis (int): 0 to drop rows, 1 to drop columns
            threshold (float): The threshold for non-NA values. Rows/Columns with less than this number of non-NA values will be dropped.
        """
        self.axis = axis
        self.threshold = threshold


    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows or columns with missing values based on the threshold

        Parameters:
            df (pd.DataFrame): The DataFrame to handle missing values

        Returns:
            pd.DataFrame: The DataFrame with missing values dropped
        """
        thresh = int(len(df.columns) * self.threshold)

        logging.info(f"Dropping missing values with axis {self.axis} and threshold {thresh} from the DataFrame")
        df_cleaned  = df.dropna(axis=self.axis, thresh=thresh)
        logging.info("Missing values dropped successfully")
        return df_cleaned


# Concrete strategy class for filling missing values
class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with specific parameters

        Parameters:
            method (str): The method to use for filling missing values.
                Options: 'mean', 'median', 'mode', 'constant'
            fill_value (any): The constant values to fill missing values with
       """

        # Store configuration chosen by user
        self.method = method
        self.fill_value = fill_value

        # Mapping of each column -> function that compute fill value each column
        # Each lambda receives a pandas series (column)
        self.methods: Dict[str, Callable] = {
            "mean": lambda col: col.mean(),
            "median": lambda col: col.median(),
            "mode": lambda col: col.mode().iloc[0] if not col.mode().empty else np.nan,

            # for constant strategy, return predefined value
            "constant": lambda col: self.fill_value
        }

        if self.method == "constant" and self.fill_value is None:
            raise ValueError("fill_value must be provided when method='constant")

        if self.method not in self.methods:
            raise ValueError(f"Invalid method: {self.method}")


    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values based on the method

        Parameters:
            df (pd.DataFrame): The DataFrame to handle missing values

        Returns:
            pd.DataFrame: The DataFrame with missing values handled
        """

        # Work on a copy to avoid mutating original dataset
        df_cleaned = df.copy()

        # Select only numeric columns (statistical methods apply to numbers)
        numeric_features = df_cleaned.select_dtypes(include=np.number).columns

        # Fill missing values column-wise
        for col in numeric_features:
            fill_val = self.methods[self.method](df_cleaned[col])
            df_cleaned[col] = df_cleaned[col].fillna(fill_val)

        logging.info("Missing values filled successfully")
        return df_cleaned


# Context class
class MissingValueHandler:
    def __init__(self, strategy: MissingValuesHandlingStrategy):
        """
        Initializes the HandleMissingValues with a specific strategy

        Parameters:
            strategy (MissingValuesHandlingStrategy): The strategy to use for handling missing values
        """
        self._strategy = strategy


    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle the missing values using current strategy

        Parameters:
            df (pd.DataFrame): The DataFrame to handle missing values

        Returns:
            pd.DataFrame: The DataFrame with missing values handled
        """
        return self._strategy.handle(df)


    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        """
        Set the strategy to use for handling missing values

        Parameters:
            strategy (MissingValuesHandlingStrategy): The strategy to use for handling missing values
        """
        self._strategy = strategy

