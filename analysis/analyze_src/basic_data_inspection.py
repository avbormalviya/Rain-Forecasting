from abc import ABC, abstractmethod

import pandas as pd


# Abstract base class for data inspection
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection

        Parameters:
            df (pd.DataFrame): The DataFrame on which Inspection will happen

        Returns:
            None: This function print the result directly
        """
        ...


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and print the data types and non-null counts

        Parameters:
            df (pd.DataFrame): The DataFrame to be inspected

        Returns:
            None: Print the data and non-null count to the console
        """
        print("\nData Type and Non-null Count")
        df.info()


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and Print the Summary Statistic for Numerical and Categorical Features

        Parameters:
            df (pd.DataFrame): The DataFrame to be inspected

        Returns:
            None: Prints Summary Statistics to the Console
        """
        print("\nSummary Statistics (Numerical Features)")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features)")
        print(df.describe(include=['O']))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initialize the DataInspection class with specific inspection strategy

        Parameters:
            strategy (DataInspectionStrategy): The strategy to be used for data inspection

        Returns:
            None
        """
        self._strategy = strategy


    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Set the new strategy for Data Inspection

        Parameters:
            strategy (DataInspectionStrategy): The new strategy to be used for data Inspection

        Returns:
            None
        """
        self._strategy = strategy


    def execute_inspection(self, df: pd.DataFrame):
        """
        Execute the data inspection using the current strategy

        Parameters:
            df (pd.DataFrame): The DataFrame to be inspected

        Returns:
            None
        """
        self._strategy.inspect(df)


# Example Usage
if __name__ == "__main__":
    ...