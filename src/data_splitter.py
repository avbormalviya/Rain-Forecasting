import logging

from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

import pandas as pd


# Abstract class for data splitting strategy
class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into train and testing datasets

        Parameters:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column name to be used as target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple of train and test datasets
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplitterStrategy):
    def __init__(self, test_split=0.2, random_state=42):
        """
        Initialize the SimpleTrainTestSplitStrategy with test split and random state

        Parameters:
            test_split (float): Fraction of data to be used as test
            random_state (int): Random seed for reproducibility
        """
        self.test_split = test_split
        self.random_state = random_state


    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the input DataFrame into train and test datasets

        Parameters:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column name to be used as target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple of train and test datasets
        """

        logging.info("Performing data splitting")

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_split,
            random_state=self.random_state
        )

        logging.info(f"Splitting data into train and test datasets with {self.test_split*100}% test size")

        return X_train, X_test, y_train, y_test


# Context class for data splitting
class DataSplitter:
    def __init__(self, strategy: DataSplitterStrategy):
        """
        Initialize the DataSplitter with a specific strategy

        Parameters:
            strategy (DataSplitterStrategy): The strategy to use for data splitting
        """
        self.strategy = strategy


    def set_strategy(self, strategy: DataSplitterStrategy):
        """
        Set a new strategy for data splitting

        Parameters:
            strategy (DataSplitterStrategy): The new strategy to use
        """
        self.strategy = strategy


    def split(self, df: pd.DataFrame, target_column: str):
        """
        Execute the data splitting process

        Parameters:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column name to be used as target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple of train and test datasets
        """
        return self.strategy.split_data(df, target_column)