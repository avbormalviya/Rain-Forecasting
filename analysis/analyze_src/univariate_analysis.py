from abc import ABC, abstractmethod

import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns


# Abstract base class for univariate analysis strategy.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, filter: dict = None) -> None:
        """
        Performs univariate analysis on a specific feature in dataframe

        Parameters:
            df (pd.DataFrame): The dataframe containing data.
            feature (str): The name of feature(column) to be analyzed.
            filter (dict, optional): Dictionary of filters to apply to the feature before analysis. Defaults to None.

        Return:
            None: This method visualizes the distribution of the feature.
        """
        ...


# Concrete Strategy for Numerical features
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, filter: dict = None) -> None:
        """
        Plot the distribution of the numerical feature using histogram and KDE.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            feature (str): The name of the numerical feature(column) to be analyzed.
            filter (dict, optional): Dictionary of filters to apply to the feature before analysis. Defaults to None.

        Return:
            None: Display the histogram with KDE plot.
        """

        if feature not in df.columns:
            raise ValueError(f"{feature} not found in the DataFrame")

        if not is_numeric_dtype(df[feature]):
            raise ValueError(f"{feature} is not numeric")

        data = df.copy()

        if filter and feature in filter:
            condition = filter[feature]
            data = data[condition(data[feature])]

        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, bins=100)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# Concreate Strategy for Categorical features.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, filter: dict = None) -> None:
        """
        PLot the distribution of the categorical feature using bar plot.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            feature (str): The name of the categorical feature(column) to be analyzed.
            filter (dict, optional): Dictionary of filters to apply to the feature before analysis. Defaults to None.

        Return:
              None: Display the bar plot showing frequency of each category.
        """
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in the DataFrame")

        if is_numeric_dtype(df[feature]):
            # Check if it's actually binary (0/1) — treat as categorical
            unique_vals = df[feature].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                df = df.copy()
                df[feature] = df[feature].astype(str)  # convert to "0", "1"
            else:
                raise ValueError(f"{feature} is not categorical")

        plt.figure(figsize=(12, 8))
        sns.countplot(x=feature, data=df, hue=df[feature])
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Concrete class for UnivariateAnalyzer
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific strategy.

        Parameters:
            strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
            None
        """
        self._strategy = strategy


    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Set the new strategy for univariate analysis.

        Parameters:
            strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
            None
        """
        self._strategy = strategy


    def execute_analysis(self, df: pd.DataFrame, feature: str, filter: dict = None):
        """
        Execute the univariate analysis using the current strategy

        Parameters:
            df (pd.DataFrame): The dataframe containing data.
            feature (str): The name of feature(column) to be analyzed.
            filter (dict, optional): Dictionary of filters to apply to the feature before analysis. Defaults to None.

        Returns:
            None
        """
        self._strategy.analyze(df, feature, filter)


if __name__ == "__main__":
    ...