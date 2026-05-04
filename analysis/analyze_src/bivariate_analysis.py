from abc import ABC, abstractmethod

from typing import Dict, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Define available plot types for categorical-numerical analysis
plots: Dict[str, Callable] = {
    "box": sns.boxplot,
    "violin": sns.violinplot,
    "strip": sns.stripplot,
}


# Abstract Base Class for Bivariate Analysis
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, filter: dict = None, sort_order: Optional[str] = None) -> None:
        """
        Analyze the relationship between two features.

        Parameters:
            df (pd.DataFrame): the DataFrame containing the data
            feature1 (str): the name of the first feature
            feature2 (str): the name of the second feature
            filter (dict, optional): a dictionary of filter conditions
            sort_order (str, optional): 'ascending' or 'descending' for sorting categorical data

        Returns:
            None: Displays the bivariate analysis plot
        """
        ...


# Concrete Strategy for Numerical-Numerical Analysis
class NumericalNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, filter: dict = None, sort_order: Optional[str] = None) -> None:
        """
        Analyze the relationship between two numerical features.

        Parameters:
            df (pd.DataFrame): the DataFrame containing the data
            feature1 (str): the name of the first feature
            feature2 (str): the name of the second feature
            filter (dict, optional): a dictionary of filter conditions
            sort_order (str, optional): 'ascending' or 'descending' for sorting (not applicable for scatter plots)

        Returns:
            None: Displays the bivariate analysis plot
        """

        data = df.copy()

        if filter:
            condition = pd.Series([True] * len(data), index=data.index)

            if feature1 in filter:
                condition &= filter[feature1](data[feature1])

            if feature2 in filter:
                condition &= filter[feature2](data[feature2])

            data = data[condition]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=data)
        plt.title(f"Scatter Plot: {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete Strategy for Categorical-Numerical Analysis
class CategoricalNumericalAnalysis(BivariateAnalysisStrategy):
    def __init__(self, plot: str = "box"):
        """
        Initialize the CategoricalNumericalAnalysis with a specific plot type.

        Parameters:
            plot (str): the type of plot to use for the analysis. Default is "box".
        """
        self.plot = plot


    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, filter: dict = None, sort_order: Optional[str] = None) -> None:
        """
        Analyze the relationship between a numerical feature and a categorical feature.

        Parameters:
            df (pd.DataFrame): the DataFrame containing the data
            feature1 (str): the name of the categorical feature
            feature2 (str): the name of the numerical feature
            filter (dict, optional): a dictionary of filter conditions
            sort_order (str, optional): 'ascending' or 'descending' for sorting categorical data

        Returns:
            None: Displays the bivariate analysis plot
        """
        if self.plot not in plots:
            raise ValueError(f"Invalid plot type: {self.plot}")

        data = df.copy()

        if filter:
            condition = pd.Series([True] * len(data), index=data.index)

            if feature1 in filter:
                condition &= filter[feature1](data[feature1])

            if feature2 in filter:
                condition &= filter[feature2](data[feature2])

            data = data[condition]

        # Sort categorical data if sort_order is specified
        if sort_order and feature1 in data.columns:
            ascending = sort_order.lower() == 'ascending'

            order = (
                data.groupby(feature1)[feature2]
                .mean()
                .sort_values(ascending=ascending)
                .index
            )
        else:
            order = None

        params: Dict[str, any] = {
            "x": feature1,
            "y": feature2,
            "data": data
        }
        
        if order is not None:
            params["order"] = order

        plt.figure(figsize=(10, 6))
        plots[self.plot](**params)
        plt.title(f"{feature2} distribution by {feature1}")
        plt.xticks(rotation=90)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete Strategy for Categorical-Categorical Analysis
class CategoricalCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, filter: dict = None, sort_order: Optional[str] = None) -> None:
        """
        Analyze the relationship between two categorical features.

        Parameters:
            df (pd.DataFrame): the DataFrame containing the data
            feature1 (str): the name of the first feature
            feature2 (str): the name of the second feature
            filter (dict, optional): a dictionary of filter conditions
            sort_order (str, optional): 'ascending' or 'descending' for sorting categorical data

        Returns:
            None: Displays the bivariate analysis plot
        """
        data = df.copy()

        if filter:
            condition = pd.Series([True] * len(data), index=data.index)

            if feature1 in filter:
                condition &= filter[feature1](data[feature1])

            if feature2 in filter:
                condition &= filter[feature2](data[feature2])

            data = data[condition]

        # Sort categorical data if sort_order is specified
        if sort_order and feature1 in data.columns:
            if sort_order.lower() == 'ascending':
                order = data[feature1].value_counts().sort_values().index
            elif sort_order.lower() == 'descending':
                order = data[feature1].value_counts().sort_values(ascending=False).index
            else:
                raise ValueError("sort_order must be 'ascending' or 'descending'")
        else:
            order = None

        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature1, hue=feature2, data=data, order=order)
        plt.title(f"Count Plot: {feature1} by {feature2}")
        plt.xlabel(feature1)
        plt.ylabel("Count")
        plt.show()


# Context Class
class BivariateAnalysisContext:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initialize the context with a strategy.

        Parameters:
            strategy (BivariateAnalysisStrategy): The strategy to use for bivariate analysis
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy) -> None:
        """
        Set a new strategy for bivariate analysis.

        Parameters:
            strategy (BivariateAnalysisStrategy): The new strategy to use

        Returns:
            None: Sets the new strategy
        """
        self._strategy = strategy

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, filter: dict = None, sort_order: Optional[str] = None) -> None:
        """
        Analyze the relationship between two features using the current strategy.

        Parameters:
            df (pd.DataFrame): the DataFrame containing the data
            feature1 (str): the name of the first feature
            feature2 (str): the name of the second feature
            filter (dict, optional): a dictionary of filter conditions
            sort_order (str, optional): 'ascending' or 'descending' for sorting categorical data

        Returns:
            None: Analyzes the relationship between the two features
        """
        self._strategy.analyze(df, feature1, feature2, filter, sort_order)


if __name__ == "__main__":
    pass