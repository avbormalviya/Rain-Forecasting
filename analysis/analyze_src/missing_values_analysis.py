from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Abstract base class template for Missing Values Analysis
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """
        Analyze missing values in the dataset.

        Parameter:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: This method performs analysis in place and does not return any value.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)


    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """
        Performs missing values analysis by identifying the missing values in the dataset.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: This method should print the missing values analysis results.
        """
        ...


    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """
        Visualize missing values in the dataset.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: This method performs visualization in place and does not return any value.
        """
        ...


# Concrete implementation of Missing Values Analysis
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """
        Prints count of missing values in each column.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: Prints count of missing values in each column to the console.
        """
        print("\n Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])


    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Create a heatmap to visualize missing values in the dataset.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: Display a heatmap of missing values.
        """
        print("\n Visualizing Missing Values:")
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cmap="Blues", cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()


# Example Usage
if __name__ == "__main__":
    ...