from abc import ABC, abstractmethod

import pandas as pd



# Abstract base class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame):
        """
        Abstract method to apply feature engineering transformation.

  Parameters:
            df (pd.DataFrame): The DataFrame containing feature to be transformed.

        Returns:
            pd.DataFrame: DataFrame with applied transformation.
        """
        ...


# this strategy is used to extract date time features from a datetime column
class DateTimeFeatureStrategy(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initialize the DateTimeFeatureStrategy with a specific feature/column

        Parameters:
            features (str): The name of the feature/column to be transformed
        """
        self.features = features


    def apply_transformation(self, df: pd.DataFrame):
        """
        Apply datetime feature extraction to the specified column

        Parameters:
            df (pd.DataFrame): The DataFrame containing the datetime column to be transformed

        Returns:
            pd.DataFrame: DataFrame with extracted datetime features

        Raises:
            ValueError: If the specified column cannot be converted to datetime
        """
        converted = pd.to_datetime(df[self.features], errors="coerce")

        # Check if all values could be converted to datetime
        if converted.isna().all():
            raise ValueError("All values in the specified column could not be converted to datetime.")

        # Extract year, month, and day from the converted datetime
        df["year"] = converted.dt.year
        df["month"] = converted.dt.month
        df["day"] = converted.dt.day

        return df


class MonsoonSeasonFeatureStrategy(FeatureEngineeringStrategy):
    def __init__(self, feature):
        """
        Initialize the MonsoonSeasonFeatureStrategy with a specific feature/columns

        Parameters:
            feature (str): The name of the feature/column to be transformed
        """
        self.feature = feature

    def apply_transformation(self, df: pd.DataFrame):
        """
        Apply monsoon season feature extraction to the specified column

        Parameters:
            df (pd.DataFrame): The DataFrame containing the column to be transformed

        Returns:
            pd.DataFrame: DataFrame with the monsoon season feature added
        """
        df["monsoon_season"] = df[self.feature].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)

        return df


# this strategy is used to create lag features for time series data
class LagShiftStrategy(FeatureEngineeringStrategy):
    def __init__(self, group_by, date_col, target_col, lag):
        """
        Initialize the LagShiftStrategy with grouping, date, target column and lag values

        Parameters:
            group_by (str): The column name to group by for lag calculation
            date_col (str): The date column name for sorting
            target_col (str): The target column name to create lag features from
            lag (int): The number of periods to shift for lag calculation
        """
        self.group_by = group_by
        self.date_col = date_col
        self.target_col = target_col
        self.lag = lag

    def apply_transformation(self, df: pd.DataFrame):
        """
        Apply lag shift transformation to create lag features

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be transformed

        Returns:
            pd.DataFrame: DataFrame with lag features added
        """
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.group_by, self.date_col]).reset_index(drop=True)

        df[f"lag_{self.lag}"] = df.groupby(self.group_by)[self.target_col].shift(self.lag)

        return df


# this strategy is used to create rolling mean features for time series data
class RollingMeanStrategy(FeatureEngineeringStrategy):
    def __init__(self, group_by, date_col, target_col, window):
        """
        Initialize the RollingMeanStrategy with grouping, date, target column and window values

        Parameters:
            group_by (str): The column name to group by for rolling calculation
            date_col (str): The date column name for sorting
            target_col (str): The target column name to calculate rolling mean from
            window (int): The window size for rolling mean calculation
        """
        self.group_by = group_by
        self.date_col = date_col
        self.target_col = target_col
        self.window = window

    def apply_transformation(self, df: pd.DataFrame):
        """
        Apply rolling mean transformation to create rolling average features

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be transformed

        Returns:
            pd.DataFrame: DataFrame with rolling mean features added
        """
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.group_by, self.date_col]).reset_index(drop=True)

        df[f"rolling_{self.window}_mean"] = df.groupby(self.group_by)[self.target_col].transform(
            lambda x: x.shift(1).rolling(window=self.window).mean()
        )

        return df


# this strategy is used to calculate dry streak features for time series data
class DryStreakStrategy(FeatureEngineeringStrategy):
    def __init__(self, group_by, date_col, target_col):
        """
        Initialize the DryStreakStrategy with grouping, date, and target column values

        Parameters:
            group_by (str): The column name to group by for dry streak calculation
            date_col (str): The date column name for sorting
            target_col (str): The target column name to calculate dry streaks from (typically rainfall)
        """
        self.group_by = group_by
        self.date_col = date_col
        self.target_col = target_col

    def apply_transformation(self, df: pd.DataFrame):
        """
        Apply dry streak transformation to calculate consecutive days with no rainfall

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be transformed

        Returns:
            pd.DataFrame: DataFrame with dry streak feature added
        """
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.group_by, self.date_col]).reset_index(drop=True)

        df['dry_streak'] = df.groupby(self.group_by)[self.target_col].transform(
            lambda x: x.shift(1).eq(0).groupby((x.shift(1) != 0).cumsum()).cumsum()
        )

        return df


# Context class for FeatureEngineering
class FeatureEngineerHandler:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineerHandler with specific feature

        Parameter:
            strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering
        """
        self._strategy = strategy


    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Set a new strategy for FeatureEngineerHandler

        Parameters:
            strategy (FeatureEngineeringStrategy): The new Strategy to be used for feature engineering
        """
        self._strategy = strategy


    def apply_feature_engineering(self, df: pd.DataFrame):
        """
        Execute the feature engineering transformation using the current strategy

        Parameters:
            df (pd.DataFrame): the DataFrame containing features to be transformed

        Returns:
            pd.DataFrame: the DataFrame with applied feature engineering
        """
        return self._strategy.apply_transformation(df)
