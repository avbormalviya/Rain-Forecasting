import pandas as pd

from src.feature_engineering import (
    FeatureEngineerHandler,
    DateTimeFeatureStrategy,
    MonsoonSeasonFeatureStrategy,
    LagShiftStrategy,
    RollingMeanStrategy,
    DryStreakStrategy
)


# Apply all feature engineering
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    # DateTime Feature
    handler = FeatureEngineerHandler(
        strategy=DateTimeFeatureStrategy('date')
    )
    df = handler.apply_feature_engineering(df)

    handler.set_strategy(MonsoonSeasonFeatureStrategy('month'))
    df = handler.apply_feature_engineering(df)

    # Lag Shift
    handler.set_strategy(LagShiftStrategy('state_name', 'date', 'actual', 1))
    df = handler.apply_feature_engineering(df)

    handler.set_strategy(LagShiftStrategy('state_name', 'date', 'actual', 7))
    df = handler.apply_feature_engineering(df)

    handler.set_strategy(LagShiftStrategy('state_name', 'date', 'actual', 30))
    df = handler.apply_feature_engineering(df)

    # Rolling Mean
    handler.set_strategy(RollingMeanStrategy('state_name', 'date', 'actual', 7))
    df = handler.apply_feature_engineering(df)

    handler.set_strategy(RollingMeanStrategy('state_name', 'date', 'actual', 30))
    df = handler.apply_feature_engineering(df)

    # Dry Streak
    handler.set_strategy(DryStreakStrategy('state_name', 'date', 'actual'))
    df = handler.apply_feature_engineering(df)

    return df


# Preprocess
def preprocess_new_data(new_df: pd.DataFrame) -> pd.DataFrame:
    # Load history
    history = pd.read_csv('data/history_buffer.csv')

    # Combine history and new data
    combined = pd.concat([history, new_df]).reset_index(drop=True)
    combined = apply_all_features(combined)

    # drop NaNs from lag creation
    combined.dropna(inplace=True)

    # drop leaky columns
    combined.drop(columns=['id', 'date', 'deviation', 'state_name'], inplace=True)

    # Return only new data
    return combined.tail(len(new_df))
