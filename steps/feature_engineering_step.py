import pandas as pd
from src.preprocess import apply_all_features
from zenml import step


@step
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_all_features(df)

    # save history buffer for inference later
    df.groupby('state_name').tail(30).to_csv('data/history_buffer.csv', index=False)

    # Save processed data
    df.to_csv('data/processed_data.csv', index=False)

    # drop leaky columns
    df.drop(columns=['id', 'date', 'deviation', 'state_name'], inplace=True)

    return df