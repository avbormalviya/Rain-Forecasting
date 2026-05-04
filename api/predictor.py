import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta


# absolute path of current file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load model and history once at startup
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'trained_pipeline.pkl'))
history  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'history_buffer.csv'))
processed_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed_data.csv'))
history['date'] = pd.to_datetime(history['date'])

# state metadata - normal values per state per month
STATE_NORMAL = (
    processed_data[['state_name', 'state_code', 'month', 'normal']]
    .drop_duplicates()
)

FEATURE_COLS = [
    'state_code', 'rfs', 'normal', 'year', 'month', 'day',
    'monsoon_season', 'lag_1', 'lag_7', 'lag_30',
    'rolling_7_mean', 'rolling_30_mean', 'dry_streak'
]


def get_state_normal(state_name: str, month: int) -> tuple:
    """Get state_code and normal rainfall for a given state and month."""
    row = STATE_NORMAL[
        (STATE_NORMAL['state_name'] == state_name) & (STATE_NORMAL['month'] == month)
    ]

    if row.empty:
        raise ValueError(f"No normal rainfall data found for state '{state_name}' and month {month}")

    return row['state_code'].iloc[0], row['normal'].iloc[0]


def get_forecast(state_name: str, start_date: str) -> list:
    """
    Generate 7-day recursive rainfall forecast for a given state.

    Parameters:
        state_name: Name of the Indian state
        start_date: Start date string in format YYYY-MM-DD

    Returns:
        List of dicts with date and predicted_rainfall_mm
    """
    state_history = (
        history[history['state_name'] == state_name]
        .sort_values(by='date')
        .tail(30)['actual']
        .to_list()
    )

    if len(state_history) == 0:
        raise ValueError(f"No history found for state: {state_name}")

    # pad with zeros if less than 30 days available
    while len(state_history) < 30:
        state_history.insert(0, 0)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    predictions = []

    for i in range(7):
        forecast_date = start + timedelta(days=i)
        month = forecast_date.month
        year = forecast_date.year
        day = forecast_date.day

        # get state metadata for this month
        state_code, state_normal = get_state_normal(state_name, month)

        # compute lag features from rolling history
        lag_1 = state_history[-1]
        lag_7 = state_history[-7]
        lag_30 = state_history[-30]

        rolling_7_mean = float(np.mean(state_history[-7:]))
        rolling_30_mean = float(np.mean(state_history[-30:]))

        # compute dry streak
        dry_streak = 0
        for val in reversed(state_history):
            if val == 0:
                dry_streak += 1
            else:
                break

        monsoon_season = 1 if month in [6, 7, 8, 9] else 0

        # use normal as proxy for rfs since no govt forecast available
        rfs = state_normal

        row = pd.DataFrame([{
            'state_code': state_code,
            'rfs': rfs,
            'normal': state_normal,
            'year': year,
            'month': month,
            'day': day,
            'monsoon_season': monsoon_season,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_30': lag_30,
            'rolling_7_mean': rolling_7_mean,
            'rolling_30_mean': rolling_30_mean,
            'dry_streak': dry_streak
        }])[FEATURE_COLS]  # ensure correct column order

        pred = float(pipeline.predict(row)[0])
        pred = max(0.0, round(pred, 2))  # no negative rainfall

        predictions.append({
            'date': forecast_date.strftime("%Y-%m-%d"),
            'predicted_rainfall_mm': pred
        })

        # append prediction to rolling history for next iteration
        state_history.append(pred)
        state_history = state_history[-30:]  # keep only last 30

    return predictions


def get_available_states() -> list:
    """Return sorted list of available states."""
    return sorted(history['state_name'].unique().tolist())

