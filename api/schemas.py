from pydantic import BaseModel
from typing import List

class ForecastRequest(BaseModel):
    state_name: str
    start_date: str

class DayForecast(BaseModel):
    date: str
    predicted_rainfall_mm: float

class ForecastResponse(BaseModel):
    state_name: str
    forecast: List[DayForecast]