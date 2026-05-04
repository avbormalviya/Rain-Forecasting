from fastapi import FastAPI, HTTPException
from api.schemas import ForecastRequest, ForecastResponse
from api.predictor import get_forecast, get_available_states

app = FastAPI(
    title="Rain Forecasting API",
    description="7-day state-level rainfall forecast for India",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "running", "model": "LightGBM v1.0"}

@app.get("/states")
def states():
    return {"states": get_available_states()}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        predictions = get_forecast(request.state_name, request.start_date)
        return ForecastResponse(
            state_name=request.state_name,
            forecast=predictions
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))