# 🌧️ India State-Level Rain Forecasting

A machine learning system that predicts daily rainfall for 36 Indian states, 
outperforming the government forecasting model by **34x** during peak monsoon months.

## 🎯 Results

| Metric | Government Model | Our Model |
|--------|-----------------|-----------|
| Overall MAE | 7.88 mm | 0.24 mm |
| July MAE | 22.37 mm | 0.84 mm |
| R² | - | 0.9874 |

### Live Validation (May 2026 — Unseen Data)

| State | Climate | Result |
|-------|---------|--------|
| Andaman & Nicobar | Tropical heavy rain | 🔥 ~0.03mm average error |
| Rajasthan | Desert dry | ✅ Near perfect |
| Maharashtra | Pre-monsoon dry | ✅ Near perfect |
| Kerala | Early monsoon onset | ⚠️ Missed (requires atmospheric data) |

## 🏗️ Project Structure

```
├── README.md
├── analysis
│   ├── EDA.ipynb
│   └── analyze_src
│       ├── basic_data_inspection.py
│       ├── bivariate_analysis.py
│       ├── missing_values_analysis.py
│       ├── multivariate_analysis.py
│       └── univariate_analysis.py
├── api
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── data
│   ├── X_test.csv
│   ├── archive.zip
│   ├── history_buffer.csv
│   ├── predictions.csv
│   ├── processed_data.csv
│   └── y_test.csv
├── extracted_data
│   └── daily-rainfall-at-state-level.csv
├── models
│   └── trained_pipeline.pkl
├── notebooks
│   └── evaluation.ipynb
├── pipelines
│   └── training_pipeline.py
├── reports
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── monthly_comparison.png
├── scripts
│   └── tune_hyperparams.py
├── src
│   ├── data_splitter.py
│   ├── feature_engineering.py
│   ├── handle_missing_values.py
│   ├── ingest_data.py
│   ├── model_evaluator.py
│   ├── outlier_detection.py
│   └── preprocess.py
├── steps
│   ├── data_ingestion_step.py
│   ├── data_splitter_step.py
│   ├── feature_engineering_step.py
│   ├── handle_missing_values_step.py
│   ├── model_building_step.py
│   └── model_evaluator_step.py
├── streamlit_app
│   ├── app.py
│   └── pages
│       ├── 1_forecast.py
│       ├── 2_performance.py
│       └── 3_about.py
└── tests
    └── run_pipeline.py
```

## 🔧 Tech Stack

- **ML:** LightGBM, Optuna, Scikit-learn
- **Pipeline:** ZenML
- **API:** FastAPI
- **App:** Streamlit
- **Data:** Indian state-level daily rainfall 2009-2024

## 📊 Approach

### Problem
Government rainfall model systematically overestimates during monsoon months.
July MAE of 22.37mm makes it practically useless for planning.

### Solution
1. **Feature Engineering** — lag features (1, 7, 30 days), rolling means, 
   dry streak, monsoon season flag
2. **Model** — LightGBM with Optuna hyperparameter tuning
3. **Monsoon Weighting** — 4x sample weight for July/August during training
4. **Validation** — time-based split (train: 2009-2021, test: 2022-2024)

### Key Insight
Government forecast (`rfs`) used as an input feature — model learns 
when to trust and when to correct the government prediction.

## 🚀 Running Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training pipeline
```bash
python run_pipeline.py
```

### Run API
```bash
uvicorn api.main:app --reload
```

### Run Streamlit app
```bash
streamlit run streamlit_app/app.py
```

## 📡 API Usage

### Get 7-day forecast
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"state_name": "Maharashtra", "start_date": "2026-05-03"}'
```

### Response
```json
{
  "state_name": "Maharashtra",
  "forecast": [[1_forecast.py](streamlit_app/pages/1_forecast.py)
    {"date": "2026-05-03", "predicted_rainfall_mm": 0.05},
    {"date": "2026-05-04", "predicted_rainfall_mm": 0.03}
  ]
}
```

## 🌐 Live API
Base URL: `https://web-production-87ad8.up.railway.app`

- Docs: https://web-production-87ad8.up.railway.app/docs
- States: https://web-production-87ad8.up.railway.app/states
- Forecast: POST https://web-production-87ad8.up.railway.app/forecast

## ⚠️ Limitations

- State-level only — no district granularity
- Recursive forecasting — accuracy decreases after day 1
- Cannot predict sudden weather events (monsoon onset, cyclones)
- Requires atmospheric data for high-variability states like Meghalaya

## 📁 Dataset

- Source: [Kaggle — Indian Daily Rainfall Dataset](https://www.kaggle.com/datasets/vimalborana/india-state-level-daily-rainfall-2009-2024/)
- 36 Indian states
- Daily granularity: 2009-2024
- ~200,000 rows

## 👤 Author
Vimal borana — [GitHub](https://github.com/avbormalviya) — [LinkedIn](https://www.linkedin.com/in/vimal-borana/)