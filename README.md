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
│   ├── .ipynb_checkpoints
│   │   └── EDA-checkpoint.ipynb
│   ├── EDA.ipynb
│   └── analyze_src
│       ├── .ipynb_checkpoints
│       │   └── missing_values_analysis-checkpoint.py
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
├── mlflow.db
├── mlruns
│   └── 1
│       ├── 1ab2d31135cc417f9a06c0650a0194ef
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 258cf83c2e344764b30d38f0a1d0c343
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 5929bd8378334ccb84b439b938180f5d
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 713403a8ddbe4e2c91d62420a65e92f6
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 7498f383126f4b7b90ebf3058d4e547e
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 7c8ad9af69874a22820e5924cd1a0b34
│       │   └── artifacts
│       │       └── estimator.html
│       ├── 90a5f3144ade4472a9c0ca7cd80b4f67
│       │   └── artifacts
│       │       └── estimator.html
│       ├── afd3194a25fc4af9a2ff601501b1e802
│       │   └── artifacts
│       │       └── estimator.html
│       ├── cdc330c64f834ff891ab7115a912547d
│       │   └── artifacts
│       │       └── estimator.html
│       ├── dca2d0fe374543d4934a0fff5b21f289
│       │   └── artifacts
│       │       └── estimator.html
│       └── models
│           ├── m-11f0ad0b77a347068cc6f23f61625fc4
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-17e4024f41e142b18b0dd414fafde35d
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-21f80118b32b4e028517f91e87666736
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-3dce1fbcdbd64cc2a65f6b97324b5060
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-5edbe87107544a6e99802a106d47cae5
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-64f7d53313f94871b18cc2ad5dfe4c15
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-b84f9f24da854cc396de7e53abbc3f68
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-c32ac4da714647958bd97846379dce6d
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           ├── m-c8a8e50128d24577a7604de2ab15830f
│           │   └── artifacts
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           └── m-daca75dd38bb4288b1ab02db5c995f72
│               └── artifacts
│                   ├── MLmodel
│                   ├── conda.yaml
│                   ├── model.pkl
│                   ├── python_env.yaml
│                   └── requirements.txt
├── models
│   └── trained_pipeline.pkl
├── notebooks
│   ├── .ipynb_checkpoints
│   │   └── evaluation-checkpoint.ipynb
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
  "forecast": [
    {"date": "2026-05-03", "predicted_rainfall_mm": 0.05},
    {"date": "2026-05-04", "predicted_rainfall_mm": 0.03}
  ]
}
```

## ⚠️ Limitations

- State-level only — no district granularity
- Recursive forecasting — accuracy decreases after day 1
- Cannot predict sudden weather events (monsoon onset, cyclones)
- Requires atmospheric data for high-variability states like Meghalaya

## 📁 Dataset

- Source: [Kaggle — Indian Daily Rainfall Dataset](#) 
- 36 Indian states
- Daily granularity: 2009-2024
- ~200,000 rows

## 👤 Author
[Your Name] — [GitHub](https://github.com/avbormalviya) — [LinkedIn](https://www.linkedin.com/in/vimal-borana/)