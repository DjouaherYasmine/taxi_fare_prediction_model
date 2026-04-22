# NYC Taxi Fare Predictor 

A portfolio ML app that predicts NYC yellow taxi fares using XGBoost — both a regression model (exact fare) and a classification model (fare range: low / medium / high).

**Live demo:** https://taxifarepredictionmodel-zzxpfhbqkerndvzp9iefua.streamlit.app/

---

## Features used by the model

| Category | Features |
|---|---|
| Trip | distance, duration, avg speed |
| Time | hour (cyclic), day of week (cyclic), rush-hour flag, night flag, weekend, holiday |
| Location | pickup/dropoff zone ID, pickup/dropoff borough, airport trip flag |
| Weather | avg temperature, humidity, precipitation, snow, cloud cover, rain/snow flags |
| Demand | zone-hour avg fare, zone-hour ride count, zone-hour fare std |

---

## Model performance (January 2025 test set)

| Model | Metric | Value |
|---|---|---|
| XGBoost Regressor | MAE | ~$2.50 |
| XGBoost Regressor | R² | ~0.92 |
| XGBoost Classifier | Accuracy | ~0.87 |


