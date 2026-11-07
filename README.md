FOOD RAW MATERIAL IMPORT VALUE DASHBOARD — REVISED CODE

Folder structure:
  app.py
  model_config.py
  pages/
    summary-page.py
    forecasting-page.py
    evaluation-page.py
  data/
    dataset_final_2014_2026.csv
    testing_metrics_sarima_xgboost.csv
    testing_predictions_sarima_xgboost.csv

The supplied dataset and two test-result CSV files are included in data/.

Install dependencies:
  pip install streamlit pandas numpy plotly statsmodels xgboost

Run from this project folder:
  streamlit run app.py

Important:
- Model parameters are locked from the two-fold expanding-window validation.
- Forecasting refits SARIMA and XGBoost on all 149 monthly observations
  (Jan 2014–May 2026), then generates June 2026–May 2027 forecasts.
- XGBoost is displayed as point forecasts only. It has no confidence interval.
- SARIMA intervals are model-based 95% forecast intervals on the log scale,
  exponentiated back to the original scale.
- Test metrics and predictions are loaded from the supplied test CSV files.
- The test period is not used to re-tune the locked model configurations.
