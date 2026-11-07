import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import sys
from pathlib import Path

# Memasukkan folder 'dashboard' ke dalam sys.path Python
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.append(str(DASHBOARD_DIR))

from model_config import (
    COMMODITY_INFO, SARIMA_CONFIG, XGB_CONFIG, XGB_FEATURES,
    VALIDATION_MAPE, FORECAST_HORIZON, FORECAST_START, FORECAST_END,
    RANDOM_STATE,
)

# Project structure expected:
# project/
#   app.py, model_config.py, data/dataset_final_2014_2026.csv
#   pages/forecasting-page.py (or equivalent page directory)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_final_2014_2026.csv"

@st.cache_data
def load_historical_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = {"Period", "HS_Code", "Import_Value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data["Period"] = pd.to_datetime(data["Period"])
    data["HS_Code"] = data["HS_Code"].astype(str).str.zfill(2)
    data["Import_Value"] = pd.to_numeric(data["Import_Value"], errors="coerce")
    data = data.sort_values(["HS_Code", "Period"]).reset_index(drop=True)
    if data[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if (data["Import_Value"] <= 0).any():
        raise ValueError("Log modelling requires all Import_Value observations to be positive.")
    if data.duplicated(["HS_Code", "Period"]).any():
        raise ValueError("Duplicate HS_Code–Period records were found.")
    return data

def build_xgb_features(series):
    """Create supervised features from log values; all features use past values only."""
    y = series["log_value"]
    X = pd.DataFrame(index=series.index)
    X["lag_1"] = y.shift(1)
    X["lag_12"] = y.shift(12)
    X["rolling_mean_3"] = y.shift(1).rolling(3).mean()
    X["rolling_std_3"] = y.shift(1).rolling(3).std()
    X["month"] = series["Period"].dt.month
    supervised = X.copy()
    supervised["target"] = y
    supervised = supervised.dropna()
    return supervised[XGB_FEATURES], supervised["target"]

def recursive_xgb_forecast(model, history_log, future_dates):
    history = list(np.asarray(history_log, dtype=float))
    predictions = []
    for date in future_dates:
        if len(history) < 12:
            raise ValueError("At least 12 historical observations are required.")
        row = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_12": history[-12],
            "rolling_mean_3": np.mean(history[-3:]),
            "rolling_std_3": np.std(history[-3:], ddof=1),
            "month": date.month,
        }], columns=XGB_FEATURES)
        pred_log = float(model.predict(row)[0])
        predictions.append(float(np.exp(pred_log)))
        history.append(pred_log)
    return np.asarray(predictions)

@st.cache_data(show_spinner="Fitting models on the full historical dataset...")
def generate_forecasts(data):
    """Refit each locked model on Jan-2014–May-2026, then forecast 12 months."""
    rows = []
    future_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="MS")

    for hs, commodity in COMMODITY_INFO.items():
        part = data.loc[data["HS_Code"] == hs].sort_values("Period").copy()
        if part.empty:
            raise ValueError(f"No historical data found for HS {hs}.")
        if part["Period"].max() != pd.Timestamp("2026-05-01"):
            raise ValueError(
                f"HS {hs}: latest observation is {part['Period'].max():%b %Y}, "
                "but the agreed refit period ends May 2026."
            )
        if len(part) != 149:
            raise ValueError(f"HS {hs}: expected 149 monthly observations, found {len(part)}.")

        part["log_value"] = np.log(part["Import_Value"].astype(float))

        # SARIMA: refit fixed validation-selected orders on all 149 observations.
        sarima_spec = SARIMA_CONFIG[hs]
        sarima_fit = SARIMAX(
            part["log_value"],
            order=sarima_spec["order"],
            seasonal_order=sarima_spec["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        sarima_result = sarima_fit.get_forecast(steps=FORECAST_HORIZON)
        sarima_mean = np.exp(np.asarray(sarima_result.predicted_mean))

        # XGBoost: refit fixed features and hyperparameters on full log series.
        X_train, y_train = build_xgb_features(part)
        xgb = XGBRegressor(
            **XGB_CONFIG[hs],
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        xgb_values = recursive_xgb_forecast(
            xgb, part["log_value"].to_numpy(), future_dates
        )

        for i, date in enumerate(future_dates):
            rows.extend([
                {"Date": date, "HS_Code": hs, "Commodity": commodity,
                 "Model": "SARIMA", "Forecast_Value": sarima_mean[i]},
                {"Date": date, "HS_Code": hs, "Commodity": commodity,
                 "Model": "XGBoost", "Forecast_Value": xgb_values[i]},
            ])
    return pd.DataFrame(rows)

st.title("📈 Food Raw Material Import Value Forecast")
st.markdown(
    "Select a commodity and model to explore the 12-month forecast. "
    "Both models are refitted on the complete historical series through May 2026 "
    "using the configurations locked during validation."
)

try:
    historical_df = load_historical_data()
    forecast_all = generate_forecasts(historical_df)
except Exception as exc:
    st.error(f"Forecast could not be generated: {exc}")
    st.stop()

st.sidebar.title("Forecast Configuration")
hs_code = st.sidebar.selectbox(
    "Commodity group",
    list(COMMODITY_INFO),
    format_func=lambda hs: f"{COMMODITY_INFO[hs]} (HS {hs})",
)
model_name = st.sidebar.selectbox("Forecasting model", ["SARIMA", "XGBoost"])

commodity_forecast = forecast_all.query(
    "HS_Code == @hs_code and Model == @model_name"
).sort_values("Date").head(12).copy()
historical = historical_df.query("HS_Code == @hs_code").sort_values("Period")
val_mape = VALIDATION_MAPE[hs_code][model_name]

st.subheader(f"{COMMODITY_INFO[hs_code]} — HS {hs_code}")
c1, c2, c3 = st.columns(3)
c1.metric("Selected model", model_name)
c2.metric("Validation mean MAPE", f"{val_mape:.2f}%")
c3.metric("Forecast period", f"{commodity_forecast['Date'].min():%b %Y} – {commodity_forecast['Date'].max():%b %Y}")

st.caption(
    "Both SARIMA and XGBoost are displayed as point forecasts. "
    "No forecast interval is shown, so the two models can be compared consistently."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=historical["Period"], y=historical["Import_Value"],
    mode="lines", name="Historical actual",
))
fig.add_trace(go.Scatter(
    x=commodity_forecast["Date"], y=commodity_forecast["Forecast_Value"],
    mode="lines+markers", name=f"{model_name} forecast",
    line=dict(dash="dash", width=2),
))
fig.update_layout(
    title=f"{COMMODITY_INFO[hs_code]} — Historical and Forecasted Import Value",
    xaxis_title="Period", yaxis_title="Import Value (USD)",
    yaxis_tickprefix="$", yaxis_tickformat=",",
    hovermode="x unified", legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

table = commodity_forecast[["Date", "Forecast_Value"]].rename(columns={
    "Date": "Period",
    "Forecast_Value": "Forecast Value (USD)",
})
st.subheader("Forecast Details")
st.dataframe(
    table.style.format({c: "${:,.0f}" for c in table.columns if c != "Period"}),
    use_container_width=True, hide_index=True,
)
csv_bytes = commodity_forecast.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download forecast CSV", csv_bytes,
    file_name=f"forecast_HS_{hs_code}_{model_name}_12_months.csv",
    mime="text/csv",
)
