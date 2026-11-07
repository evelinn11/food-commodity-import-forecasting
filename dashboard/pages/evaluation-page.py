import sys

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.append(str(DASHBOARD_DIR))

from model_config import (
    COMMODITY_INFO, SARIMA_CONFIG, XGB_CONFIG, XGB_FEATURES,
    VALIDATION_MAPE, VALIDATION_FOLD_MAPE,
    TRAIN_START, DEVELOPMENT_END, TEST_START, TEST_END,
    FORECAST_START, FORECAST_END,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = BASE_DIR / "data" / "testing_metrics_sarima_xgboost.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "testing_predictions_sarima_xgboost.csv"

@st.cache_data
def load_test_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Test metrics file not found: {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH)
    df["HS_Code"] = df["HS_Code"].astype(str).str.zfill(2)
    return df

@st.cache_data
def load_test_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Test predictions file not found: {PREDICTIONS_PATH}")
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Period"] = pd.to_datetime(df["Period"])
    df["HS_Code"] = df["HS_Code"].astype(str).str.zfill(2)
    return df

st.title("📋 Food Raw Material Import Value Model Evaluation")
st.markdown(
    "Review the validation procedure, locked model configurations, "
    "and out-of-sample performance on the historical test period."
)

st.subheader("Evaluation timeline")
a, b, c = st.columns(3)
a.metric("Model development / training", "Jan 2014 – May 2025")
b.metric("Hold-out testing", "Jun 2025 – May 2026")
c.metric("Future forecast", "Jun 2026 – May 2027")

st.info(
    "Two expanding-window folds were used to select each commodity-specific "
    "SARIMA and XGBoost configuration. After selection, configurations were locked. "
    "The hold-out test period was not used to tune parameters. For deployment, "
    "each locked configuration is refitted using all observations through May 2026."
)

st.subheader("1. Expanding-window validation design")
fold_df = pd.DataFrame([
    {"Fold": "Fold 1", "Training period": "Jan 2014 – May 2023",
     "Validation period": "Jun 2023 – May 2024", "Validation horizon": "12 months"},
    {"Fold": "Fold 2", "Training period": "Jan 2014 – May 2024",
     "Validation period": "Jun 2024 – May 2025", "Validation horizon": "12 months"},
])
st.dataframe(fold_df, use_container_width=True, hide_index=True)
st.caption(
    "The selected configuration minimizes mean validation MAPE across the two folds; "
    "MAE and RMSE were used as tie-breakers."
)

st.subheader("2. Locked configurations and validation performance")
config_rows = []
for hs, commodity in COMMODITY_INFO.items():
    sar = SARIMA_CONFIG[hs]
    xgb = XGB_CONFIG[hs]
    config_rows.append({
        "HS": hs, "Commodity": commodity,
        "SARIMA order": str(sar["order"]),
        "SARIMA seasonal order": str(sar["seasonal_order"]),
        "SARIMA validation MAPE": VALIDATION_MAPE[hs]["SARIMA"],
        "XGBoost validation MAPE": VALIDATION_MAPE[hs]["XGBoost"],
    })
config_df = pd.DataFrame(config_rows)
st.dataframe(
    config_df.style.format({
        "SARIMA validation MAPE": "{:.2f}%",
        "XGBoost validation MAPE": "{:.2f}%",
    }),
    use_container_width=True, hide_index=True,
)

with st.expander("View validation MAPE by fold"):
    fold_rows = []
    for hs, commodity in COMMODITY_INFO.items():
        for model in ["SARIMA", "XGBoost"]:
            values = VALIDATION_FOLD_MAPE[hs][model]
            fold_rows.append({
                "HS": hs, "Commodity": commodity, "Model": model,
                "Fold 1 MAPE": values[0], "Fold 2 MAPE": values[1],
                "Mean MAPE": VALIDATION_MAPE[hs][model],
            })
    fold_table = pd.DataFrame(fold_rows)
    st.dataframe(
        fold_table.style.format({
            "Fold 1 MAPE": "{:.2f}%", "Fold 2 MAPE": "{:.2f}%",
            "Mean MAPE": "{:.2f}%",
        }),
        use_container_width=True, hide_index=True,
    )

with st.expander("View locked XGBoost features and hyperparameters"):
    st.write("Features used for every commodity:", ", ".join(XGB_FEATURES))
    xgb_rows = [
        {"HS": hs, "Commodity": COMMODITY_INFO[hs], **params}
        for hs, params in XGB_CONFIG.items()
    ]
    st.dataframe(pd.DataFrame(xgb_rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("3. Hold-out test performance")
st.markdown(
    "The following metrics summarize the fixed 12-month test forecasts "
    "for June 2025–May 2026. Lower MAPE indicates lower percentage error."
)

try:
    metrics = load_test_metrics()
    predictions = load_test_predictions()
except Exception as exc:
    st.error(f"Could not load test results: {exc}")
    st.stop()

metrics["Commodity"] = metrics["HS_Code"].map(COMMODITY_INFO)
metrics["Best_Model_Test"] = metrics["Best_Model_Test"].astype(str)
summary = metrics[[
    "HS_Code", "Commodity", "SARIMA_MAPE", "XGBoost_MAPE", "Best_Model_Test"
]].copy()
summary = summary.rename(columns={
    "HS_Code": "HS",
    "SARIMA_MAPE": "SARIMA MAPE (%)",
    "XGBoost_MAPE": "XGBoost MAPE (%)",
    "Best_Model_Test": "Lower test MAPE",
})
st.dataframe(
    summary.style.format({
        "SARIMA MAPE (%)": "{:.2f}%",
        "XGBoost MAPE (%)": "{:.2f}%",
    }),
    use_container_width=True, hide_index=True,
)

sarima_macro = metrics["SARIMA_MAPE"].mean()
xgb_macro = metrics["XGBoost_MAPE"].mean()
m1, m2, m3 = st.columns(3)
m1.metric("Mean commodity MAPE — SARIMA", f"{sarima_macro:.2f}%")
m2.metric("Mean commodity MAPE — XGBoost", f"{xgb_macro:.2f}%")
m3.metric(
    "Commodities with lower XGBoost MAPE",
    f"{int((metrics['XGBoost_MAPE'] < metrics['SARIMA_MAPE']).sum())} of {len(metrics)}",
)
st.caption(
    "The mean values above are unweighted arithmetic averages of the five "
    "commodity-level MAPE scores; they are not pooled MAPE across all monthly observations."
)

plot_df = summary.melt(
    id_vars=["HS", "Commodity"],
    value_vars=["SARIMA MAPE (%)", "XGBoost MAPE (%)"],
    var_name="Model", value_name="MAPE (%)",
)
fig = px.bar(
    plot_df, x="Commodity", y="MAPE (%)", color="Model",
    barmode="group", title="Hold-out test MAPE by commodity",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Actual versus test predictions")
selected_hs = st.selectbox(
    "Select commodity",
    list(COMMODITY_INFO),
    format_func=lambda hs: f"{COMMODITY_INFO[hs]} (HS {hs})",
)
actual_plot = predictions.query("HS_Code == @selected_hs").sort_values("Period")
fig_pred = px.line(
    actual_plot, x="Period",
    y=["Actual", "SARIMA_Prediction", "XGBoost_Prediction"],
    title=f"Test-period actual and model predictions — HS {selected_hs}",
    labels={"value": "Import Value (USD)", "variable": "Series", "Period": "Month"},
)
fig_pred.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",", hovermode="x unified")
st.plotly_chart(fig_pred, use_container_width=True)

with st.expander("View test prediction data"):
    st.dataframe(actual_plot, use_container_width=True, hide_index=True)

st.warning(
    "Test-period performance is a retrospective comparison, not a reason to "
    "change the locked configurations. The dashboard lets users choose either "
    "model for forecasting; it does not declare one universal champion."
)
