import streamlit as st
import pandas as pd
import plotly.express as px

from model_config import (
    CHAMPION_MODELS,
    GENERAL_MODEL_RESULTS
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📈 Model Evaluation")

st.markdown(
    """
    Evaluate the forecasting performance of the candidate models
    and review the final deployed model selected for each food
    raw material commodity.
    """
)


# ============================================================
# 1. EVALUATION METHODOLOGY
# ============================================================

st.subheader("Evaluation Methodology")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.metric(
            label="Training Period",
            value="Jan 2014 – Dec 2024"
        )

with col2:
    with st.container(border=True):
        st.metric(
            label="Testing Period",
            value="Jan 2025 – May 2026"
        )

with col3:
    with st.container(border=True):
        st.metric(
            label="Evaluation Metric",
            value="MAPE"
        )

st.info(
    """
    Models are evaluated using the same hold-out testing period.
    MAPE is used as the primary evaluation metric because the
    five commodity groups have different import-value scales.
    Lower MAPE indicates lower forecasting error.
    """
)


st.markdown("---")


# ============================================================
# 2. FINAL CHAMPION MODELS
# ============================================================

st.subheader("Deployed Models Evaluation")

st.markdown(
    """
    These models are automatically used by the Forecasting page.
    Each commodity is assigned its selected final deployed model based
    on the experimental model-selection process.
    """
)


# ------------------------------------------------------------
# Create Champion Table
# ------------------------------------------------------------

champion_rows = []

for hs_code, config in CHAMPION_MODELS.items():

    champion_rows.append({
        "HS": hs_code,
        "Commodity": config["commodity"],
        "Champion Model": config["model_type"],
        "MAPE": config["mape"]
    })


champion_df = pd.DataFrame(champion_rows)


# ------------------------------------------------------------
# Display Champion Table
# ------------------------------------------------------------

st.dataframe(
    champion_df.style.format({
        "MAPE": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)


st.markdown("---")


# ============================================================
# 3. FINAL MODEL CONFIGURATION
# ============================================================

st.subheader("Deployed Model Configuration")

st.markdown(
    """
    Select a commodity to view the configuration of the model
    used by the deployed forecasting system.
    """
)


selected_commodity = st.selectbox(
    "Choose a commodity:",
    list(CHAMPION_MODELS.keys()),
    format_func=lambda hs: (
        f"{CHAMPION_MODELS[hs]['commodity']} (HS {hs})"
    )
)


selected_model = CHAMPION_MODELS[selected_commodity]


with st.container(border=True):

    st.markdown(
        f"### {selected_model['commodity']} "
        f"(HS {selected_commodity})"
    )

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.metric(
            label="Model",
            value=selected_model["model_type"]
        )

    with config_col2:

        st.metric(
            label="Validation MAPE",
            value=f"{selected_model['mape']:.2f}%"
        )


    st.markdown(
        f"""
        **Log Transformation:** \
        {'Applied' if selected_model['is_logged'] else 'Not Applied'}
        """
    )


    # --------------------------------------------------------
    # XGBoost Configuration
    # --------------------------------------------------------

    if selected_model["model_type"] == "XGBoost":

        st.markdown("### Features")

        st.write(
            ", ".join(
                selected_model["features"]
            )
        )

        st.markdown("### Hyperparameters")

        params_df = pd.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value
                }
                for key, value
                in selected_model["params"].items()
            ]
        )

        st.dataframe(
            params_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # SARIMA Configuration
    # --------------------------------------------------------

    elif selected_model["model_type"] == "SARIMA":

        st.markdown("### SARIMA Parameters")

        param_col1, param_col2 = st.columns(2)

        with param_col1:

            st.metric(
                label="Order",
                value=str(
                    selected_model["order"]
                )
            )

        with param_col2:

            st.metric(
                label="Seasonal Order",
                value=str(
                    selected_model["seasonal_order"]
                )
            )


st.markdown("---")


# ============================================================
# 4. EXPERIMENT PROCESS
# ============================================================

st.subheader("How Were the Deployed Models Selected?")

st.markdown(
    """
    The final deployed models were selected through multiple
    experimental stages. The purpose of this process was to compare
    forecasting approaches and investigate whether commodity-specific
    optimization could improve predictive performance.
    """
)


with st.expander("View Experimental Process"):

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    st.markdown(
        "### Experiment 1 — General Model Comparison"
    )

    st.write(
        """
        SARIMA and XGBoost were first evaluated using a general
        model configuration across the five commodity groups.
        This experiment established the initial performance
        comparison between the two forecasting approaches.
        """
    )


    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    st.markdown(
        "### Experiment 2 — Commodity-Specific Optimization"
    )

    st.write(
        """
        Subsequent experiments allowed the XGBoost configuration
        to vary by commodity. Different feature combinations,
        hyperparameters, and transformation choices were evaluated
        according to the characteristics of each commodity time series.
        """
    )


    # --------------------------------------------------------
    # Final Selection
    # --------------------------------------------------------

    st.markdown(
        "### Final Model Selection"
    )

    st.write(
        """
        The final forecasting system uses the selected deployed
        configuration for each commodity. These models are then
        retrained using the complete available dataset and used
        to generate future forecasts.
        """
    )


st.markdown("---")


# ============================================================
# 5. MODEL PERFORMANCE COMPARISON
# ============================================================

st.subheader("Model Performance Comparison")

st.markdown(
    """
    The following results show the initial comparison between
    SARIMA and XGBoost using the general model configuration.
    Lower MAPE indicates better forecasting performance.
    """
)


# ------------------------------------------------------------
# Convert results into DataFrame
# ------------------------------------------------------------

comparison_df = (
    pd.DataFrame
    .from_dict(
        GENERAL_MODEL_RESULTS,
        orient="index"
    )
    .reset_index()
    .rename(
        columns={
            "index": "HS_Code"
        }
    )
)


# ------------------------------------------------------------
# Comparison Table
# ------------------------------------------------------------

display_comparison = comparison_df[
    [
        "HS_Code",
        "commodity",
        "SARIMA Base",
        "SARIMA Tuned",
        "XGBoost Base",
        "XGBoost Tuned"
    ]
].copy()


display_comparison.rename(
    columns={
        "HS_Code": "HS",
        "commodity": "Commodity"
    },
    inplace=True
)


st.dataframe(
    display_comparison.style.format({
        "SARIMA Base": "{:.2f}%",
        "SARIMA Tuned": "{:.2f}%",
        "XGBoost Base": "{:.2f}%",
        "XGBoost Tuned": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)