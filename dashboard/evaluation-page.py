import streamlit as st
import pandas as pd
import plotly.express as px

from model_config import (
    CHAMPION_MODELS,
    GENERAL_MODEL_RESULTS
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📈 Food Raw Material Import Value Model Evaluation")

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
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Training Period</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'Jan 2014 – Dec 2024'
            f'</div>',
            unsafe_allow_html=True
        )


with col2:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Testing Period</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'Jan 2025 – May 2026'
            f'</div>',
            unsafe_allow_html=True
        )


with col3:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Evaluation Metric</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'MAPE'
            f'</div>',
            unsafe_allow_html=True
        )

st.info(
    """
    Models are evaluated using the same hold-out testing period.
    MAPE is used as the primary evaluation metric because the
    five commodity groups have different import value scales.
    **Lower MAPE indicates lower forecasting error.**
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
    The final deployed models were selected through two experimental
    stages. The first experiment compared SARIMA and XGBoost under
    consistent general configurations, while the second experiment
    optimized each local model according to the characteristics of
    each commodity.
    """
)


# ============================================================
# Experiment 2 Results
# ============================================================

EXPERIMENT_2_RESULTS = {
    "04": {
        "Commodity": "Dairy & Honey",
        "SARIMA": 14.63,
        "XGBoost": 10.37
    },

    "07": {
        "Commodity": "Vegetables",
        "SARIMA": 17.88,
        "XGBoost": 19.68
    },

    "10": {
        "Commodity": "Cereals",
        "SARIMA": 77.51,
        "XGBoost": 30.14
    },

    "12": {
        "Commodity": "Seeds & Oleaginous Fruits",
        "SARIMA": 25.36,
        "XGBoost": 19.76
    },

    "17": {
        "Commodity": "Sugar",
        "SARIMA": 71.53,
        "XGBoost": 36.18
    }
}


with st.expander("View Experimental Process"):

    # ========================================================
    # EXPERIMENT 1
    # ========================================================

    st.markdown(
        "### Experiment 1 — General Model Comparison"
    )

    st.write(
        """
        SARIMA and XGBoost were first evaluated using consistent
        general configurations across all five commodity groups.

        The same modeling configuration within each approach and
        the same log-transformed target treatment were applied to
        every commodity. This experiment provides a controlled
        baseline comparison of the two forecasting approaches.
        """
    )


    # --------------------------------------------------------
    # Experiment 1 Comparison Table
    # --------------------------------------------------------

    experiment_1_df = (
        pd.DataFrame
        .from_dict(
            GENERAL_MODEL_RESULTS,
            orient="index"
        )
        .reset_index()
        .rename(
            columns={
                "index": "HS"
            }
        )
    )


    # Keep only Experiment 1 results
    experiment_1_display = experiment_1_df[
        [
            "HS",
            "commodity",
            "SARIMA Base",
            "XGBoost Base",
            "SARIMA Tuned",
            "XGBoost Tuned"
        ]
    ].copy()


    experiment_1_display.rename(
        columns={
            "commodity": "Commodity",
            "SARIMA Base": "SARIMA Base",
            "XGBoost Base": "XGBoost Base",
            "SARIMA Tuned": "SARIMA Tuned",
            "XGBoost Tuned": "XGBoost Tuned"
        },
        inplace=True
    )


    st.dataframe(
        experiment_1_display.style.format({
            "SARIMA Base": "{:.2f}%",
            "XGBoost Base": "{:.2f}%",
            "SARIMA Tuned": "{:.2f}%",
            "XGBoost Tuned": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Lower MAPE indicates better forecasting performance."
    )
    
    st.info(
        """
        **Interpretation:** The general comparison shows that model
        performance varies across commodities, indicating that one
        forecasting approach does not consistently perform best for
        every time series. XGBoost tends to achieve lower MAPE for
        several commodities, while SARIMA remains competitive for
        commodities with more regular historical patterns. This
        suggests that model selection should consider the
        characteristics of each commodity rather than applying one
        model universally.
        """
    )

    st.markdown("---")


    # ========================================================
    # EXPERIMENT 2
    # ========================================================

    st.markdown(
        "### Experiment 2 — Commodity-Specific Optimization"
    )

    st.write(
        """
        In the second experiment, SARIMA and XGBoost were optimized
        separately for each commodity.

        Model parameters, feature configurations, and target
        transformation choices were evaluated to identify the most
        suitable configuration for each local commodity model.
        The table below compares the best SARIMA and XGBoost results
        obtained for each commodity.
        """
    )


    # --------------------------------------------------------
    # Experiment 2 Comparison Table
    # --------------------------------------------------------

    experiment_2_rows = []

    for hs_code, result in EXPERIMENT_2_RESULTS.items():

        experiment_2_rows.append({
            "HS": hs_code,
            "Commodity": result["Commodity"],
            "SARIMA": result["SARIMA"],
            "XGBoost": result["XGBoost"]
        })


    experiment_2_df = pd.DataFrame(
        experiment_2_rows
    )


    st.dataframe(
        experiment_2_df.style.format({
            "SARIMA": "{:.2f}%",
            "XGBoost": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Lower MAPE indicates better forecasting performance."
    )
    
    st.info(
        """
        **Interpretation:** After commodity-specific optimization,
        XGBoost achieved lower MAPE for 4 of the 5 commodity
        groups, while SARIMA achieved the lowest MAPE for
        HS 07 (Vegetables). The advantage of XGBoost is particularly large for HS 10
        and HS 17, which also exhibit substantial historical
        fluctuations. However, the results also show that no model
        is universally superior, as SARIMA remains more accurate
        for HS 07.
        """
    )

    st.markdown("---")


    # ========================================================
    # FINAL MODEL SELECTION
    # ========================================================

    st.markdown(
        "### Final Model Selection"
    )

    st.write(
        """
        The best-performing configuration for each commodity was
        selected based on the experimental results.

        The selected models were then retrained using the complete
        available historical dataset and are used by the Forecasting
        page to generate future import-value forecasts.
        """
    )