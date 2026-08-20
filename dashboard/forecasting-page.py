import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from model_config import CHAMPION_MODELS

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORICAL_DATA_PATH = (
    BASE_DIR / "data" / "dataset_final_2014_2026.csv"
)

FORECAST_DATA_PATH = (
    BASE_DIR / "data" / "forecast_results.csv"
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_historical_data():
    df = pd.read_csv(HISTORICAL_DATA_PATH)

    df['Period'] = pd.to_datetime(df['Period'])

    # Make HS codes consistent with the dashboard
    df['HS_Code'] = (
        df['HS_Code']
        .astype(str)
        .str.zfill(2)
    )

    return df


@st.cache_data
def load_forecast_data():
    forecast_df = pd.read_csv(FORECAST_DATA_PATH)

    forecast_df['Date'] = pd.to_datetime(
        forecast_df['Date']
    )

    # CSV contains 4, 7, 10, 12, 17
    # Convert them into 04, 07, 10, 12, 17
    forecast_df['HS_Code'] = (
        forecast_df['HS_Code']
        .astype(str)
        .str.zfill(2)
    )

    return forecast_df


df = load_historical_data()
forecast_df = load_forecast_data()

# ============================================================
# COMMODITY MAPPING
# ============================================================

HS_MAPPING = {
    'Dairy & Honey (HS 04)': '04',
    'Vegetables (HS 07)': '07',
    'Cereals (HS 10)': '10',
    'Seeds & Oleaginous Fruits (HS 12)': '12',
    'Sugar (HS 17)': '17'
}

# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "📈 Food Raw Materials Import Value Forecast"
)

st.markdown(
    """
    Explore future import value forecasts for each food raw
    material commodity. Select a commodity and forecasting
    horizon to view the projected import value and its
    associated uncertainty.
    """
)

# ============================================================
# FORECAST CONFIGURATION
# ============================================================

st.sidebar.title("Forecast Configuration")

commodity_selectbox = st.sidebar.selectbox(
    "Choose a commodity group:",
    list(HS_MAPPING.keys())
)

hs_code = HS_MAPPING[commodity_selectbox]


forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon:",
    [1, 3, 6, 12, 24],
    index=3,
    format_func=lambda x: f"{x} Month" if x == 1 else f"{x} Months"
)

# ============================================================
# GET CHAMPION MODEL INFORMATION
# ============================================================

model_info = CHAMPION_MODELS[hs_code]

model_name = model_info['model_type']
mape = model_info['mape']


# ============================================================
# FILTER FORECAST DATA
# ============================================================

commodity_forecast = (
    forecast_df[
        forecast_df['HS_Code'] == hs_code
    ]
    .sort_values('Date')
    .copy()
)


# Safety check
if commodity_forecast.empty:

    st.error(
        "No forecast data is available for the selected commodity."
    )

    st.stop()


# Select only the requested horizon
forecast_display = commodity_forecast.head(
    forecast_horizon
).copy()

# ============================================================
# FORECAST DATE INFORMATION
# ============================================================

forecast_start = forecast_display['Date'].min()
forecast_end = forecast_display['Date'].max()


# ============================================================
# SUMMARY CARDS
# ============================================================

st.subheader(
    f"Forecast Overview — {commodity_selectbox}"
)

col1, col2, col3 = st.columns(3)


with col1:
    with st.container(border=True):

        st.metric(
            label="Forecast Model",
            value=model_name
        )


with col2:
    with st.container(border=True):

        st.metric(
            label="Validation MAPE",
            value=f"{mape:.2f}%"
        )


with col3:
    with st.container(border=True):

        st.metric(
            label="Forecast Period",
            value=(
                f"{forecast_start.strftime('%b %Y')}"
                f" – "
                f"{forecast_end.strftime('%b %Y')}"
            )
        )


# ============================================================
# INFORMATION
# ============================================================

st.info(
    """
    The forecast represents the expected monthly import value
    for the selected commodity. The shaded area represents
    the 95% confidence interval and reflects forecast uncertainty.
    """
)


st.markdown("---")

# ============================================================
# HISTORICAL + FORECAST DATA
# ============================================================

historical = (
    df[df['HS_Code'] == hs_code]
    .sort_values('Period')
    .copy()
)


# ============================================================
# FORECAST CHART
# ============================================================

st.subheader("Import Value Forecast")


fig = go.Figure()

# ------------------------------------------------------------
# Historical Actual
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=historical['Period'],
        y=historical['Import_Value'],
        mode='lines',
        name='Historical Actual',
        line=dict(
            width=2
        ),
        hovertemplate=(
            "%{x|%b %Y}"
            "<br>"
            "Import Value: $%{y:,.0f}"
            "<extra></extra>"
        )
    )
)

# ------------------------------------------------------------
# Forecast
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=forecast_display['Date'],
        y=forecast_display['Forecast_Value'],
        mode='lines+markers',
        name='Forecast',
        line=dict(
            dash='dash',
            width=2
        ),
        hovertemplate=(
            "%{x|%b %Y}"
            "<br>"
            "Forecast: $%{y:,.0f}"
            "<extra></extra>"
        )
    )
)


# ------------------------------------------------------------
# Upper Confidence Interval
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=forecast_display['Date'],
        y=forecast_display['Upper_CI'],
        mode='lines',
        line=dict(
            width=0
        ),
        name='Upper 95% CI',
        showlegend=False,
        hoverinfo='skip'
    )
)


# ------------------------------------------------------------
# Lower Confidence Interval
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=forecast_display['Date'],
        y=forecast_display['Lower_CI'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(100, 149, 237, 0.20)',
        line=dict(
            width=0
        ),
        name='95% Confidence Interval',
        hovertemplate=(
            "%{x|%b %Y}"
            "<br>"
            "Lower CI: $%{y:,.0f}"
            "<extra></extra>"
        )
    )
)


# ============================================================
# CHART LAYOUT
# ============================================================

fig.update_layout(

    title=(
        f"{commodity_selectbox} — "
        f"Historical and Forecasted Import Value"
    ),

    xaxis_title="Period",

    yaxis_title="Import Value (USD)",

    yaxis_tickprefix="$",

    yaxis_tickformat=",",

    hovermode="x unified",

    dragmode="zoom",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    margin=dict(
        l=20,
        r=20,
        t=80,
        b=20
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# FORECAST TABLE
# ============================================================

# Prepare download data first
download_df = forecast_display[
    [
        'Date',
        'HS_Code',
        'Forecast_Value',
        'Lower_CI',
        'Upper_CI',
        'Model_Used'
    ]
].copy()

download_df['Date'] = (
    download_df['Date']
    .dt.strftime('%Y-%m-%d')
)

csv_data = download_df.to_csv(
    index=False
).encode('utf-8')


# ------------------------------------------------------------
# Table title + download button
# ------------------------------------------------------------

table_col, download_col = st.columns([5, 1])

with table_col:
    st.subheader("Forecast Details")

with download_col:
    st.write("")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name=(
            f"forecast_HS_{hs_code}_"
            f"{forecast_horizon}_months.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ------------------------------------------------------------
# Display table
# ------------------------------------------------------------

forecast_table = forecast_display[
    [
        'Date',
        'Forecast_Value',
        'Lower_CI',
        'Upper_CI'
    ]
].copy()


forecast_table['Date'] = (
    forecast_table['Date']
    .dt.strftime('%b %Y')
)


forecast_table = forecast_table.rename(
    columns={
        'Date': 'Period',
        'Forecast_Value': 'Forecast Value (USD)',
        'Lower_CI': 'Lower 95% CI (USD)',
        'Upper_CI': 'Upper 95% CI (USD)'
    }
)


st.dataframe(
    forecast_table.style.format({
        'Forecast Value (USD)': '${:,.0f}',
        'Lower 95% CI (USD)': '${:,.0f}',
        'Upper 95% CI (USD)': '${:,.0f}'
    }),
    use_container_width=True,
    hide_index=True
)