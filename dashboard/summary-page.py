import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# Page Constants
# ============================================================

COMMODITY_OPTIONS = {
    "Dairy & Honey (HS 04)": "04",
    "Vegetables (HS 07)": "07",
    "Cereals (HS 10)": "10",
    "Seeds & Oleaginous Fruits (HS 12)": "12",
    "Sugar (HS 17)": "17",
}

COMMODITY_TABLE = pd.DataFrame(
    {
        "HS Code": ["04", "07", "10", "12", "17"],
        "Commodity Group": [
            "Dairy & Honey",
            "Vegetables",
            "Cereals",
            "Seeds & Oleaginous Fruits",
            "Sugar",
        ],
    }
)


# ============================================================
# Data Loading
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset_final_2014_2026.csv")

    # Convert Period into datetime
    df["Period"] = pd.to_datetime(df["Period"])

    # Keep HS codes in two-digit format
    df["HS_Code"] = df["HS_Code"].astype(str).str.zfill(2)

    # Sort chronologically
    df = df.sort_values(
        ["HS_Code", "Period"]
    ).reset_index(drop=True)

    return df


df = load_data()


# ============================================================
# Dataset Summary Calculations
# ============================================================

start_period = df["Period"].min()
end_period = df["Period"].max()

num_records = len(df)
num_commodities = df["HS_Code"].nunique()

# Data quality checks
missing_values = int(
    df.isna().sum().sum()
)

duplicate_rows = int(
    df.duplicated(
        subset=["Period", "HS_Code"]
    ).sum()
)

# Check whether every commodity has a complete monthly sequence
expected_periods = pd.date_range(
    start=start_period,
    end=end_period,
    freq="MS"
)

missing_period_count = 0

for hs_code in df["HS_Code"].unique():

    actual_periods = set(
        df.loc[
            df["HS_Code"] == hs_code,
            "Period"
        ]
    )

    missing_period_count += len(
        set(expected_periods) - actual_periods
    )


# ============================================================
# Header
# ============================================================

st.title("📕 Food Raw Material Import Value Forecasting")

st.subheader(
    "Summary"
)

st.markdown(
    """
    An interactive forecasting dashboard designed to help food
    business owners understand historical import-value movements
    and anticipate future changes in selected food raw-material
    commodity groups in Indonesia.
    """
)

st.info(
    """
    The dashboard uses official monthly import-value data from
    Badan Pusat Statistik (BPS) Indonesia. Forecasts are intended as an additional planning reference,
    not as a direct prediction of the ingredient price paid by
    an individual business.
    """
)

st.divider()


# ============================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================

st.header("1. Business Understanding")

left, right = st.columns(2)


# -------------------------------
# Business Problem
# -------------------------------

with left:

    with st.container(border=True):

        st.subheader("Business Problem")

        st.markdown(
            """
            Food businesses depend on stable access to raw materials
            to manage production costs and protect profit margins.

            Changes in imported food raw materials can indicate shifts
            in the external market environment, making procurement and
            financial planning more difficult.
            """
        )

        st.markdown(
            """
            When input costs rise, business owners may need to
            reconsider purchasing quantities, budgets, inventory plans,
            or selling prices. However, increasing selling prices can
            also affect customer demand.
            """
        )


# -------------------------------
# Project Objective
# -------------------------------

with right:

    with st.container(border=True):

        st.subheader("Project Objective")

        st.markdown(
            """
            This project forecasts the monthly import value of selected
            food raw-material commodity groups in Indonesia.

            The goal is to provide food business owners with a
            forward-looking reference that complements historical trends
            when preparing procurement and other business decisions.
            """
        )

        st.markdown(
            """
            The forecasting study compares statistical and
            machine-learning approaches, while the dashboard makes the
            final results easier to explore and interpret.
            """
        )


# ============================================================
# Application Benefits
# ============================================================

st.subheader("How This Application Can Help")

benefit_1, benefit_3, benefit_4 = st.columns(3)


with benefit_1:

    with st.container(border=True):

        st.markdown("#### 📦 Procurement Planning")

        st.write(
            """
            Use expected import-value movements as an additional
            reference when preparing future purchasing plans.
            """
        )

with benefit_3:

    with st.container(border=True):

        st.markdown("#### 🏷️ Pricing Consideration")

        st.write(
            """
            Support earlier evaluation of whether cost changes may
            require adjustments to pricing or product strategy.
            """
        )


with benefit_4:

    with st.container(border=True):

        st.markdown("#### 📊 Decision Support")

        st.write(
            """
            Combine historical patterns and forecasts to add a
            data-driven reference to business planning decisions.
            """
        )


# Important limitation
st.warning(
    """
    Important: Import value is not the same as market price or a
    business's actual ingredient cost. Import value can be influenced
    by import quantity, prices, exchange rates, and other market factors.
    """
)

st.divider()


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

st.header("2. Dataset Overview")

st.markdown(
    """
    The dataset contains monthly Indonesian import values for five
    selected food raw-material commodity groups. The data was collected from official BPS import statistics and
    consolidated into one time-series dataset.
    """
)


# Dataset cards
metric_2, metric_3, metric_4 = st.columns(3)

with metric_2:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Period</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'{start_period.strftime("%b %Y")} – {end_period.strftime("%b %Y")}'
            f'</div>',
            unsafe_allow_html=True
        )

with metric_3:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Observations</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'{num_records:,}'
            f'</div>',
            unsafe_allow_html=True
        )

with metric_4:
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">Commodity Groups</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:16px;">'
            f'{num_commodities}'
            f'</div>',
            unsafe_allow_html=True
        )

st.caption(
    "Frequency: Monthly | Target Variable: Import Value (USD) | Source: BPS Indonesia"
)


# ============================================================
# Commodity Table
# ============================================================

st.subheader("Commodity Groups")

st.dataframe(
    COMMODITY_TABLE,
    hide_index=True,
    use_container_width=True
)

st.divider()


# ============================================================
# 3. DATA PREPARATION
# ============================================================

st.header("3. Data Preparation")

st.markdown(
    """
    The original BPS dataset requires restructuring before they could be used
    for time-series analysis and forecasting.
    """
)


# ============================================================
# Preparation Row 1
# ============================================================

prep_1, prep_2, prep_3 = st.columns(3)


with prep_1:

    with st.container(border=True):

        st.markdown("#### 1️⃣ Clean Raw BPS Files")

        st.write(
            """
            Removed non-data header and total rows and forward-filled
            merged HS-code cells from the original Excel structure.
            """
        )


with prep_2:

    with st.container(border=True):

        st.markdown("#### 2️⃣ Extract & Standardize")

        st.write(
            """
            Extracted HS codes, standardized them into two-digit format,
            and converted import values into numeric format.
            """
        )


with prep_3:

    with st.container(border=True):

        st.markdown("#### 3️⃣ Reshape Monthly Data")

        st.write(
            """
            Converted months columns from wide format into
            a monthly long-format containing Period,
            HS_Code, and Import_Value.
            """
        )


# ============================================================
# Preparation Row 2
# ============================================================

prep_4, prep_5, prep_6 = st.columns(3)


with prep_4:

    with st.container(border=True):

        st.markdown("#### 4️⃣ Combine Datasets")

        st.write(
            """
            Combined BPS datasets covering 2014–2016, 2017–2021,
            and 2022–2026, then retained observations through May 2026.
            """
        )


with prep_5:

    with st.container(border=True):

        st.markdown("#### 5️⃣ Validate Data Quality")

        st.write(
            """
            Checked missing values, duplicate Period–HS combinations,
            and missing monthly periods for every commodity group.
            """
        )


with prep_6:

    with st.container(border=True):

        st.markdown("#### 6️⃣ Prepare for Modeling")

        st.write(
            """
            Examined distribution skewness, evaluated log
            transformation, and used a chronological train/test
            split for model evaluation.
            """
        )


# Raw dataset is optional instead of always filling the page
with st.expander("View Final Prepared Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

st.header("4. Exploratory Data Analysis")

st.markdown(
    """
    Explore the historical behavior of each commodity before moving
    to the Forecasting page. The filters below only affect this
    Exploratory Data Analysis section.
    """
)


# ============================================================
# Commodity Filter
# ============================================================

commodity_selectbox = st.selectbox(
    "Choose a commodity group:",
    list(COMMODITY_OPTIONS.keys())
)

hs_code = COMMODITY_OPTIONS[
    commodity_selectbox
]

df_filtered = (
    df[
        df["HS_Code"] == hs_code
    ]
    .copy()
    .sort_values("Period")
)


# ============================================================
# YoY Growth
# ============================================================

# Important:
# Calculate YoY BEFORE filtering the visible date range.
#
# This ensures pct_change(periods=12) still has access to
# observations from 12 months earlier.

df_filtered["YoY_Growth_Pct"] = (
    df_filtered["Import_Value"]
    .pct_change(periods=12)
    * 100
)


# ============================================================
# Date Filter
# ============================================================

min_date = (
    df_filtered["Period"]
    .min()
    .date()
)

max_date = (
    df_filtered["Period"]
    .max()
    .date()
)


date_selection = st.slider(
    "Filter chart timeline:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY"
)


start_date = pd.to_datetime(
    date_selection[0]
)

end_date = pd.to_datetime(
    date_selection[1]
)


df_chart_view = df_filtered[
    (
        df_filtered["Period"]
        >= start_date
    )
    &
    (
        df_filtered["Period"]
        <= end_date
    )
].copy()


# ============================================================
# Import Value Trend
# ============================================================

with st.container(border=True):

    fig_line = px.line(
        df_chart_view,
        x="Period",
        y="Import_Value",
        title=(
            f"Import Value Trend — "
            f"{commodity_selectbox}"
        ),
        labels={
            "Period": "Period",
            "Import_Value": "Import Value (USD)"
        },
    )


    # Average value for visible date range
    current_mean = (
        df_chart_view["Import_Value"]
        .mean()
    )


    fig_line.add_hline(
        y=current_mean,
        line_dash="dash",
        line_color="gray",
        annotation_text="Selected-period average",
        annotation_position="top left",
        opacity=0.7,
    )


    fig_line.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",.3s",
        hovermode="x unified",
        dragmode="zoom",
    )


    st.plotly_chart(
        fig_line,
        use_container_width=True
    )


# ============================================================
# YoY & Distribution Charts
# ============================================================

chart_left, chart_right = st.columns(2)


# -------------------------------
# YoY Growth
# -------------------------------

with chart_left:

    with st.container(border=True):

        yoy_colors = [
            "#d62728"
            if value < 0
            else "#2ca02c"

            for value
            in df_chart_view[
                "YoY_Growth_Pct"
            ].fillna(0)
        ]


        fig_yoy = px.bar(
            df_chart_view,
            x="Period",
            y="YoY_Growth_Pct",
            title="Year-over-Year Growth Rate",
            labels={
                "Period": "Period",
                "YoY_Growth_Pct":
                    "YoY Growth Rate (%)"
            },
        )


        fig_yoy.update_traces(
            marker_color=yoy_colors
        )


        fig_yoy.update_layout(
            yaxis_ticksuffix="%",
            yaxis_tickformat=",.1f",
            hovermode="x unified",
            dragmode="zoom",
        )


        st.plotly_chart(
            fig_yoy,
            use_container_width=True
        )


# -------------------------------
# Yearly Distribution
# -------------------------------

with chart_right:

    with st.container(border=True):

        df_chart_view["Year"] = (
            df_chart_view["Period"]
            .dt.year
            .astype(str)
        )


        fig_year = px.box(
            df_chart_view,
            x="Year",
            y="Import_Value",
            title="Yearly Distribution",
            labels={
                "Year": "Year",
                "Import_Value":
                    "Import Value (USD)"
            },
        )


        fig_year.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=",.3s",
        )


        st.plotly_chart(
            fig_year,
            use_container_width=True
        )

# ============================================================
# Selected Period Highlights
# ============================================================

st.subheader("Selected-Period Highlights")

if not df_chart_view.empty:

    highest_row = df_chart_view.loc[
        df_chart_view["Import_Value"].idxmax()
    ]

    lowest_row = df_chart_view.loc[
        df_chart_view["Import_Value"].idxmin()
    ]

    latest_yoy_series = (
        df_chart_view["YoY_Growth_Pct"]
        .dropna()
    )

    latest_yoy = (
        latest_yoy_series.iloc[-1]
        if not latest_yoy_series.empty
        else None
    )

    highlight_1, highlight_2, highlight_3 = st.columns(3)

    # Highest value
    with highlight_1:
        with st.container(border=True):

            st.metric(
                "Highest Import Value",
                f"${highest_row['Import_Value']:,.0f}"
            )

            st.caption(
                f"Peak month: "
                f"{highest_row['Period'].strftime('%b %Y')}"
            )

    # Lowest value
    with highlight_2:
        with st.container(border=True):

            st.metric(
                "Lowest Import Value",
                f"${lowest_row['Import_Value']:,.0f}"
            )

            st.caption(
                f"Lowest month: "
                f"{lowest_row['Period'].strftime('%b %Y')}"
            )

    # Latest YoY
    with highlight_3:
        with st.container(border=True):

            st.metric(
                "Latest YoY Change",
                (
                    f"{latest_yoy:.1f}%"
                    if latest_yoy is not None
                    else "N/A"
                )
            )

            st.caption(
                "Latest 12-month comparison"
            )