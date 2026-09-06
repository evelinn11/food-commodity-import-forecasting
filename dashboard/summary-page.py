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

st.title("📕 Food Raw Material Import Value Dashboard")

st.markdown(
    """
    An interactive forecasting dashboard designed to help food
    business owners understand historical import value movements
    and anticipate future movements in the aggregate import value
    of selected food raw material commodity groups in Indonesia.
    """
)

st.info(
    """
    The dashboard uses official monthly import value data from
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

    with st.container(border=True, height=470):

        st.subheader("Business Problem")

        st.markdown(
            """
            Food businesses that rely on imported raw materials operate
            within changing international trade and supply conditions.

            Changes in import value can reflect movements in import quantity,
            prices, exchange rates, or a combination of these factors.
            Therefore, import-value movements can provide information about
            changes in the external import environment, but they do not
            directly represent the raw-material price or production cost
            faced by an individual business.
            """
        )

        st.markdown(
            """
            For firms that use imported inputs, changes in import conditions
            may still be relevant to procurement and business performance.
            Previous studies in Indonesia show that imported inputs are
            associated with firm productivity and export performance, while
            raw-material import intensity has also been linked to the
            price-cost margin of the food industry.
            """
        )


# -------------------------------
# Project Objective
# -------------------------------

with right:

    with st.container(border=True, height=470):

        st.subheader("Project Objective")

        st.markdown(
            """
            This project forecasts the monthly import value of selected
            food raw material commodity groups in Indonesia.

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

    with st.container(border=True, height=210):

        st.markdown("#### 📦 Procurement Planning")

        st.write(
            """
            Use expected import-value movements as an additional reference
            when preparing procurement plans, alongside actual supplier
            prices, inventory needs, and other business-specific information.
            """
        )

with benefit_3:

    with st.container(border=True, height=210):

        st.markdown("#### 🏷️ Pricing Consideration")

        st.write(
            """
            Use import-value movements as contextual market information
            when reviewing pricing or product strategies, together with
            actual input prices and business-specific cost information. 
            \n
            """
        )


with benefit_4:

    with st.container(border=True, height=210):

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
    Important: Import value is not the same as market price,
    unit import price, or a business's actual raw-material cost.
    Changes in import value can be influenced by import quantity,
    prices, exchange rates, and other market factors. Forecasts
    should therefore be interpreted as indicators of aggregate
    import-value movements, not as direct forecasts of input prices
    or production costs.
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
    selected food raw material commodity groups. The data was collected from official BPS import statistics and
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
    The original BPS dataset requires reshaping before they could be used
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
    Explore the historical behavior of each commodity from
    January 2014 to May 2026. Select a commodity group below
    to view its import value trend, monthly year-over-year growth,
    and yearly distribution.
    """
)


# ============================================================
# EDA INTERPRETATIONS
# ============================================================

EDA_INTERPRETATIONS = {

    "04": {
        "trend": """
        Import values declined during the early observation period
        before gradually recovering from 2017 onward. A noticeable
        increase occurred during 2021–2022, followed by a decline
        in 2023. Import values afterward remained relatively higher
        than in most of the earlier years.
        """,

        "distribution": """
        The yearly distributions show that monthly import values
        shifted toward higher levels around 2021–2022. Some years
        also display wider distributions, indicating greater
        month-to-month variation in import values.
        """
    },

    "07": {
        "trend": """
        Vegetable import values show a generally increasing
        long-term pattern, although several temporary declines
        occur throughout the observation period. Import values
        reached relatively high levels in the later years before
        declining again toward the end of the period.
        """,

        "distribution": """
        The yearly distributions generally shift toward higher
        monthly import values in the later years compared with
        the beginning of the observation period. However, the
        distributions still overlap considerably, indicating
        continued month-to-month variation.
        """
    },

    "10": {
        "trend": """
        Cereals have the highest import-value scale among the five
        commodity groups and show substantial fluctuations over time.
        A strong upward movement is visible during the later years,
        particularly from 2021 onward, followed by a noticeable
        decline toward the end of the observation period.
        """,

        "distribution": """
        Cereals show a wide distribution of monthly import values
        and several unusually high observations. This indicates
        greater variability compared with the other commodity groups
        and is consistent with the strongly right-skewed distribution
        observed in the data.
        """
    },

    "12": {
        "trend": """
        Import values remained comparatively stable during much
        of the earlier period before increasing substantially
        during 2021–2022. After reaching higher levels during this
        period, import values show a declining tendency in the
        following years.
        """,

        "distribution": """
        The yearly distributions shift upward during 2021–2022,
        reflecting generally higher monthly import values during
        those years. The distributions then move toward lower
        levels in the following period.
        """
    },

    "17": {
        "trend": """
        Sugar import values show pronounced fluctuations throughout
        the observation period. Several periods of substantial
        increase and decline are visible, with import values generally
        reaching higher levels in the later years than at the
        beginning of the series.
        """,

        "distribution": """
        The yearly distributions indicate considerable variation
        in monthly sugar import values. Later years generally show
        higher monthly import levels than the early observation
        period, although substantial within-year variation remains.
        """
    }
}


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


# Filter selected commodity and sort chronologically
df_filtered = (
    df[
        df["HS_Code"] == hs_code
    ]
    .copy()
    .sort_values("Period")
    .reset_index(drop=True)
)


# ============================================================
# Monthly Year-over-Year Growth
# ============================================================

# Each month's import value is compared with
# the same month 12 months earlier.
df_filtered["YoY_Growth_Pct"] = (
    df_filtered["Import_Value"]
    .pct_change(periods=12)
    * 100
)


# Create year column for yearly distribution chart
df_filtered["Year"] = (
    df_filtered["Period"]
    .dt.year
    .astype(str)
)


# ============================================================
# Historical Highlights
# ============================================================

st.subheader("Historical Highlights")


# Highest monthly import value
highest_row = df_filtered.loc[
    df_filtered["Import_Value"].idxmax()
]


# Lowest monthly import value
lowest_row = df_filtered.loc[
    df_filtered["Import_Value"].idxmin()
]


# Latest available monthly YoY
latest_yoy_series = (
    df_filtered[
        "YoY_Growth_Pct"
    ]
    .dropna()
)


latest_yoy = (
    latest_yoy_series.iloc[-1]
    if not latest_yoy_series.empty
    else None
)


latest_yoy_period = (
    df_filtered.loc[
        latest_yoy_series.index[-1],
        "Period"
    ]
    if not latest_yoy_series.empty
    else None
)


highlight_1, highlight_2, highlight_3 = st.columns(3)


# -------------------------------
# Highest Import Value
# -------------------------------

with highlight_1:

    with st.container(border=True):

        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">'
            f'Highest Monthly Import Value'
            f'</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:4px;">'
            f'${highest_row["Import_Value"]:,.0f}'
            f'</div>'
            f'<div style="font-size:14px; color:gray; margin-bottom:12px;">'
            f'Peak month: {highest_row["Period"].strftime("%b %Y")}'
            f'</div>',
            unsafe_allow_html=True
        )


# -------------------------------
# Lowest Import Value
# -------------------------------

with highlight_2:

    with st.container(border=True):

        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">'
            f'Lowest Monthly Import Value'
            f'</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:4px;">'
            f'${lowest_row["Import_Value"]:,.0f}'
            f'</div>'
            f'<div style="font-size:14px; color:gray; margin-bottom:12px;">'
            f'Lowest month: {lowest_row["Period"].strftime("%b %Y")}'
            f'</div>',
            unsafe_allow_html=True
        )


# -------------------------------
# Latest Monthly YoY
# -------------------------------

with highlight_3:

    with st.container(border=True):

        latest_yoy_text = (
            f"{latest_yoy:.1f}%"
            if latest_yoy is not None
            else "N/A"
        )

        latest_period_text = (
            latest_yoy_period.strftime("%b %Y")
            if latest_yoy_period is not None
            else "N/A"
        )

        st.markdown(
            f'<div style="font-size:16px; margin-bottom:4px;">'
            f'Latest YoY Change'
            f'</div>'
            f'<div style="font-size:24px; font-weight:600; margin-bottom:4px;">'
            f'{latest_yoy_text}'
            f'</div>'
            f'<div style="font-size:14px; color:gray; margin-bottom:12px;">'
            f'12-month comparison: {latest_period_text}'
            f'</div>',
            unsafe_allow_html=True
        )


# ============================================================
# IMPORT VALUE TREND
# ============================================================

with st.container(border=True):

    fig_line = px.line(
        df_filtered,
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


    # Historical average across the entire period
    historical_mean = (
        df_filtered[
            "Import_Value"
        ]
        .mean()
    )


    fig_line.add_hline(
        y=historical_mean,
        line_dash="dash",
        line_color="gray",
        annotation_text="Historical average",
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


    # -------------------------------
    # Trend Interpretation
    # -------------------------------

    with st.expander("View Interpretation"):

        st.markdown(
            EDA_INTERPRETATIONS[
                hs_code
            ]["trend"]
        )


# ============================================================
# YoY & Distribution Charts
# ============================================================

chart_left, chart_right = st.columns(2)


# ============================================================
# MONTHLY YEAR-OVER-YEAR GROWTH
# ============================================================

with chart_left:

    with st.container(border=True):

        # First 12 observations do not have
        # a previous-year comparison.
        yoy_chart_data = (
            df_filtered
            .dropna(
                subset=[
                    "YoY_Growth_Pct"
                ]
            )
            .copy()
        )


        # Positive growth = green
        # Negative growth = red
        yoy_colors = [
            "#2ca02c"
            if value >= 0
            else "#d62728"

            for value
            in yoy_chart_data[
                "YoY_Growth_Pct"
            ]
        ]


        fig_yoy = px.bar(
            yoy_chart_data,
            x="Period",
            y="YoY_Growth_Pct",
            title="Monthly Year-over-Year Growth Rate",
            labels={
                "Period": "Period",
                "YoY_Growth_Pct":
                    "YoY Growth Rate (%)"
            },
        )


        fig_yoy.update_traces(
            marker_color=yoy_colors
        )


        # Zero reference line
        fig_yoy.add_hline(
            y=0,
            line_width=1,
            line_color="gray"
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
        # Monthly YoY Interpretation
        # -------------------------------

        with st.expander("View Interpretation"):

            if not yoy_chart_data.empty:

                # Strongest increase
                highest_yoy_row = yoy_chart_data.loc[
                    yoy_chart_data[
                        "YoY_Growth_Pct"
                    ].idxmax()
                ]


                # Strongest decline
                lowest_yoy_row = yoy_chart_data.loc[
                    yoy_chart_data[
                        "YoY_Growth_Pct"
                    ].idxmin()
                ]


                # Latest comparison
                latest_yoy_row = (
                    yoy_chart_data
                    .iloc[-1]
                )


                highest_yoy = (
                    highest_yoy_row[
                        "YoY_Growth_Pct"
                    ]
                )

                lowest_yoy = (
                    lowest_yoy_row[
                        "YoY_Growth_Pct"
                    ]
                )

                latest_yoy_value = (
                    latest_yoy_row[
                        "YoY_Growth_Pct"
                    ]
                )


                highest_period = (
                    highest_yoy_row[
                        "Period"
                    ]
                    .strftime("%b %Y")
                )

                lowest_period = (
                    lowest_yoy_row[
                        "Period"
                    ]
                    .strftime("%b %Y")
                )

                latest_period = (
                    latest_yoy_row[
                        "Period"
                    ]
                    .strftime("%b %Y")
                )


                # Describe latest YoY direction
                if latest_yoy_value > 0:

                    latest_interpretation = (
                        "higher than the same month "
                        "one year earlier"
                    )

                elif latest_yoy_value < 0:

                    latest_interpretation = (
                        "lower than the same month "
                        "one year earlier"
                    )

                else:

                    latest_interpretation = (
                        "approximately unchanged from "
                        "the same month one year earlier"
                    )


                st.markdown(
                    f"""
                    Monthly YoY growth compares each month's import
                    value with the same month one year earlier.

                    For {commodity_selectbox}, the strongest
                    YoY increase occurred in {highest_period}
                    at {highest_yoy:.1f}%, while the largest
                    YoY decline occurred in {lowest_period}
                    at {lowest_yoy:.1f}%.

                    The latest observation in {latest_period}
                    recorded a YoY change of
                    {latest_yoy_value:.1f}%, meaning the import
                    value was {latest_interpretation}.
                    """
                )

            else:

                st.write(
                    """
                    At least 12 months of historical data are required
                    to calculate a year-over-year comparison.
                    """
                )


# ============================================================
# YEARLY DISTRIBUTION
# ============================================================

with chart_right:

    with st.container(border=True):

        fig_year = px.box(
            df_filtered,
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


        # -------------------------------
        # Distribution Interpretation
        # -------------------------------

        with st.expander("View Interpretation"):

            st.markdown(
                EDA_INTERPRETATIONS[
                    hs_code
                ]["distribution"]
            )

            st.caption(
                """
                The line inside each box represents the median
                monthly import value. A taller box indicates
                greater variation among monthly values within
                that year. The 2026 distribution should be
                interpreted carefully because it only contains
                data from January to May.
                """
            )