import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Title and Description
st.title("📕 Food Raw Materials Commodities Import Value Prediction Dashboard")
st.markdown("This dashboard is a predictive analytics tool designed to forecast the import values of essential food raw materials. " \
"Explore historical trends, analyze market anomalies, and view machine learning-powered projections to make informed purchasing decisions.")
st.info("**Data Source:** All visualizations and forecasts are built on official macroeconomic data from Badan Pusat Statistik " \
"Indonesia covering monthly import records (in USD) from January 2014 to May 2026 across five commodity groups.")

# Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset_final_2014_2026.csv")
    df['Period'] = pd.to_datetime(df['Period'])
    df['HS_Code'] = df['HS_Code'].astype(str).str.zfill(2)
    return df

df = load_data()

# Add a selectbox to the sidebar (Removed 'All Commodities')
st.sidebar.title("Filter")
commodity_selectbox = st.sidebar.selectbox(
    'Choose a commodity group to display:',
    ('Dairy & Honey (HS 04)', 'Vegetables (HS 07)', 'Cereals (HS 10)', 'Seeds & Oleaginous Fruits (HS 12)', 'Sugar (HS 17)')
)

# Map UI Selection to Kode_HS
hs_mapping = {
    'Dairy & Honey (HS 04)': '04',
    'Vegetables (HS 07)': '07',
    'Cereals (HS 10)': '10',
    'Seeds & Oleaginous Fruits (HS 12)': '12',
    'Sugar (HS 17)': '17'
}

hs_code = hs_mapping[commodity_selectbox]
df_filtered = df[df['HS_Code'] == hs_code].copy()

# Calculate Cards
mean_val = df_filtered['Import_Value'].mean()
min_val = df_filtered['Import_Value'].min()
max_val = df_filtered['Import_Value'].max()
skew_val = df_filtered['Import_Value'].skew()
kurt_val = df_filtered['Import_Value'].kurtosis()

# MAPE Mapping
mape_mapping = {
    '04': '10.37%',
    '07': '17.88%',
    '10': '29.03%',
    '12': '19.76%',
    '17': '36.18%'
}

st.subheader(f"Data Overview for {commodity_selectbox}")

# First row
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.metric(label="Mean Import Value (USD)", value=f"${mean_val:,.0f}")
with col2:
    with st.container(border=True):
        st.metric(label="Min Import Value (USD)", value=f"${min_val:,.0f}")
with col3:
    with st.container(border=True):
        st.metric(label="Max Import Value (USD)", value=f"${max_val:,.0f}")

# Second row
col4, col5, col6 = st.columns(3)
with col4:
    with st.container(border=True):
        st.metric(label="Skewness", value=f"{skew_val:.2f}")
with col5:
    with st.container(border=True):
        st.metric(label="Kurtosis", value=f"{kurt_val:.2f}")
with col6:
    with st.container(border=True):
        mape_value = mape_mapping[hs_code]
        st.metric(label="Best MAPE (Model Accuracy)", value=mape_value)

st.markdown("---")

# ==========================================
# Exploratory Data Analysis Section
# ==========================================
st.subheader(f"Exploratory Data Analysis for {commodity_selectbox}")

# The Local Date Slider
min_date = df_filtered['Period'].min().date()
max_date = df_filtered['Period'].max().date()

date_selection = st.slider(
    "Filter Chart Timeline:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY" 
)

# Create the temporary dataframe for the charts
start_date = pd.to_datetime(date_selection[0])
end_date = pd.to_datetime(date_selection[1])
df_chart_view = df_filtered[(df_filtered['Period'] >= start_date) & (df_filtered['Period'] <= end_date)].copy()

with st.container(border=True):
        fig_line = px.line(
            df_chart_view, 
            x='Period', 
            y='Import_Value',
            title=f"Import Value Trend",
            labels={'Period': 'Period', 'Import_Value': 'Import Value (USD)'},
        )
        
        # Add a horizontal line for the mean value
        current_mean = df_chart_view['Import_Value'].mean()
        fig_line.add_hline(
            y=current_mean, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text="Average", 
            annotation_position="top left",
            opacity=0.7
        )
        
        # Update layout for better visualization
        fig_line.update_layout(
            title_x=0.425,
            yaxis_tickprefix='$',
            yaxis_tickformat=',.3s',
            hovermode='x unified',
            dragmode='zoom'
        )
        st.plotly_chart(fig_line, use_container_width=True)

# Calculate YoY Growth on the chart view
df_chart_view['YoY_Growth_Pct'] = df_chart_view['Import_Value'].pct_change(periods=12) * 100

# Render the Charts side-by-side
col7, col8 = st.columns(2)

with col7:
    with st.container(border=True):
            # Determine bar colors based on YoY growth values
            colors = ['#d62728' if val < 0 else '#2ca02c' for val in df_chart_view['YoY_Growth_Pct']]
        
            fig_yoy = px.bar(
                df_chart_view,
                x='Period',
                y='YoY_Growth_Pct',
                title=f"Year-over-Year Growth Rate",
                labels={'Period': 'Period', 'YoY_Growth_Pct': 'YoY Growth Rate (%)'},
            )
            
            # Update the bar colors based on positive or negative growth
            fig_yoy.update_traces(marker_color=colors)
            
            # Update layout for better visualization
            fig_yoy.update_layout(
                title_x=0.325,
                yaxis_ticksuffix='%',
                yaxis_tickformat=',.1f',
                hovermode='x unified',
                dragmode='zoom' 
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

with col8:
    with st.container(border=True):
        # Yearly Distribution
        df_chart_view['Year'] = df_chart_view['Period'].dt.year
        fig_year = px.box(
            df_chart_view, 
            x='Year', 
            y='Import_Value',
            color_discrete_sequence=['#1f77b4'],
            title="Yearly Distribution",
            labels={'Year': 'Year', 'Import_Value': 'Import Value (USD)'}
        )
        fig_year.update_layout(
            title_x=0.4,
            yaxis_tickprefix='$',
            yaxis_tickformat=',.3s'
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
st.markdown("---")
st.subheader("More Analysis")

# ==========================================
# Advanced Statistical Analysis Section
# ==========================================

with st.expander("View Advanced Statistical Analysis (Seasonal Decomposition & Lag Correlations)"):
    st.markdown("Break down the data into Trend, Seasonality, and Residuals, and analyze lag correlations to determine ARIMA/SARIMA parameters (p, d, q).")
    
    with st.container(border=True):
        # Prepare the data for statsmodels by setting the Datetime index
        df_ts = df_filtered.sort_values('Period').set_index('Period')
        series_to_decompose = df_ts['Import_Value'].rename('Import Value (USD)')
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(series_to_decompose, model='additive', period=12)
        
        # Plot the decomposition results
        fig_decomp = decomposition.plot()
        fig_decomp.set_size_inches(12, 8)
        fig_decomp.suptitle(f"Seasonal Decomposition of {commodity_selectbox}", fontsize=14, fontweight='bold')
        fig_decomp.tight_layout()
        st.pyplot(fig_decomp)
    
    with st.container(border=True):
        # Plot ACF and PACF
        # Define the differencing order for each commodity group based on prior analysis
        d_orders = {'04': 1, '07': 1, '10': 0, '12': 1, '17': 1}
        
        # Determine the differencing order for the selected commodity
        d = d_orders[hs_code]
        if d == 1:
            data_stationary = df_ts['Import_Value'].diff().dropna()
            title_suffix = "(Differencing d=1)"
        else:
            data_stationary = df_ts['Import_Value'].dropna()
            title_suffix = "(Level d=0)"
    
        # Create a 1x2 grid of subplots using Matplotlib
        fig_acf_pacf, axes = plt.subplots(1, 2, figsize=(16, 4))
        fig_acf_pacf.suptitle(f"Autocorrelation (ACF) & Partial Autocorrelation (PACF)", fontsize=14, fontweight='bold')
        
        # Generate the plots inside the specific axesaxes[0].set_title(f'ACF - HS {hs_code} {title_suffix}', fontweight='bold')
        plot_acf(data_stationary, lags=36, ax=axes[0], color='#2c7bb6')
        axes[0].set_title(f'ACF - HS {hs_code} {title_suffix}', fontweight='bold')
        axes[0].set_xlabel('Lags')
        axes[0].set_ylabel('Correlation')

        # Generate PACF Plot (determines 'p' and 'P')
        plot_pacf(data_stationary, lags=36, ax=axes[1], color='#d7191c', method='ywm')
        axes[1].set_title(f'PACF - HS {hs_code} {title_suffix}', fontweight='bold')
        axes[1].set_xlabel('Lags')
        axes[1].set_ylabel('Partial Correlation')

        # Add a clean grid for readability
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        fig_acf_pacf.tight_layout()
        st.pyplot(fig_acf_pacf)
    
with st.expander("View Raw Data Table"):
    st.dataframe(df_filtered.sort_values('Period').reset_index(drop=True), use_container_width=True)