import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Food Ingredient Commodities Import Value Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar Navigation
summary_page = st.Page(
    "summary-page.py",
    title="Summary",
    icon="📕"
)

forecasting_page = st.Page(
    "forecasting-page.py",
    title="Forecasting",
    icon="📈"
)

evaluation_page = st.Page(
    "evaluation-page.py",
    title="Evaluation",
    icon="📋"
)

# Set up navigation
pg = st.navigation([
    summary_page,
    forecasting_page,
    evaluation_page
])

pg.run()