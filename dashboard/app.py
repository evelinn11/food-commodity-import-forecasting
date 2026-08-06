import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Food Ingredient Commodities Import Value Prediction Dashboard",
    page_icon="📝",
    layout="wide"
)

# Sidebar Navigation & Inputs
# Defining pages for navigation
overview_page = st.Page("overview-page.py", title="Overview", icon="📕")
forecasting_page = st.Page("forecasting-page.py", title="Forecasting", icon="📊")
evaluation_page = st.Page("evaluation-page.py", title="Evaluation", icon="📈")

# Set up navigation
pg = st.navigation([overview_page, forecasting_page, evaluation_page])
pg.run()