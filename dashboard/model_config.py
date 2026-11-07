"""Locked model configurations and experiment results.

Configurations were selected using two expanding-window validation folds.
Do not change these parameters based on hold-out test performance.
"""

COMMODITY_INFO = {
    "04": "Dairy & Honey",
    "07": "Vegetables",
    "10": "Cereals",
    "12": "Seeds & Oleaginous Fruits",
    "17": "Sugar",
}

TRAIN_START = "2014-01-01"
DEVELOPMENT_END = "2025-05-01"
TEST_START = "2025-06-01"
TEST_END = "2026-05-01"
FORECAST_START = "2026-06-01"
FORECAST_END = "2027-05-01"
FORECAST_HORIZON = 12
SEASONAL_PERIOD = 12
RANDOM_STATE = 42

SARIMA_CONFIG = {
    "04": {"order": (2, 0, 0), "seasonal_order": (1, 1, 1, 12)},
    "07": {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)},
    "10": {"order": (2, 0, 0), "seasonal_order": (1, 1, 1, 12)},
    "12": {"order": (1, 1, 1), "seasonal_order": (0, 0, 1, 12)},
    "17": {"order": (2, 0, 1), "seasonal_order": (0, 0, 0, 12)},
}

XGB_FEATURES = [
    "lag_1", "lag_12", "rolling_mean_3", "rolling_std_3", "month"
]

XGB_CONFIG = {
    "04": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03,
           "subsample": 1.0, "colsample_bytree": 0.8},
    "07": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10,
           "subsample": 0.8, "colsample_bytree": 0.8},
    "10": {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03,
           "subsample": 0.8, "colsample_bytree": 1.0},
    "12": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03,
           "subsample": 0.8, "colsample_bytree": 0.8},
    "17": {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.10,
           "subsample": 0.8, "colsample_bytree": 0.8},
}

# Mean MAPE across validation folds, in percent.
VALIDATION_MAPE = {
    "04": {"SARIMA": 18.868998, "XGBoost": 20.788036},
    "07": {"SARIMA": 17.888259, "XGBoost": 21.548667},
    "10": {"SARIMA": 40.905967, "XGBoost": 61.471861},
    "12": {"SARIMA": 22.686091, "XGBoost": 19.081374},
    "17": {"SARIMA": 27.860031, "XGBoost": 27.364783},
}

VALIDATION_FOLD_MAPE = {
    "04": {"SARIMA": [17.280341, 20.457654], "XGBoost": [31.107347, 10.468725]},
    "07": {"SARIMA": [25.002368, 10.774151], "XGBoost": [26.747639, 16.349694]},
    "10": {"SARIMA": [44.214046, 37.597888], "XGBoost": [45.299402, 77.644321]},
    "12": {"SARIMA": [25.882648, 19.489535], "XGBoost": [17.184531, 20.978216]},
    "17": {"SARIMA": [14.287587, 41.432475], "XGBoost": [27.492967, 27.236599]},
}
