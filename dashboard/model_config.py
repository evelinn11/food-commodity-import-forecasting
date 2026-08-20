# ============================================================
# Champion Model Metadata
# Source: Final Model phase of the forecasting experiment
# ============================================================

CHAMPION_MODELS = {
    '04': {
        'commodity': 'Dairy & Honey',
        'model_type': 'XGBoost',
        'mape': 10.37,
        'is_logged': True,
        'features': ['Lag_1', 'Lag_2', 'Month', 'Lag_12', 'Lag_3'],
        'params': {
            'learning_rate': 0.01,
            'max_depth': 3,
            'n_estimators': 200
        }
    },

    '07': {
        'commodity': 'Vegetables',
        'model_type': 'SARIMA',
        'mape': 17.88,
        'is_logged': True,
        'order': (1, 1, 1),
        'seasonal_order': (1, 0, 1, 12)
    },

    '10': {
        'commodity': 'Cereals',
        'model_type': 'XGBoost',
        'mape': 29.03,
        'is_logged': True,
        'features': ['Lag_1', 'Lag_2', 'Lag_12', 'Month'],
        'params': {
            'learning_rate': 0.05,
            'max_depth': 3,
            'n_estimators': 50
        }
    },

    '12': {
        'commodity': 'Seeds & Oleaginous Fruits',
        'model_type': 'XGBoost',
        'mape': 19.76,
        'is_logged': True,
        'features': ['Lag_1', 'Month', 'Lag_2', 'Lag_12', 'Lag_6'],
        'params': {
            'learning_rate': 0.01,
            'max_depth': 5,
            'n_estimators': 50
        }
    },

    '17': {
        'commodity': 'Sugar',
        'model_type': 'XGBoost',
        'mape': 36.18,
        'is_logged': True,
        'features': ['Lag_1', 'Month', 'Lag_3', 'Lag_2', 'Lag_12'],
        'params': {
            'learning_rate': 0.01,
            'max_depth': 3,
            'n_estimators': 100
        }
    }
}