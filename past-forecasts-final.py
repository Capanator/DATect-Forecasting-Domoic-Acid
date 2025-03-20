import os

import numpy as np
import pandas as pd

from joblib import Memory

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor, GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (LinearRegression, LogisticRegression, RidgeCV)
from sklearn.metrics import (accuracy_score, mean_absolute_error,r2_score)
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR

import xgboost as xgb

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


# ---------------------------------------------------------
# Global Flags and Caching Settings
# ---------------------------------------------------------
ENABLE_SEASON_SPECIFIC_TRAINER = False
ENABLE_LINEAR_LOGISTIC = True
ENABLE_RANDOM_ANCHOR_FORECASTS = False
ENABLE_GRIDSEARCHCV = True

cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)

print("[INFO] Caching directory is ready at:", cache_dir)


# ---------------------------------------------------------
# 1) Preprocessing & Feature Engineering Helpers
# ---------------------------------------------------------
def create_numeric_pipeline():
    """
    Returns a pipeline for numeric preprocessing:
      - Median imputation
      - MinMax scaling
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])


def create_numeric_transformer(df: pd.DataFrame, drop_cols: list):
    """
    Builds and returns a ColumnTransformer pipeline for numeric features.
    Non-numeric columns are passed through without transformation.
    """
    X = df.drop(columns=drop_cols, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    pipeline = create_numeric_pipeline()
    transformer = ColumnTransformer(
        transformers=[('num', pipeline, numeric_cols)],
        remainder='passthrough'
    )
    return transformer, X


# ---------------------------------------------------------
# 2) Data Loading & Feature Engineering (cached)
# ---------------------------------------------------------
@memory.cache
def load_and_prepare_data(file_path: str, season: str = None) -> pd.DataFrame:
    """
    Loads data from a Parquet file, applies optional seasonal filtering,
    adds time-based features, and creates lag features & categories.
    """
    print(f"[INFO] Loading data from {file_path}")
    data = pd.read_parquet(file_path, engine='pyarrow')
    
    # Convert and sort by date
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)
    
    # Optional seasonal subset
    if season and ENABLE_SEASON_SPECIFIC_TRAINER:
        print(f"[INFO] Applying seasonal filter for '{season}'")
        if season == 'spring':
            data = data[data['Date'].dt.month.isin([3, 4, 5, 6, 7])]
        elif season == 'fall':
            data = data[data['Date'].dt.month.isin([9, 10, 11, 12])]
    
    # Cyclical day-of-year transformations
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # Create lag features
    for lag in [1, 2, 3]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)
    
    # Create DA_Level categories
    def categorize_da_levels(x):
        if x <= 5:
            return 0
        elif x <= 20:
            return 1
        elif x <= 40:
            return 2
        return 3

    data['DA_Category'] = data['DA_Levels'].apply(categorize_da_levels)
    return data


# ---------------------------------------------------------
# 3) Common Model Training Functions
# ---------------------------------------------------------
def train_model(model, X_train, y_train, X_test, model_type='regression',
                cv=None, param_grid=None):
    """
    Trains a given model (regression or classification). If ENABLE_GRIDSEARCHCV
    and param_grid are provided, will run GridSearchCV to find best parameters.
    Returns the trained (or best) model and predictions for X_test.
    """
    if param_grid is not None and ENABLE_GRIDSEARCHCV:
        print("[INFO] Using GridSearchCV for parameter tuning.")
        scoring = 'r2' if model_type == 'regression' else 'accuracy'
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("[INFO] GridSearchCV complete. Best params:", grid_search.best_params_)
    else:
        model.fit(X_train, y_train)
        best_model = model
        if param_grid is not None:
            print("[INFO] GridSearchCV is disabled; trained with default params.")

    predictions = best_model.predict(X_test)
    return best_model, predictions


# ---------------------------------------------------------
# 4) Core Train & Predict Functions (Random Forest, etc.)
# ---------------------------------------------------------
def train_and_predict(data: pd.DataFrame):
    """
    Splits the data into train/test sets, trains both a RandomForest regressor
    (for DA_Levels) and a RandomForest classifier (for DA_Category), and returns
    predictions and performance metrics.
    """
    print("[INFO] Training RandomForest models (regression & classification)...")
    tscv = TimeSeriesSplit(n_splits=5)

    # ------------------ Regression Setup ------------------
    data_reg = data.drop(['DA_Category'], axis=1)
    train_reg, test_reg = train_test_split(data_reg, test_size=0.2, random_state=42)
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']

    transformer_reg, X_train_reg = create_numeric_transformer(train_reg, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        test_reg.drop(columns=drop_cols_reg, errors='ignore')
    )

    # RandomForest Regressor
    param_grid_reg = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_reg, y_pred_reg = train_model(
        RandomForestRegressor(random_state=42),
        X_train_reg_processed,
        train_reg['DA_Levels'],
        X_test_reg_processed,
        model_type='regression',
        cv=tscv,
        param_grid=param_grid_reg
    )

    test_reg = test_reg.copy()
    test_reg['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(test_reg['DA_Levels'], y_pred_reg)
    overall_mae_reg = mean_absolute_error(test_reg['DA_Levels'], y_pred_reg)

    # Per-site performance for regression
    site_stats_reg = (
        test_reg.groupby('Site')[['DA_Levels', 'Predicted_DA_Levels']]
        .apply(lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'mae': mean_absolute_error(x['DA_Levels'], x['Predicted_DA_Levels'])
        }))
    )

    # ------------------ Classification Setup ------------------
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_cls, test_cls = train_test_split(data_cls, test_size=0.2, random_state=42)
    drop_cols_cls = ['DA_Category', 'Date', 'Site']

    transformer_cls, X_train_cls = create_numeric_transformer(train_cls, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        test_cls.drop(columns=drop_cols_cls, errors='ignore')
    )

    # RandomForest Classifier
    param_grid_cls = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_cls, y_pred_cls = train_model(
        RandomForestClassifier(random_state=42),
        X_train_cls_processed,
        train_cls['DA_Category'],
        X_test_cls_processed,
        model_type='classification',
        cv=tscv,
        param_grid=param_grid_cls
    )

    test_cls = test_cls.copy()
    test_cls['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(test_cls['DA_Category'], y_pred_cls)

    # Per-site performance for classification
    site_stats_cls = (
        test_cls.groupby('Site')[['DA_Category', 'Predicted_DA_Category']]
        .apply(lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category']))
    )

    print("[INFO] RandomForest training complete.")
    return {
        "DA_Level": {
            "test_df": test_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_mae": overall_mae_reg
        },
        "DA_Category": {
            "test_df": test_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls
        }
    }


# ---------------------------------------------------------
# 5) Linear & Logistic Regression Train & Predict
# ---------------------------------------------------------
def train_and_predict_lr(data: pd.DataFrame):
    """
    Trains LinearRegression for DA_Levels and LogisticRegression for DA_Category.
    Returns a dictionary of predictions and metrics.
    """
    print("[INFO] Training Linear/Logistic Regression models...")

    # ------------------ Linear Regression ------------------
    data_reg = data.drop(['DA_Category'], axis=1)
    train_reg, test_reg = train_test_split(data_reg, test_size=0.25, random_state=42)
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']

    transformer_reg, X_train_reg = create_numeric_transformer(train_reg, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        test_reg.drop(columns=drop_cols_reg, errors='ignore')
    )

    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train_reg_processed, train_reg['DA_Levels'])
    y_pred_reg = lin_regressor.predict(X_test_reg_processed)

    test_reg = test_reg.copy()
    test_reg['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(test_reg['DA_Levels'], y_pred_reg)
    overall_mae_reg = mean_absolute_error(test_reg['DA_Levels'], y_pred_reg)

    site_stats_reg = (
        test_reg.groupby('Site')[['DA_Levels', 'Predicted_DA_Levels']]
        .apply(lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'mae': mean_absolute_error(x['DA_Levels'], x['Predicted_DA_Levels'])
        }))
    )

    # ------------------ Logistic Regression ------------------
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_cls, test_cls = train_test_split(data_cls, test_size=0.25, random_state=42)
    drop_cols_cls = ['DA_Category', 'Date', 'Site']

    transformer_cls, X_train_cls = create_numeric_transformer(train_cls, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        test_cls.drop(columns=drop_cols_cls, errors='ignore')
    )

    log_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_classifier.fit(X_train_cls_processed, train_cls['DA_Category'])
    y_pred_cls = log_classifier.predict(X_test_cls_processed)

    test_cls = test_cls.copy()
    test_cls['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(test_cls['DA_Category'], y_pred_cls)

    site_stats_cls = (
        test_cls.groupby('Site')[['DA_Category', 'Predicted_DA_Category']]
        .apply(lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category']))
    )

    print("[INFO] Linear/Logistic Regression training complete.")
    return {
        "DA_Level": {
            "test_df": test_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_mae": overall_mae_reg
        },
        "DA_Category": {
            "test_df": test_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls
        }
    }


# ---------------------------------------------------------
# 6) Forecast Functions (Random Anchor) for Various Models
# ---------------------------------------------------------
def forecast_next_date(df: pd.DataFrame, anchor_date, site):
    """
    Forecasts the next date's DA_Levels and DA_Category (if it exists) for
    a given site after anchor_date using RandomForest models.
    """
    print(f"[INFO] Forecasting next date for site '{site}' after '{anchor_date}' - ML")
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None

    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_train.empty or df_test.empty:
        return None

    # Regression (RandomForest)
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(df_train, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        df_test.drop(columns=drop_cols_reg, errors='ignore')
    )

    reg_model = RandomForestRegressor(random_state=42, n_estimators=100)
    reg_model.fit(X_train_reg_processed, df_train['DA_Levels'])
    y_pred_reg = reg_model.predict(X_test_reg_processed)

    # Classification (RandomForest)
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(df_train, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        df_test.drop(columns=drop_cols_cls, errors='ignore')
    )

    cls_model = RandomForestClassifier(random_state=42, n_estimators=100)
    cls_model.fit(X_train_cls_processed, df_train['DA_Category'])
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]),
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0])
    }


def forecast_next_date_lr(df: pd.DataFrame, anchor_date, site):
    """
    Forecasts the next date's DA_Levels and DA_Category for a given site
    after anchor_date using Linear & Logistic Regression.
    """
    print(f"[INFO] Forecasting next date for site '{site}' after '{anchor_date}' - LR")
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None

    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_train.empty or df_test.empty:
        return None

    # LinearRegression for DA_Levels
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(df_train, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        df_test.drop(columns=drop_cols_reg, errors='ignore')
    )

    lin_model = LinearRegression()
    lin_model.fit(X_train_reg_processed, df_train['DA_Levels'])
    y_pred_reg = lin_model.predict(X_test_reg_processed)

    # LogisticRegression for DA_Category
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(df_train, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        df_test.drop(columns=drop_cols_cls, errors='ignore')
    )

    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_model.fit(X_train_cls_processed, df_train['DA_Category'])
    y_pred_cls = log_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]),
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0])
    }


# ---------------------------------------------------------
# 7) Stacking Ensemble Train & Predict + Forecast
# ---------------------------------------------------------
def train_stacking_ensemble(X_train, y_train, X_test, model_type='regression',
                            cv=None, param_grid=None):
    """
    Creates and trains a Stacking ensemble for regression or classification,
    optionally with GridSearchCV if ENABLE_GRIDSEARCHCV is True.
    """
    print(f"[INFO] Training Stacking Ensemble ({model_type})...")

    if model_type == 'regression':
        base_models = [
            ('rf', RandomForestRegressor(random_state=42, n_estimators=200)),
            ('gbr', GradientBoostingRegressor(random_state=42, n_estimators=100)),
            ('xgb', xgb.XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ]
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5, n_jobs=-1
        )

        if param_grid and ENABLE_GRIDSEARCHCV:
            print("[INFO] GridSearchCV for stacking regression ensemble.")
            grid_search = GridSearchCV(
                estimator=stacking_model,
                param_grid=param_grid,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print("[INFO] Best params:", grid_search.best_params_)
        else:
            stacking_model.fit(X_train, y_train)
            best_model = stacking_model

        predictions = best_model.predict(X_test)
        return best_model, predictions

    # Classification case
    base_models = [
        ('rf', RandomForestClassifier(random_state=42, n_estimators=200)),
        ('gbc', GradientBoostingClassifier(random_state=42, n_estimators=100)),
        ('xgb', xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, n_jobs=-1
    )

    if param_grid and ENABLE_GRIDSEARCHCV:
        print("[INFO] GridSearchCV for stacking classification ensemble.")
        grid_search = GridSearchCV(
            estimator=stacking_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("[INFO] Best params:", grid_search.best_params_)
    else:
        stacking_model.fit(X_train, y_train)
        best_model = stacking_model

    predictions = best_model.predict(X_test)
    return best_model, predictions


def train_and_predict_stacking(data: pd.DataFrame):
    """
    Splits data, trains stacking ensemble for both regression & classification,
    and returns predictions and performance metrics.
    """
    print("[INFO] Training Stacking Ensemble (regression & classification)...")
    tscv = TimeSeriesSplit(n_splits=5)

    # ------------------ Regression Setup ------------------
    data_reg = data.drop(['DA_Category'], axis=1)
    train_reg, test_reg = train_test_split(data_reg, test_size=0.2, random_state=42)
    drop_cols_reg = ['DA_Levels', 'Date', 'Site']

    transformer_reg, X_train_reg = create_numeric_transformer(train_reg, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        test_reg.drop(columns=drop_cols_reg, errors='ignore')
    )

    # Train Stacking Regressor
    best_reg, y_pred_reg = train_stacking_ensemble(
        X_train_reg_processed,
        train_reg['DA_Levels'],
        X_test_reg_processed,
        model_type='regression',
        cv=tscv,
        param_grid=None
    )
    test_reg = test_reg.copy()
    test_reg['Predicted_DA_Levels'] = y_pred_reg
    overall_r2_reg = r2_score(test_reg['DA_Levels'], y_pred_reg)
    overall_mae_reg = mean_absolute_error(test_reg['DA_Levels'], y_pred_reg)

    site_stats_reg = (
        test_reg.groupby('Site')[['DA_Levels', 'Predicted_DA_Levels']]
        .apply(lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'mae': mean_absolute_error(x['DA_Levels'], x['Predicted_DA_Levels'])
        }))
    )

    # ------------------ Classification Setup ------------------
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_cls, test_cls = train_test_split(data_cls, test_size=0.2, random_state=42)
    drop_cols_cls = ['DA_Category', 'Date', 'Site']

    transformer_cls, X_train_cls = create_numeric_transformer(train_cls, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        test_cls.drop(columns=drop_cols_cls, errors='ignore')
    )

    # Train Stacking Classifier
    best_cls, y_pred_cls = train_stacking_ensemble(
        X_train_cls_processed,
        train_cls['DA_Category'],
        X_test_cls_processed,
        model_type='classification',
        cv=tscv,
        param_grid=None
    )
    test_cls = test_cls.copy()
    test_cls['Predicted_DA_Category'] = y_pred_cls
    overall_accuracy_cls = accuracy_score(test_cls['DA_Category'], y_pred_cls)

    site_stats_cls = (
        test_cls.groupby('Site')[['DA_Category', 'Predicted_DA_Category']]
        .apply(lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category']))
    )

    print("[INFO] Stacking Ensemble training complete.")
    return {
        "DA_Level": {
            "test_df": test_reg,
            "site_stats": site_stats_reg,
            "overall_r2": overall_r2_reg,
            "overall_mae": overall_mae_reg
        },
        "DA_Category": {
            "test_df": test_cls,
            "site_stats": site_stats_cls,
            "overall_accuracy": overall_accuracy_cls
        }
    }


def forecast_next_date_stacking(df: pd.DataFrame, anchor_date, site):
    """
    Forecasts the next date's DA_Levels and DA_Category using a Stacking ensemble.
    """
    print(f"[INFO] Forecasting next date for site '{site}' after '{anchor_date}' - Stacking")
    df_site = df[df['Site'] == site].copy().sort_values('Date')
    df_future = df_site[df_site['Date'] > anchor_date]
    if df_future.empty:
        return None

    next_date = df_future['Date'].iloc[0]
    df_train = df_site[df_site['Date'] <= anchor_date].copy()
    df_test = df_site[df_site['Date'] == next_date].copy()
    if df_train.empty or df_test.empty:
        return None

    # ------------------ Regression Forecast ------------------
    drop_cols_reg = ['DA_Levels', 'DA_Category', 'Date', 'Site']
    transformer_reg, X_train_reg = create_numeric_transformer(df_train, drop_cols_reg)
    X_train_reg_processed = transformer_reg.fit_transform(X_train_reg)
    X_test_reg_processed = transformer_reg.transform(
        df_test.drop(columns=drop_cols_reg, errors='ignore')
    )

    base_models_reg = [
        ('rf', RandomForestRegressor(random_state=42, n_estimators=100)),
        ('gbr', GradientBoostingRegressor(random_state=42, n_estimators=100)),
        ('xgb', xgb.XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1))
    ]
    reg_model = StackingRegressor(
        estimators=base_models_reg,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        cv=3
    )
    reg_model.fit(X_train_reg_processed, df_train['DA_Levels'])
    y_pred_reg = reg_model.predict(X_test_reg_processed)

    # ------------------ Classification Forecast ------------------
    drop_cols_cls = ['DA_Category', 'DA_Levels', 'Date', 'Site']
    transformer_cls, X_train_cls = create_numeric_transformer(df_train, drop_cols_cls)
    X_train_cls_processed = transformer_cls.fit_transform(X_train_cls)
    X_test_cls_processed = transformer_cls.transform(
        df_test.drop(columns=drop_cols_cls, errors='ignore')
    )

    base_models_cls = [
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('gbc', GradientBoostingClassifier(random_state=42, n_estimators=100)),
        ('xgb', xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
    ]
    cls_model = StackingClassifier(
        estimators=base_models_cls,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )
    cls_model.fit(X_train_cls_processed, df_train['DA_Category'])
    y_pred_cls = cls_model.predict(X_test_cls_processed)

    return {
        'AnchorDate': anchor_date,
        'Site': site,
        'NextDate': next_date,
        'Predicted_DA_Levels': float(y_pred_reg[0]),
        'Actual_DA_Levels': float(df_test['DA_Levels'].iloc[0]),
        'Predicted_DA_Category': int(y_pred_cls[0]),
        'Actual_DA_Category': int(df_test['DA_Category'].iloc[0])
    }


# ---------------------------------------------------------
# 8) Random Anchor Forecast Helper
# ---------------------------------------------------------
def get_random_anchor_forecasts(data: pd.DataFrame, forecast_func):
    """
    Draws random anchor dates post-2010 from each site, performs forecasting,
    and returns the results in a DataFrame along with performance metrics.
    """
    print("[INFO] Generating random anchor forecasts...")
    NUM_RANDOM_ANCHORS = 50
    df_after_2010 = data[data['Date'].dt.year >= 2010].copy().sort_values(['Site', 'Date'])
    pairs_after_2010 = df_after_2010[['Site', 'Date']].drop_duplicates()

    # Sample up to NUM_RANDOM_ANCHORS per site
    df_random_anchors = pd.concat([
        group.sample(n=min(NUM_RANDOM_ANCHORS, len(group)), random_state=42)
        for _, group in pairs_after_2010.groupby('Site')
    ]).reset_index(drop=True)

    results_list = []
    for _, row in df_random_anchors.iterrows():
        site_ = row['Site']
        anchor_date_ = row['Date']
        result = forecast_func(data, anchor_date_, site_)
        if result is not None:
            results_list.append(result)

    df_results_anchors = pd.DataFrame(results_list)
    if df_results_anchors.empty:
        return df_results_anchors, {}, None, None

    mae_anchors = mean_absolute_error(
        df_results_anchors['Actual_DA_Levels'],
        df_results_anchors['Predicted_DA_Levels']
    )
    acc_anchors = (
        df_results_anchors['Actual_DA_Category'] ==
        df_results_anchors['Predicted_DA_Category']
    ).mean()

    # Build line plots per site
    figs_random_site = {}
    for site, df_site in df_results_anchors.groupby("Site"):
        df_line = df_site.sort_values("NextDate")
        df_plot_melt = df_line.melt(
            id_vars=["NextDate"],
            value_vars=["Actual_DA_Levels", "Predicted_DA_Levels"],
            var_name="Type", value_name="DA_Level"
        )
        fig = px.line(
            df_plot_melt,
            x="NextDate",
            y="DA_Level",
            color="Type",
            title=f"Random Anchor Forecast for Site {site}"
        )
        fig.update_layout(xaxis_title="Next Date", yaxis_title="DA Level")
        figs_random_site[site] = fig

    return df_results_anchors, figs_random_site, mae_anchors, acc_anchors


# ---------------------------------------------------------
# 9) Prepare Data & Predictions
# ---------------------------------------------------------
print("[INFO] Loading and preparing data...")
file_path = 'final_output_og.parquet'

# Annual data
raw_data_annual = load_and_prepare_data(file_path, season=None)
raw_data_dict = {'annual': raw_data_annual}

# Optional seasonal data
if ENABLE_SEASON_SPECIFIC_TRAINER:
    raw_data_spring = load_and_prepare_data(file_path, season='spring')
    raw_data_fall = load_and_prepare_data(file_path, season='fall')
    raw_data_dict.update({'spring': raw_data_spring, 'fall': raw_data_fall})

# Random Forest / ML predictions
print("[INFO] Computing ML-based predictions...")
predictions_ml = {
    'annual': train_and_predict(raw_data_annual),
    'spring': train_and_predict(raw_data_spring) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
    'fall': train_and_predict(raw_data_fall) if ENABLE_SEASON_SPECIFIC_TRAINER else None
}

random_anchors_ml = None
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    random_anchors_ml = {
        'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date),
        'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
        'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date) if ENABLE_SEASON_SPECIFIC_TRAINER else None
    }

# Stacking Ensemble predictions
print("[INFO] Computing Stacking Ensemble-based predictions...")
predictions_stacking = {
    'annual': train_and_predict_stacking(raw_data_annual),
    'spring': train_and_predict_stacking(raw_data_spring) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
    'fall': train_and_predict_stacking(raw_data_fall) if ENABLE_SEASON_SPECIFIC_TRAINER else None
}

random_anchors_stacking = None
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    random_anchors_stacking = {
        'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date_stacking),
        'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date_stacking) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
        'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date_stacking) if ENABLE_SEASON_SPECIFIC_TRAINER else None
    }

# Linear/Logistic Regression predictions
predictions_lr = None
random_anchors_lr = None

if ENABLE_LINEAR_LOGISTIC:
    print("[INFO] Computing LR-based predictions...")
    predictions_lr = {
        'annual': train_and_predict_lr(raw_data_annual),
        'spring': train_and_predict_lr(raw_data_spring) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
        'fall': train_and_predict_lr(raw_data_fall) if ENABLE_SEASON_SPECIFIC_TRAINER else None
    }
    if ENABLE_RANDOM_ANCHOR_FORECASTS:
        random_anchors_lr = {
            'annual': get_random_anchor_forecasts(raw_data_annual, forecast_next_date_lr),
            'spring': get_random_anchor_forecasts(raw_data_spring, forecast_next_date_lr) if ENABLE_SEASON_SPECIFIC_TRAINER else None,
            'fall': get_random_anchor_forecasts(raw_data_fall, forecast_next_date_lr) if ENABLE_SEASON_SPECIFIC_TRAINER else None
        }
else:
    print("[INFO] Linear/Logistic Regression is disabled.")


# ---------------------------------------------------------
# 10) Dash App Setup
# ---------------------------------------------------------
print("[INFO] Setting up the Dash app layout.")
app = dash.Dash(__name__)

# Season Dropdown (if seasonal trainer is enabled)
season_dropdown = None
if ENABLE_SEASON_SPECIFIC_TRAINER:
    season_dropdown = dcc.Dropdown(
        id='season-dropdown',
        options=[
            {'label': 'Annual', 'value': 'annual'},
            {'label': 'Spring Bloom', 'value': 'spring'},
            {'label': 'Fall Bloom', 'value': 'fall'}
        ],
        value='annual',
        style={'width': '30%'}
    )

dummy_store = dcc.Store(id='dummy-store', data='initial')

analysis_layout = html.Div([
    html.H3("Overall Analysis (Time-Series)"),
    dcc.Dropdown(
        id='forecast-type-dropdown',
        options=[
            {'label': 'DA Levels', 'value': 'DA_Level'},
            {'label': 'DA Category', 'value': 'DA_Category'}
        ],
        value='DA_Level',
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        placeholder='Select Site',
        style={'width': '50%'}
    ),
    dcc.Graph(id='analysis-graph')
])

# Random Anchors layout
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    random_anchors_layout = html.Div([
        html.H3("Random Anchor Dates Forecast -> Next Date by Site"),
        html.Div(id='random-anchor-container')
    ])
else:
    random_anchors_layout = html.Div([
        html.H3("Random Anchor Forecasts are disabled.")
    ])

# Tabs
tabs_children = [dcc.Tab(label='Analysis', children=[analysis_layout])]
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    tabs_children.append(
        dcc.Tab(label='Random Anchor Forecast', children=[random_anchors_layout])
    )

layout_children = [
    html.H1("Domoic Acid Forecast Dashboard"),
    html.Div([
        season_dropdown if season_dropdown else html.Div(),  # Show only if enabled
        html.Label("Select Forecast Method"),
        dcc.Dropdown(
            id='forecast-method-dropdown',
            options=(
                [
                    {'label': 'Machine Learning Forecasts', 'value': 'ml'},
                    {'label': 'Linear/Logistic Regression Forecasts', 'value': 'lr'},
                    {'label': 'Stacking Ensemble Forecasts', 'value': 'stacking'}
                ] if ENABLE_LINEAR_LOGISTIC else
                [
                    {'label': 'Machine Learning Forecasts', 'value': 'ml'},
                    {'label': 'Stacking Ensemble Forecasts', 'value': 'stacking'}
                ]
            ),
            value='ml',
            style={'width': '30%', 'marginLeft': '20px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
    dcc.Tabs(id="tabs", children=tabs_children),
    dummy_store
]

app.layout = html.Div(layout_children)


# ---------------------------------------------------------
# 11) Dash Callbacks
# ---------------------------------------------------------
if ENABLE_SEASON_SPECIFIC_TRAINER:
    @app.callback(
        Output('site-dropdown', 'options'),
        [Input('season-dropdown', 'value')]
    )
    def update_site_dropdown(selected_season):
        data = raw_data_dict[selected_season]
        options = [{'label': 'All Sites', 'value': 'All Sites'}] + [
            {'label': site, 'value': site} for site in data['Site'].unique()
        ]
        return options
else:
    @app.callback(
        Output('site-dropdown', 'options'),
        [Input('dummy-store', 'data')]
    )
    def update_site_dropdown(_):
        # Always "annual" when no seasons
        data = raw_data_dict['annual']
        options = [{'label': 'All Sites', 'value': 'All Sites'}] + [
            {'label': site, 'value': site} for site in data['Site'].unique()
        ]
        return options


@app.callback(
    Output('analysis-graph', 'figure'),
    [
        Input('forecast-type-dropdown', 'value'),
        Input('site-dropdown', 'value'),
        Input('forecast-method-dropdown', 'value')
    ] + ([Input('season-dropdown', 'value')] if ENABLE_SEASON_SPECIFIC_TRAINER else [])
)
def update_graph(forecast_type, selected_site, forecast_method, *season):
    """
    Updates the main analysis graph based on forecast type (DA_Level or DA_Category),
    site, forecast method, and season (if enabled).
    """
    if ENABLE_SEASON_SPECIFIC_TRAINER:
        selected_season = season[0] if season else 'annual'
    else:
        selected_season = 'annual'
    
    # Choose predictions dictionary
    if forecast_method == 'ml':
        pred = predictions_ml[selected_season]
    elif forecast_method == 'lr':
        pred = (predictions_lr[selected_season]
                if predictions_lr else predictions_ml[selected_season])
    elif forecast_method == 'stacking':
        pred = predictions_stacking[selected_season]
    else:
        pred = predictions_ml[selected_season]  # Default fallback

    if forecast_type == 'DA_Level':
        df_plot = pred['DA_Level']['test_df'].copy()
        site_stats = pred['DA_Level']['site_stats']
        overall_r2 = pred['DA_Level']['overall_r2']
        overall_mae = pred['DA_Level']['overall_mae']
        y_axis_title = 'Domoic Acid Levels'

        df_plot_melted = pd.melt(
            df_plot,
            id_vars=['Date', 'Site'],
            value_vars=['DA_Levels', 'Predicted_DA_Levels'],
            var_name='Metric',
            value_name='Value'
        )

        if (selected_site is None) or (selected_site == 'All Sites'):
            performance_text = f"Overall R² = {overall_r2:.2f}, MAE = {overall_mae:.2f}"
        else:
            if selected_site in site_stats.index:
                site_r2 = site_stats.loc[selected_site, 'r2']
                site_mae = site_stats.loc[selected_site, 'mae']
                performance_text = f"R² = {site_r2:.2f}, MAE = {site_mae:.2f}"
            else:
                performance_text = "No data for selected site."
    else:
        df_plot = pred['DA_Category']['test_df'].copy()
        site_stats = pred['DA_Category']['site_stats']
        overall_accuracy = pred['DA_Category']['overall_accuracy']
        y_axis_title = 'Domoic Acid Category'

        df_plot_melted = pd.melt(
            df_plot,
            id_vars=['Date', 'Site'],
            value_vars=['DA_Category', 'Predicted_DA_Category'],
            var_name='Metric',
            value_name='Value'
        )

        if (selected_site is None) or (selected_site == 'All Sites'):
            performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
        else:
            if selected_site in site_stats.index:
                site_accuracy = site_stats.loc[selected_site]
                performance_text = f"Accuracy = {site_accuracy:.2f}"
            else:
                performance_text = "No data for selected site."

    # Filter by site if chosen
    if selected_site and selected_site != 'All Sites':
        df_plot_melted = df_plot_melted[df_plot_melted['Site'] == selected_site]
    df_plot_melted.sort_values('Date', inplace=True)

    # Build the figure
    if selected_site == 'All Sites' or not selected_site:
        fig = px.line(
            df_plot_melted, x='Date', y='Value',
            color='Site', line_dash='Metric',
            title=f"{y_axis_title} Forecast - All Sites"
        )
    else:
        fig = px.line(
            df_plot_melted, x='Date', y='Value',
            color='Metric', title=f"{y_axis_title} Forecast - {selected_site}"
        )

    fig.update_layout(
        yaxis_title=y_axis_title,
        xaxis_title='Date',
        annotations=[{
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.2,
            'xanchor': 'center', 'yanchor': 'top',
            'text': performance_text,
            'showarrow': False
        }]
    )
    return fig


# Random Anchor Forecast (if enabled)
if ENABLE_RANDOM_ANCHOR_FORECASTS:
    @app.callback(
        Output('random-anchor-container', 'children'),
        [
            Input('forecast-method-dropdown', 'value')
        ] + ([Input('season-dropdown', 'value')] if ENABLE_SEASON_SPECIFIC_TRAINER else [])
    )
    def update_random_anchor_layout(forecast_method, *season):
        """
        Displays forecast graphs for random anchor dates using the chosen method.
        """
        if ENABLE_SEASON_SPECIFIC_TRAINER:
            selected_season = season[0] if season else 'annual'
        else:
            selected_season = 'annual'

        # If LR is selected but disabled, fall back to ML
        if forecast_method == 'lr' and not ENABLE_LINEAR_LOGISTIC:
            forecast_method = 'ml'

        # Retrieve random anchor forecasts based on selected method
        if forecast_method == 'ml':
            df_results, figs_random_site, mae, acc = random_anchors_ml[selected_season]
        elif forecast_method == 'stacking':
            df_results, figs_random_site, mae, acc = random_anchors_stacking[selected_season]
        else:
            df_results, figs_random_site, mae, acc = random_anchors_lr[selected_season]

        # Build a set of figures and overall metrics
        graphs = [dcc.Graph(figure=fig) for site, fig in figs_random_site.items()]
        metrics = html.Div([
            html.H4("Overall Performance on Random Anchor Forecasts"),
            html.Ul([
                html.Li(f"MAE (DA Levels): {mae:.3f}") if mae is not None else html.Li("No MAE (no data)"),
                html.Li(f"Accuracy (DA Category): {acc:.3f}") if acc is not None else html.Li("No Accuracy (no data)")
            ])
        ], style={'marginTop': 20})

        return html.Div(graphs + [metrics])


# ---------------------------------------------------------
# 12) Run the Dash App
# ---------------------------------------------------------
if __name__ == '__main__':
    print("[INFO] Starting Dash app on port 8071")
    app.run_server(debug=True, port=8071)