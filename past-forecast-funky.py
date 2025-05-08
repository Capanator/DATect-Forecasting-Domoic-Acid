import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any, Optional
import random # For random anchor selection
from tqdm import tqdm # Import tqdm for the progress bar

from sklearn.compose import ColumnTransformer, make_column_selector # Added make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
from sklearn.feature_selection import RFECV

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "ENABLE_LAG_FEATURES": False,
    "ENABLE_LINEAR_LOGISTIC": True,
    "DATA_FILE": "final_output.parquet", # Make sure this file exists or change path
    "PORT": 8071,
    "NUM_RANDOM_ANCHORS_PER_SITE_EVAL": 100, # Can be reduced for faster testing
    "N_SPLITS_TS_GRIDSEARCH": 5,  # Can be reduced for faster testing
    "MIN_TEST_DATE": "2008-01-01",
    "N_JOBS_EVAL": -1, # Use all cores
    "RANDOM_SEED": 42,
    # Feature Selection Config
    "ENABLE_FEATURE_SELECTION": True,
    "N_JOBS_RFE": -1, # Use all cores
    "N_SPLITS_TS_RFE": 3, # Can be reduced for faster testing
    "MIN_FEATURES_TO_SELECT_RFE": 3,
}
random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

# --- GridSearchCV Parameter Grids ---
PARAM_GRID_REG = {
    "model__n_estimators": [100], # Simplified for speed, can be expanded
    "model__max_depth": [10, None], # Simplified
    # "model__min_samples_split": [2, 5], # Example of more params
    # "model__min_samples_leaf": [1, 3],  # Example of more params
}

PARAM_GRID_CLS = {
    "model__n_estimators": [100], # Simplified
    "model__max_depth": [10, None], # Simplified
    # "model__min_samples_split": [2, 5],
    # "model__min_samples_leaf": [1, 3],
    "model__class_weight": ["balanced", None],
}
# --- End Parameter Grids ---

# ---------------------------------------------------------
# Data Processing Functions
# ---------------------------------------------------------
def create_numeric_transformer( # MODIFIED FUNCTION
    df: pd.DataFrame,
    drop_cols: List[str], # Still useful if feature_columns_to_use is None
    feature_columns_to_use: Optional[List[str]] = None
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    X_input_df: pd.DataFrame
    if feature_columns_to_use is not None:
        # Use only the specified features that are present in the DataFrame
        potential_feature_cols = [col for col in feature_columns_to_use if col in df.columns]
        X_input_df = df[potential_feature_cols].copy()
    else:
        X_input_df = df.drop(columns=drop_cols, errors="ignore")

    if X_input_df.empty:
        print(f"[WARNING] Input DataFrame to create_numeric_transformer is empty. Features specified: {feature_columns_to_use if feature_columns_to_use else 'derived by dropping cols'}")
    
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    
    # Use make_column_selector to identify numeric columns by dtype
    numeric_selector = make_column_selector(dtype_include=np.number)
    
    # Check if any numeric columns would be selected by the selector in X_input_df
    # This is for warning purposes; ColumnTransformer handles empty selections.
    try:
        selected_numeric_cols_by_selector = numeric_selector(X_input_df)
        if not selected_numeric_cols_by_selector and not X_input_df.empty:
             print(f"[WARNING] No numeric columns would be selected by make_column_selector. Features in X_input_df: {list(X_input_df.columns)}")
    except Exception as e:
        # This might happen if X_input_df is unusual, e.g., all non-numeric and selector fails
        if not X_input_df.empty:
            print(f"[WARNING] Error when trying to apply numeric_selector for check: {e}. Features in X_input_df: {list(X_input_df.columns)}")


    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_selector)
        ], 
        remainder="passthrough", 
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer, X_input_df


def add_lag_features(
    df: pd.DataFrame, group_col: str, value_col: str, lags: List[int]
) -> pd.DataFrame:
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from {file_path}")
    try:
        data = pd.read_parquet(file_path, engine="pyarrow")
    except Exception as e:
        print(f"[ERROR] Could not load data from {file_path}: {e}")
        return pd.DataFrame(columns=['date', 'site', 'da', 'sin_day_of_year', 'cos_day_of_year', 'da-category'])

    if 'date' not in data.columns or 'site' not in data.columns or 'da' not in data.columns:
        print(f"[ERROR] Essential columns ('date', 'site', 'da') not found in {file_path}. Please check the data.")
        return pd.DataFrame(columns=['date', 'site', 'da', 'sin_day_of_year', 'cos_day_of_year', 'da-category'])

    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["site", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    day_of_year = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

    if CONFIG["ENABLE_LAG_FEATURES"]:
        print("[INFO] Creating lag features...")
        data = add_lag_features(data, group_col="site", value_col="da", lags=[1, 2, 3])
    else:
        print("[INFO] Skipping lag features creation per configuration.")

    print("[INFO] Categorizing 'da'...")
    data["da-category"] = pd.cut(
        data["da"], bins=[-float("inf"), 5, 20, 40, float("inf")],
        labels=[0, 1, 2, 3], right=True,
    ).astype(pd.Int64Dtype())

    lag_cols = [f"da_lag_{lag}" for lag in [1,2,3]] if CONFIG["ENABLE_LAG_FEATURES"] else []
    critical_cols_for_dropna = lag_cols + ["da", "da-category"] 
    actual_critical_cols = [col for col in critical_cols_for_dropna if col in data.columns]

    initial_rows = len(data)
    if actual_critical_cols:
        data.dropna(subset=actual_critical_cols, inplace=True)
        print(f"[INFO] Dropped {initial_rows - len(data)} rows due to NaNs in critical columns: {actual_critical_cols}.")
    else:
        print("[INFO] No critical columns specified for NaN dropping or they don't exist in data.")
    data.reset_index(drop=True, inplace=True)
    return data

# ---------------------------------------------------------
# Feature Selection Function
# ---------------------------------------------------------
def select_features_rfecv(
    data_df: pd.DataFrame, target_col: str, initial_feature_candidates: List[str],
    base_model_for_rfe: Any, cv_splits_rfe: int, scoring: str,
    min_features_to_select: int = 1, n_jobs_rfe: int = -1, model_type_log: str = "model"
) -> List[str]:
    print(f"\n[INFO] Starting feature selection ({model_type_log}) using RFECV with scoring: {scoring}...")
    
    valid_initial_features = [f for f in initial_feature_candidates if f in data_df.columns]
    if not valid_initial_features:
        print(f"[ERROR] No valid initial feature candidates found in data for RFECV ({model_type_log}). Returning empty list.")
        return []
    
    X_potential = data_df[valid_initial_features].copy()
    y_target = data_df[target_col].copy()

    if model_type_log == "Classifier": y_target = y_target.astype(int)
    else: y_target = y_target.astype(float)

    common_index = X_potential.index.intersection(y_target.index)
    X_potential = X_potential.loc[common_index]
    y_target = y_target.loc[common_index]
    valid_target_indices = ~y_target.isna()
    X_potential = X_potential[valid_target_indices]
    y_target = y_target[valid_target_indices]

    if X_potential.empty or y_target.empty:
        print(f"[ERROR] X_potential or y_target is empty for RFECV ({model_type_log}). Returning all valid initial candidates.")
        return valid_initial_features

    # The preprocessor is defined to operate on the columns specified by valid_initial_features
    # using make_column_selector.
    initial_preprocessor_for_rfe, _ = create_numeric_transformer(
        df=data_df, # Pass the original df context if needed by create_numeric_transformer
        drop_cols=[], # No columns are dropped by name here; scope is defined by feature_columns_to_use
        feature_columns_to_use=valid_initial_features 
    )
    pipeline_rfe = Pipeline([
        ("preprocessor", initial_preprocessor_for_rfe), 
        ("model_rfe", clone(base_model_for_rfe))
    ])
    cv_ts = TimeSeriesSplit(n_splits=cv_splits_rfe)
    rfe_selector = RFECV(
        estimator=pipeline_rfe, step=1, cv=cv_ts, scoring=scoring,
        min_features_to_select=min_features_to_select, n_jobs=n_jobs_rfe, verbose=0, 
    )
    print(f"[INFO] Fitting RFECV ({model_type_log}) on {len(X_potential)} samples, {len(X_potential.columns)} potential features...")
    rfe_selector.fit(X_potential, y_target) # X_potential is a DataFrame
    selected_original_feature_names = list(X_potential.columns[rfe_selector.support_])
    print(f"[INFO] RFECV ({model_type_log}) selected {rfe_selector.n_features_} features: {selected_original_feature_names}")
    if hasattr(rfe_selector, "cv_results_") and 'mean_test_score' in rfe_selector.cv_results_:
         print(f"  RFECV mean test scores: {rfe_selector.cv_results_['mean_test_score']}")
    return selected_original_feature_names

# ---------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------
def safe_fit_predict(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model_type="regression"
):
    y_true_type = int if model_type == "classification" else float
    if y_train.hasnans:
        valid_idx = ~y_train.isna()
        X_train = X_train.loc[valid_idx]; y_train = y_train.loc[valid_idx]
    if X_train.empty or y_train.empty:
        return np.full(len(X_test), np.nan if model_type == "regression" else pd.NA)
    y_train = y_train.astype(y_true_type)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def run_grid_search(
    base_model: Any, param_grid: Dict, X: pd.DataFrame, y: pd.Series,
    preprocessor: ColumnTransformer, scoring: str, cv_splits: int, model_type: str,
) -> Dict:
    print(f"\n[INFO] Starting GridSearchCV for {model_type}...")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    cv = TimeSeriesSplit(n_splits=cv_splits)
    common_idx = X.index.intersection(y.index)
    X_aligned, y_aligned = X.loc[common_idx], y.loc[common_idx]
    valid_y_idx = ~y_aligned.isna()
    X_aligned, y_aligned = X_aligned.loc[valid_y_idx], y_aligned.loc[valid_y_idx]
    if X_aligned.empty or y_aligned.empty: 
        print(f"[ERROR] X or y is empty for GridSearchCV {model_type}. Returning empty best_params.")
        return {}
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring=scoring, cv=cv,
        n_jobs=-1, verbose=0, error_score="raise",
    )
    print(f"[INFO] Fitting GridSearchCV on {len(X_aligned)} samples...")
    grid_search.fit(X_aligned, y_aligned)
    print(f"[INFO] GridSearchCV for {model_type} complete. Best Score ({scoring}): {grid_search.best_score_:.4f}")
    print(f"  Best Params: {grid_search.best_params_}")
    return {key.replace("model__", ""): value for key, value in grid_search.best_params_.items()}

def get_model_configs(
    best_reg_params: Dict = None, best_cls_params: Dict = None
) -> Dict[str, Dict[str, Any]]:
    best_reg_params = best_reg_params or {}; best_cls_params = best_cls_params or {}
    return {
        "ml": {"reg": RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **best_reg_params),
               "cls": RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **best_cls_params)},
        "lr": {"reg": LinearRegression(n_jobs=1),
               "cls": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=CONFIG["RANDOM_SEED"], n_jobs=1)},
    }

def process_anchor_forecast(
    anchor_info: Tuple[str, pd.Timestamp],
    full_data: pd.DataFrame, 
    reg_model_base: Any,
    cls_model_base: Any,
    selected_reg_features_names: List[str],
    selected_cls_features_names: List[str],
    min_target_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    site_for_prediction, anchor_date = anchor_info

    train_df = full_data[full_data["date"] <= anchor_date].copy()
    site_specific_data_for_test = full_data[full_data["site"] == site_for_prediction].copy()
    potential_test_points = site_specific_data_for_test[site_specific_data_for_test["date"] > anchor_date]

    if train_df.empty or potential_test_points.empty: return None
    test_df_single_row = potential_test_points.iloc[[0]].copy() 
    if test_df_single_row["date"].min() < min_target_date: return None
    
    std_drop_cols_for_features = ["da", "da-category", "site", "date"] 

    y_pred_reg_val = np.nan 
    if selected_reg_features_names:
        valid_reg_features = [f for f in selected_reg_features_names if f in train_df.columns]
        if valid_reg_features:
            transformer_reg, X_train_reg_unprocessed = create_numeric_transformer(
                train_df, drop_cols=std_drop_cols_for_features, feature_columns_to_use=valid_reg_features
            )
            y_train_reg = train_df["da"] 
            aligned_y_train_reg = y_train_reg.loc[X_train_reg_unprocessed.index].copy()
            valid_y_idx_reg = ~aligned_y_train_reg.isna()
            X_train_reg_unprocessed_clean = X_train_reg_unprocessed.loc[valid_y_idx_reg]
            aligned_y_train_reg_clean = aligned_y_train_reg.loc[valid_y_idx_reg]
            if not X_train_reg_unprocessed_clean.empty:
                X_train_reg_proc = transformer_reg.fit_transform(X_train_reg_unprocessed_clean)
                X_test_reg_unprocessed = test_df_single_row[[f for f in valid_reg_features if f in test_df_single_row.columns]]
                X_test_reg_unprocessed = X_test_reg_unprocessed.reindex(columns=X_train_reg_unprocessed_clean.columns, fill_value=0)
                X_test_reg_proc = transformer_reg.transform(X_test_reg_unprocessed)
                y_pred_reg_val = safe_fit_predict(clone(reg_model_base), X_train_reg_proc, aligned_y_train_reg_clean, X_test_reg_proc, "regression")[0]

    y_pred_cls_val = pd.NA
    if selected_cls_features_names:
        valid_cls_features = [f for f in selected_cls_features_names if f in train_df.columns]
        if valid_cls_features:
            transformer_cls, X_train_cls_unprocessed = create_numeric_transformer(
                train_df, drop_cols=std_drop_cols_for_features, feature_columns_to_use=valid_cls_features
            )
            y_train_cls = train_df["da-category"]
            aligned_y_train_cls = y_train_cls.loc[X_train_cls_unprocessed.index].copy()
            valid_y_idx_cls = ~aligned_y_train_cls.isna()
            X_train_cls_unprocessed_clean = X_train_cls_unprocessed.loc[valid_y_idx_cls]
            aligned_y_train_cls_clean = aligned_y_train_cls.loc[valid_y_idx_cls]
            if not X_train_cls_unprocessed_clean.empty:
                X_train_cls_proc = transformer_cls.fit_transform(X_train_cls_unprocessed_clean)
                X_test_cls_unprocessed = test_df_single_row[[f for f in valid_cls_features if f in test_df_single_row.columns]]
                X_test_cls_unprocessed = X_test_cls_unprocessed.reindex(columns=X_train_cls_unprocessed_clean.columns, fill_value=0)
                X_test_cls_proc = transformer_cls.transform(X_test_cls_unprocessed)
                y_pred_cls_val = safe_fit_predict(clone(cls_model_base), X_train_cls_proc, aligned_y_train_cls_clean, X_test_cls_proc, "classification")[0]

    test_df_single_row["Predicted_da"] = y_pred_reg_val
    test_df_single_row["Predicted_da-category"] = y_pred_cls_val
    return test_df_single_row

def train_and_evaluate(
    data: pd.DataFrame, method="ml", best_reg_params: Dict = None, best_cls_params: Dict = None,
    selected_reg_features_names: Optional[List[str]] = None, selected_cls_features_names: Optional[List[str]] = None,
):
    print(f"\n[INFO] Starting evaluation: {method.upper()} method.")
    model_configs = get_model_configs(best_reg_params, best_cls_params)
    reg_model_base, cls_model_base = model_configs[method]["reg"], model_configs[method]["cls"]
    data_sorted, min_target_date = data.copy(), pd.Timestamp(CONFIG["MIN_TEST_DATE"])
    n_jobs, num_anchors_per_site = CONFIG["N_JOBS_EVAL"], CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"]
    anchor_infos_to_process = []
    for site_val in data_sorted["site"].unique():
        site_dates = np.sort(data_sorted[data_sorted["site"] == site_val]["date"].unique())
        if len(site_dates) > 1:
            potential_anchor_dates = site_dates[:-1]
            num_to_sample = min(len(potential_anchor_dates), num_anchors_per_site)
            if num_to_sample > 0:
                sampled_anchor_dates_np = random.sample(list(potential_anchor_dates), num_to_sample)
                sampled_anchor_dates_ts = [pd.Timestamp(d) for d in sampled_anchor_dates_np]
                for ad_date in sampled_anchor_dates_ts: anchor_infos_to_process.append((site_val, ad_date))
    if not anchor_infos_to_process: return {"DA_Level": {}, "da-category": {}}
    print(f"[INFO] Generated {len(anchor_infos_to_process)} random anchor points.")
    
    all_features_fallback = [col for col in data.columns if col not in ["date", "site", "da", "da-category"]]
    current_selected_reg_features = selected_reg_features_names if selected_reg_features_names else all_features_fallback
    current_selected_cls_features = selected_cls_features_names if selected_cls_features_names else all_features_fallback
    if not current_selected_reg_features: current_selected_reg_features = all_features_fallback
    if not current_selected_cls_features: current_selected_cls_features = all_features_fallback

    results_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_anchor_forecast)(
            anchor_info, data_sorted, reg_model_base, cls_model_base, 
            current_selected_reg_features, current_selected_cls_features, min_target_date
        )
        for anchor_info in tqdm(anchor_infos_to_process, desc=f"Processing Anchors ({method.upper()})", total=len(anchor_infos_to_process))
    )
    forecast_dfs = [df for df in results_list if df is not None and not df.empty]
    print(f"[INFO] Processed {len(forecast_dfs)} non-empty anchor forecasts out of {len(anchor_infos_to_process)} generated.")
    if not forecast_dfs: return {"DA_Level": {}, "da-category": {}}
    final_test_df = pd.concat(forecast_dfs).sort_values(["date", "site"]).drop_duplicates(subset=["date", "site"], keep="last")
    results = {}
    df_reg = final_test_df.dropna(subset=["da", "Predicted_da"])
    overall_r2, overall_mae = (r2_score(df_reg["da"], df_reg["Predicted_da"]), mean_absolute_error(df_reg["da"], df_reg["Predicted_da"])) if not df_reg.empty else (np.nan, np.nan)
    site_stats_reg = df_reg.groupby("site").apply(lambda x: pd.Series({"r2": r2_score(x["da"], x["Predicted_da"]) if len(x)>1 else np.nan, "mae": mean_absolute_error(x["da"], x["Predicted_da"]) if len(x)>0 else np.nan})).reset_index() if not df_reg.empty else pd.DataFrame(columns=["site","r2","mae"])
    results["DA_Level"] = {"test_df": final_test_df, "site_stats": site_stats_reg.set_index("site") if "site" in site_stats_reg else pd.DataFrame(), "overall_r2": overall_r2, "overall_mae": overall_mae, "model": reg_model_base}
    df_cls = final_test_df.dropna(subset=["da-category", "Predicted_da-category"])
    if not df_cls.empty:
        df_cls["da-category"] = pd.to_numeric(df_cls["da-category"], errors='coerce').astype('Int64')
        df_cls["Predicted_da-category"] = pd.to_numeric(df_cls["Predicted_da-category"], errors='coerce').astype('Int64')
        df_cls.dropna(subset=["da-category", "Predicted_da-category"], inplace=True)
    overall_accuracy = accuracy_score(df_cls["da-category"], df_cls["Predicted_da-category"]) if not df_cls.empty else np.nan
    site_stats_cls = df_cls.groupby("site").apply(lambda x: accuracy_score(x["da-category"], x["Predicted_da-category"]) if not x.empty else np.nan).reset_index(name="accuracy") if not df_cls.empty else pd.DataFrame(columns=["site","accuracy"])
    results["da-category"] = {"test_df": final_test_df, "site_stats": site_stats_cls.set_index("site")["accuracy"] if "site" in site_stats_cls and "accuracy" in site_stats_cls else pd.Series(dtype=float), "overall_accuracy": overall_accuracy, "model": cls_model_base}
    print(f"[INFO] {method.upper()} evaluation complete. R2: {overall_r2:.3f}, MAE: {overall_mae:.3f}, Acc: {overall_accuracy:.3f}")
    return results

def prepare_all_predictions(
    data: pd.DataFrame, best_reg_params: Dict = None, best_cls_params: Dict = None,
    selected_reg_features_names: Optional[List[str]] = None, selected_cls_features_names: Optional[List[str]] = None
) -> Dict:
    predictions = {}
    print("\n--- Evaluating ML Models (Random Forest) ---")
    predictions["ml"] = train_and_evaluate(
        data, method="ml", best_reg_params=best_reg_params, best_cls_params=best_cls_params,
        selected_reg_features_names=selected_reg_features_names, selected_cls_features_names=selected_cls_features_names
    )
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        print("\n--- Evaluating Linear/Logistic Regression Models ---")
        all_potential_features = [col for col in data.columns if col not in ["date", "site", "da", "da-category"]]
        lr_reg_features = selected_reg_features_names if selected_reg_features_names else all_potential_features
        lr_cls_features = selected_cls_features_names if selected_cls_features_names else all_potential_features
        predictions["lr"] = train_and_evaluate(
            data, method="lr", selected_reg_features_names=lr_reg_features, selected_cls_features_names=lr_cls_features
        )
    return predictions

# ---------------------------------------------------------
# Dash App Setup
# ---------------------------------------------------------
def create_dash_app(predictions: Dict, data: pd.DataFrame):
    app = dash.Dash(__name__)
    sites_list = sorted(data["site"].unique().tolist()) if not data.empty and "site" in data.columns else []
    forecast_methods = [{"label": "Random Forest (ML)", "value": "ml"}]
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        forecast_methods.append({"label": "Linear/Logistic Regression (LR)", "value": "lr"})

    app.layout = html.Div([
        html.H1("Domoic Acid Forecast Dashboard (Random Anchor Evaluation)"),
        html.Div([
            html.Label("Select Model Method:"),
            dcc.Dropdown(id="forecast-method-dropdown", options=forecast_methods, value="ml", clearable=False, style={"width": "30%", "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
        html.Div([
            html.H3("Overall Analysis (Aggregated Random Anchor Forecasts)"),
            dcc.Dropdown(id="forecast-type-dropdown", options=[{"label": "DA Levels (Regression)", "value": "DA_Level"}, {"label": "DA Category (Classification)", "value": "da-category"}], value="DA_Level", style={"width": "50%", "marginBottom": "15px"}),
            dcc.Dropdown(id="site-dropdown", options=[{"label": "All sites", "value": "All sites"}] + [{"label": site, "value": site} for site in sites_list], value="All sites", placeholder="Select site (or All sites)", style={"width": "50%", "marginBottom": "15px"}),
            dcc.Graph(id="analysis-graph"),
        ]),
        dcc.Store(id="data-store", data={"sites": sites_list}),
    ])

    @app.callback(Output("analysis-graph", "figure"), [Input("forecast-type-dropdown", "value"), Input("site-dropdown", "value"), Input("forecast-method-dropdown", "value")])
    def update_graph(forecast_type, selected_site, forecast_method):
        fig_error = px.line(title="No data available or error in processing display.")
        try:
            if not predictions or forecast_method not in predictions or not predictions[forecast_method]:
                return fig_error.update_layout(title=f"No prediction data for method '{forecast_method}'.")
            if forecast_type not in predictions[forecast_method] or not predictions[forecast_method][forecast_type]:
                return fig_error.update_layout(title=f"No data for '{forecast_type}' using '{forecast_method.upper()}'.")
            
            results_dict = predictions[forecast_method][forecast_type]
            if "test_df" not in results_dict or results_dict["test_df"] is None or results_dict["test_df"].empty:
                return fig_error.update_layout(title=f"No test data in predictions for '{forecast_type}' using '{forecast_method.upper()}'.")

            df_plot_full = results_dict["test_df"].copy()
            df_plot = df_plot_full[df_plot_full["site"] == selected_site].copy() if selected_site and selected_site != "All sites" else df_plot_full.copy()
            
            if df_plot.empty: 
                title_suffix = f" for site '{selected_site}'" if selected_site and selected_site != "All sites" else ""
                return fig_error.update_layout(title=f"No data to plot{title_suffix} for '{forecast_type}'.")

            site_stats, performance_text = results_dict.get("site_stats"), "Metrics unavailable"
            actual_col, pred_col, y_axis_title = "", "", ""

            if forecast_type == "DA_Level":
                y_axis_title, actual_col, pred_col = "Domoic Acid Levels", "da", "Predicted_da"
                df_plot.dropna(subset=[actual_col, pred_col], inplace=True)
                if df_plot.empty: return fig_error.update_layout(title=f"No valid regression data for '{selected_site or 'All sites'}'.")
                overall_r2, overall_mae = results_dict.get("overall_r2", float("nan")), results_dict.get("overall_mae", float("nan"))
                performance_text = f"Overall R²={overall_r2:.3f}, MAE={overall_mae:.3f}"
                if selected_site and selected_site != "All sites" and site_stats is not None and not site_stats.empty and selected_site in site_stats.index:
                    try: performance_text = f"Site R²={site_stats.loc[selected_site, 'r2']:.3f}, MAE={site_stats.loc[selected_site, 'mae']:.3f}"
                    except KeyError: performance_text = "Site metrics missing for regression."
            elif forecast_type == "da-category":
                y_axis_title, actual_col, pred_col = "Domoic Acid Category", "da-category", "Predicted_da-category"
                for col in [actual_col, pred_col]: df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce').astype('Int64')
                df_plot.dropna(subset=[actual_col, pred_col], inplace=True)
                if df_plot.empty: return fig_error.update_layout(title=f"No valid classification data for '{selected_site or 'All sites'}'.")
                for col in [actual_col, pred_col]: df_plot[col] = df_plot[col].astype(str) 
                overall_accuracy = results_dict.get("overall_accuracy", float("nan"))
                performance_text = f"Overall Accuracy={overall_accuracy:.3f}"
                if selected_site and selected_site != "All sites" and site_stats is not None and not site_stats.empty and selected_site in site_stats.index:
                    try: performance_text = f"Site Accuracy={site_stats.loc[selected_site]:.3f}" 
                    except KeyError: performance_text = "Site metrics missing for classification."
            
            df_plot_melted = pd.melt(df_plot, id_vars=["date", "site"], value_vars=[actual_col, pred_col], var_name="Metric", value_name="Value")
            if df_plot_melted.empty: return fig_error.update_layout(title=f"No data to plot after melting for '{selected_site or 'All sites'}'.")
            
            df_plot_melted.sort_values("date", inplace=True)
            fig_params = {"data_frame": df_plot_melted, "x": "date", "y": "Value", "title": f"{y_axis_title} Forecast ({selected_site or 'All sites'}) - {forecast_method.upper()}", "category_orders": {"Metric": [actual_col, pred_col]}}
            fig_color_by = "site" if (selected_site == "All sites" or not selected_site) and "site" in df_plot_melted.columns else "Metric"
            fig_line_dash = "Metric" if (selected_site == "All sites" or not selected_site) and "Metric" in df_plot_melted.columns else None
            fig = px.line(**fig_params, color=fig_color_by, line_dash=fig_line_dash)
            fig.update_layout(yaxis_title=y_axis_title, xaxis_title="Date", legend_title_text="Metric/Site", margin=dict(b=100), annotations=[{"xref": "paper", "yref": "paper", "x": 0.5, "y": -0.25, "xanchor": "center", "yanchor": "top", "text": performance_text, "showarrow": False, "font": {"size": 12}}])
            if forecast_type == "da-category": fig.update_yaxes(type="category", categoryorder="array", categoryarray=sorted(df_plot_melted["Value"].unique(), key=str))
            return fig
        except Exception as e:
            print(f"Error during Dash graph update: {e}")
            return fig_error.update_layout(title=f"Error rendering graph: {str(e)[:100]}...") # Show first 100 chars of error
    return app

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading and preparing data...")
    data = load_and_prepare_data(CONFIG["DATA_FILE"])
    best_rf_reg_params, best_rf_cls_params = {}, {}

    if data.empty or 'da' not in data.columns:
        print("[ERROR] Data is empty or essential columns missing after loading. Exiting.")
    else:
        print("[INFO] Data loaded successfully.")
        common_drop_cols = ["date", "site", "da", "da-category"]
        all_potential_features = [col for col in data.columns if col not in common_drop_cols]
        
        if not all_potential_features:
            print("[ERROR] No potential features found after excluding common drop columns. Check data and column names. Exiting.")
            exit()

        selected_reg_features, selected_cls_features = all_potential_features[:], all_potential_features[:] # Default to all
        
        if CONFIG["ENABLE_FEATURE_SELECTION"]:
            print("\n======== FEATURE SELECTION PHASE ========")
            selected_reg_features = select_features_rfecv(data.copy(), "da", all_potential_features, RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1), CONFIG["N_SPLITS_TS_RFE"], "r2", CONFIG["MIN_FEATURES_TO_SELECT_RFE"], CONFIG["N_JOBS_RFE"], "Regressor")
            if not selected_reg_features: selected_reg_features = all_potential_features[:] 
            
            selected_cls_features = select_features_rfecv(data.copy(), "da-category", all_potential_features, RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1), CONFIG["N_SPLITS_TS_RFE"], "accuracy", CONFIG["MIN_FEATURES_TO_SELECT_RFE"], CONFIG["N_JOBS_RFE"], "Classifier")
            if not selected_cls_features: selected_cls_features = all_potential_features[:]
        else: print("[INFO] Feature selection disabled. Using all potential features.")

        if not selected_reg_features : selected_reg_features = all_potential_features[:] # Final fallback
        if not selected_cls_features : selected_cls_features = all_potential_features[:] # Final fallback


        print("\n======== HYPERPARAMETER TUNING PHASE ========")
        y_full_reg = data["da"].astype(float)
        valid_gs_reg_features = [f for f in selected_reg_features if f in data.columns]
        if not valid_gs_reg_features:
            print("[ERROR] No valid regression features for GridSearchCV. Using all potential features.")
            valid_gs_reg_features = all_potential_features[:]
        X_gs_reg = data[valid_gs_reg_features].copy(); y_gs_reg = y_full_reg.loc[X_gs_reg.index]
        preprocessor_gs_reg, _ = create_numeric_transformer(data, common_drop_cols, valid_gs_reg_features)
        if not X_gs_reg.empty: best_rf_reg_params = run_grid_search(RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"],n_jobs=1), PARAM_GRID_REG, X_gs_reg, y_gs_reg, preprocessor_gs_reg, "r2", CONFIG["N_SPLITS_TS_GRIDSEARCH"], "Regressor")
        
        y_full_cls = data["da-category"].astype(pd.Int64Dtype())
        valid_gs_cls_features = [f for f in selected_cls_features if f in data.columns]
        if not valid_gs_cls_features:
            print("[ERROR] No valid classification features for GridSearchCV. Using all potential features.")
            valid_gs_cls_features = all_potential_features[:]
        X_gs_cls = data[valid_gs_cls_features].copy(); y_gs_cls = y_full_cls.loc[X_gs_cls.index]
        preprocessor_gs_cls, _ = create_numeric_transformer(data, common_drop_cols, valid_gs_cls_features)
        if not X_gs_cls.empty: best_rf_cls_params = run_grid_search(RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"],n_jobs=1), PARAM_GRID_CLS, X_gs_cls, y_gs_cls, preprocessor_gs_cls, "accuracy", CONFIG["N_SPLITS_TS_GRIDSEARCH"], "Classifier")
        
        print("\n======== MODEL EVALUATION PHASE ========")
        predictions = prepare_all_predictions(data, best_rf_reg_params, best_rf_cls_params, selected_reg_features, selected_cls_features)
        
        print("\n[INFO] Creating Dash application...")
        app = create_dash_app(predictions, data)
        print(f"[INFO] Starting Dash app on http://127.0.0.1:{CONFIG['PORT']}")
        app.run_server(debug=False, port=CONFIG['PORT'])