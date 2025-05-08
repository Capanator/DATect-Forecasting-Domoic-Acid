import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any, Optional
from itertools import combinations # Added for more advanced feature subset generation if needed

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "ENABLE_LAG_FEATURES": False,
    "ENABLE_LINEAR_LOGISTIC": False,
    "DATA_FILE": "final_output.parquet",
    "PORT": 8071,
    "N_SPLITS_TS_EVAL": 200,
    "N_SPLITS_TS_GRIDSEARCH": 5,
    "MIN_TEST_DATE": "2008-01-01",
    "N_JOBS_EVAL": -1, # Set to 1 if debugging parallel issues, -1 for all cores
}

# --- GridSearchCV Parameter Grids ---
PARAM_GRID_REG = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3],
}

PARAM_GRID_CLS = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3],
    "model__class_weight": ["balanced", None],
}
# --- End Parameter Grids ---

# ---------------------------------------------------------
# Data Processing Functions
# ---------------------------------------------------------
def create_numeric_transformer(
    df: pd.DataFrame, drop_cols: List[str]
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    Creates a numeric transformer pipeline and returns the features DataFrame.
    `df` is the DataFrame to process.
    `drop_cols` are columns to remove from `df` to obtain the feature matrix `X`.
    """
    X = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if X.empty or not numeric_cols.any():
        print(f"[WARN] No numeric features found or X is empty in create_numeric_transformer. Input X shape: {X.shape}")
        # Return a "no-op" transformer or handle as an error
        # For simplicity, we'll let it proceed, but it might error downstream if X is truly empty.
        # A more robust solution would return a transformer that does nothing if numeric_cols is empty.

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols)],
        remainder="passthrough", # Keep this if you have non-numeric features you intend to pass through
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer, X


def add_lag_features(
    df: pd.DataFrame, group_col: str, value_col: str, lags: List[int]
) -> pd.DataFrame:
    """
    Adds lag features for the specified value column, grouping by the provided column.
    """
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Loads data, sorts, creates features, categorizes target, and drops rows with critical NaNs.
    """
    print(f"[INFO] Loading data from {file_path}")
    data = pd.read_parquet(file_path, engine="pyarrow")
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["date", "site"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Temporal Features
    day_of_year = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

    # Lag Features
    if CONFIG["ENABLE_LAG_FEATURES"]:
        print("[INFO] Creating lag features...")
        data = add_lag_features(data, group_col="site", value_col="da", lags=[1, 2, 3])
    else:
        print("[INFO] Skipping lag features creation per configuration")

    # Target Categorization
    print("[INFO] Categorizing 'da'...")
    data["da-category"] = pd.cut(
        data["da"],
        bins=[-float("inf"), 5, 20, 40, float("inf")],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(pd.Int64Dtype())

    critical_cols_for_dropna = ["da", "da-category"]
    if CONFIG["ENABLE_LAG_FEATURES"]:
        critical_cols_for_dropna.extend([f"da_lag_{lag}" for lag in [1,2,3]])

    initial_rows = len(data)
    data.dropna(subset=critical_cols_for_dropna, inplace=True)
    print(f"[INFO] Dropped {initial_rows - len(data)} rows due to NaNs in critical columns used for target or essential features.")
    data.reset_index(drop=True, inplace=True)
    return data


# ---------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------
def safe_fit_predict(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model_type="regression"
):
    """
    Fits a model and returns predictions. Assumes y_train has no NaNs
    due to prior cleaning in load_and_prepare_data.
    """
    if X_train.empty:
        print(f"[WARN] X_train is empty in safe_fit_predict for {model_type}. Returning NaNs.")
        return np.full(len(X_test), np.nan)

    y_train = y_train.astype(int) if model_type == "classification" else y_train.astype(float)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_grid_search(
    base_model: Any,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scoring: str,
    cv_splits: int,
    model_type: str,
) -> Dict:
    """
    Runs GridSearchCV using a pipeline.
    X here is the feature DataFrame. The preprocessor is already fitted on this X or a similar one.
    """
    print(f"\n[INFO] Starting GridSearchCV for {model_type}...")
    # The pipeline for GridSearchCV should process the raw X
    # The preprocessor passed here should be one that was fit on the full X used for grid search.
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    cv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Check if X is empty before fitting GridSearchCV
    if X.empty:
        print(f"[ERROR] X is empty for GridSearchCV ({model_type}). Cannot proceed.")
        return {} # Return empty dict or raise error

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1, # Use all cores for GridSearchCV
        verbose=0, # Reduced verbosity
        error_score="raise",
    )
    print(f"[INFO] Fitting GridSearchCV on {len(X)} samples with {X.shape[1]} features...")
    grid_search.fit(X, y) # X here should be the raw features, pipeline handles preprocessing
    print(f"[INFO] GridSearchCV for {model_type} complete. Best Score ({scoring}): {grid_search.best_score_:.4f}")
    print(f"  Best Params: {grid_search.best_params_}")
    best_params_cleaned = {
        key.replace("model__", ""): value for key, value in grid_search.best_params_.items()
    }
    return best_params_cleaned


def get_model_configs(
    best_reg_params: Dict = None, best_cls_params: Dict = None
) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of model configurations.
    """
    best_reg_params = best_reg_params or {}
    best_cls_params = best_cls_params or {}
    model_configs = {
        "ml": {
            "reg": RandomForestRegressor(random_state=42, n_jobs=1, **best_reg_params),
            "cls": RandomForestClassifier(random_state=42, n_jobs=1, **best_cls_params),
        },
        "lr": {
            "reg": LinearRegression(n_jobs=1),
            "cls": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42, n_jobs=1),
        },
    }
    return model_configs


# --- Helper function for parallel processing of a single fold ---
def _process_fold(
    fold_num: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data_sorted: pd.DataFrame,
    reg_model_base: Any,
    cls_model_base: Any,
    # These drop_cols are fixed: ["date", "site", "da", "da-category"]
    # They apply to the DataFrame that *only* contains features_to_use + these cols
    fixed_drop_cols: List[str],
    min_test_date: pd.Timestamp,
    features_to_use: List[str], # NEW: Specific features for this fold
) -> Optional[pd.DataFrame]:
    """
    Processes a single fold using only the specified features_to_use.
    """
    current_test_start_date = data_sorted.iloc[test_idx]["date"].min()
    if current_test_start_date < min_test_date:
        return None

    # print(f"[DEBUG] Processing Fold {fold_num} with features: {features_to_use}") # Optional debug
    train_df_original = data_sorted.iloc[train_idx].copy()
    test_df_original = data_sorted.iloc[test_idx].copy()

    if not features_to_use: # Should be caught earlier, but as a safeguard
        print(f"[ERROR] Fold {fold_num}: No features_to_use provided. Skipping.")
        return None

    # --- Regression ---
    # Prepare DataFrames for the transformer: only selected features + necessary non-feature cols
    cols_for_reg = features_to_use + fixed_drop_cols
    train_df_for_reg_transformer = train_df_original[list(set(cols_for_reg))].copy() # Use set to avoid duplicate columns if features_to_use overlaps fixed_drop_cols
    
    transformer_reg, X_train_reg_selected = create_numeric_transformer(train_df_for_reg_transformer, fixed_drop_cols)
    # X_train_reg_selected now contains only processed columns from features_to_use

    if X_train_reg_selected.empty:
        print(f"[WARN] Fold {fold_num} Reg: X_train_reg_selected is empty after transformation. Skipping reg for this fold.")
        y_pred_reg = np.full(len(test_df_original), np.nan)
    else:
        X_train_reg_proc = transformer_reg.fit_transform(X_train_reg_selected)
        y_train_reg = train_df_for_reg_transformer["da"]

        test_df_for_reg_transformer = test_df_original[list(set(cols_for_reg))].copy()
        X_test_reg_selected = test_df_for_reg_transformer.drop(columns=fixed_drop_cols, errors="ignore")
        X_test_reg_selected = X_test_reg_selected.reindex(columns=X_train_reg_selected.columns, fill_value=0)
        
        if X_test_reg_selected.empty and not X_train_reg_selected.empty : # if X_train was not empty but X_test is
             print(f"[WARN] Fold {fold_num} Reg: X_test_reg_selected is empty while X_train was not. Predictions will be NaN.")
             y_pred_reg = np.full(len(test_df_original), np.nan)
        elif X_test_reg_selected.empty: # if both are empty (already handled for X_train)
             y_pred_reg = np.full(len(test_df_original), np.nan)
        else:
            X_test_reg_proc = transformer_reg.transform(X_test_reg_selected)
            reg_model = clone(reg_model_base)
            y_pred_reg = safe_fit_predict(reg_model, X_train_reg_proc, y_train_reg, X_test_reg_proc, model_type="regression")

    # --- Classification ---
    cols_for_cls = features_to_use + fixed_drop_cols
    train_df_for_cls_transformer = train_df_original[list(set(cols_for_cls))].copy()

    transformer_cls, X_train_cls_selected = create_numeric_transformer(train_df_for_cls_transformer, fixed_drop_cols)

    if X_train_cls_selected.empty:
        print(f"[WARN] Fold {fold_num} Cls: X_train_cls_selected is empty. Skipping cls for this fold.")
        y_pred_cls = np.full(len(test_df_original), np.nan) # Use Int64Dtype compatible NaN if possible or handle later
    else:
        X_train_cls_proc = transformer_cls.fit_transform(X_train_cls_selected)
        y_train_cls = train_df_for_cls_transformer["da-category"]

        test_df_for_cls_transformer = test_df_original[list(set(cols_for_cls))].copy()
        X_test_cls_selected = test_df_for_cls_transformer.drop(columns=fixed_drop_cols, errors="ignore")
        X_test_cls_selected = X_test_cls_selected.reindex(columns=X_train_cls_selected.columns, fill_value=0)
        
        if X_test_cls_selected.empty and not X_train_cls_selected.empty:
            print(f"[WARN] Fold {fold_num} Cls: X_test_cls_selected is empty. Predictions will be NaN.")
            y_pred_cls = np.full(len(test_df_original), np.nan)
        elif X_test_cls_selected.empty:
             y_pred_cls = np.full(len(test_df_original), np.nan)
        else:
            X_test_cls_proc = transformer_cls.transform(X_test_cls_selected)
            cls_model = clone(cls_model_base)
            y_pred_cls = safe_fit_predict(cls_model, X_train_cls_proc, y_train_cls, X_test_cls_proc, model_type="classification")

    # Store predictions on a copy of the original test_df slice
    test_df_output = test_df_original.copy()
    test_df_output["Predicted_da"] = y_pred_reg
    test_df_output["Predicted_da-category"] = y_pred_cls
    return test_df_output
# --- End Helper Function ---


def train_and_evaluate(
    data: pd.DataFrame,
    features_to_use: List[str], # NEW: List of feature column names to use
    method="ml",
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
):
    """
    Trains models using TimeSeriesSplit (parallel) and evaluates performance,
    using only the specified 'features_to_use'.
    """
    print(f"\n[INFO] Starting evaluation for features: {features_to_use[:5]}... (total {len(features_to_use)}) using method: {method}.")
    model_configs = get_model_configs(best_reg_params, best_cls_params)
    if method not in model_configs:
        raise ValueError(f"Method '{method}' not supported")

    reg_model_base = model_configs[method]["reg"]
    cls_model_base = model_configs[method]["cls"]

    data_sorted = data.copy() # Use the full data for splitting
    n_splits = CONFIG["N_SPLITS_TS_EVAL"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    min_test_date = pd.Timestamp(CONFIG["MIN_TEST_DATE"])
    n_jobs = CONFIG["N_JOBS_EVAL"]
    # print(f"[INFO] Using TimeSeriesSplit with n_splits={n_splits}, n_jobs={n_jobs}. First test fold >= {min_test_date.date()}")

    # These are the columns to be dropped from the specialized DataFrames
    # (which contain only features_to_use + these cols) inside _process_fold
    fixed_drop_cols_for_transformer = ["date", "site", "da", "da-category"]

    fold_args = [
        (i + 1, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(tscv.split(data_sorted))
    ]
    # print(f"[INFO] Starting parallel processing for {len(fold_args)} folds...")
    results_list = Parallel(n_jobs=n_jobs, verbose=0)( # Reduced verbosity for feature selection loop
        delayed(_process_fold)(
            fold_num, train_idx, test_idx, data_sorted, reg_model_base,
            cls_model_base, fixed_drop_cols_for_transformer, # Pass fixed drop cols
            min_test_date, features_to_use, # Pass features_to_use
        )
        for fold_num, train_idx, test_idx in fold_args
    )

    fold_test_dfs_with_preds = [df for df in results_list if df is not None]
    processed_fold_count = len(fold_test_dfs_with_preds)
    skipped_fold_count = len(fold_args) - processed_fold_count
    # print(f"[INFO] Parallel processing complete. Processed {processed_fold_count}, skipped {skipped_fold_count} folds.")

    if not fold_test_dfs_with_preds:
        print(f"[ERROR] No valid folds processed for features: {features_to_use}. Cannot aggregate results.")
        # Return a structure indicating failure or empty results
        empty_results_structure = {
            "test_df": pd.DataFrame(), "site_stats": pd.DataFrame(),
            "overall_r2": np.nan, "overall_mae": np.nan, "model": reg_model_base,
        }
        empty_cls_structure = {
             "test_df": pd.DataFrame(), "site_stats": pd.Series(dtype=float),
            "overall_accuracy": np.nan, "model": cls_model_base,
        }
        return {"DA_Level": empty_results_structure, "da-category": empty_cls_structure}


    # print(f"\n[INFO] Aggregating results across {processed_fold_count} folds for features: {features_to_use[:5]}...")
    final_test_df = pd.concat(fold_test_dfs_with_preds).sort_values(["date", "site"])
    final_test_df = final_test_df.drop_duplicates(subset=["date", "site"], keep="last")

    results = {}
    # Evaluate Regression
    df_reg = final_test_df.dropna(subset=["da", "Predicted_da"])
    overall_r2, overall_mae = np.nan, np.nan
    site_stats_reg = pd.DataFrame(columns=["site", "r2", "mae"])
    if not df_reg.empty:
        try:
            overall_r2 = r2_score(df_reg["da"], df_reg["Predicted_da"])
            overall_mae = mean_absolute_error(df_reg["da"], df_reg["Predicted_da"])
            # print(f"[INFO] Overall Regression R2: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
            site_stats_reg = (
                df_reg.groupby("site")
                .apply(lambda x: pd.Series({
                    "r2": r2_score(x["da"], x["Predicted_da"]) if len(x["da"].unique()) > 1 and len(x["Predicted_da"].unique()) > 0 and not x[["da", "Predicted_da"]].isnull().any().any() else np.nan,
                    "mae": mean_absolute_error(x["da"], x["Predicted_da"]) if not x[["da", "Predicted_da"]].isnull().any().any() else np.nan,
                }))
                .reset_index()
            )
        except Exception as e:
            print(f"[ERROR] Calculating regression metrics failed: {e}")


    results["DA_Level"] = {
        "test_df": final_test_df, # This df contains all original columns + predictions
        "site_stats": site_stats_reg.set_index("site") if not site_stats_reg.empty else pd.DataFrame(),
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "model": reg_model_base, # Model instance is the base one, not necessarily retrained on full data here
    }

    # Evaluate Classification
    df_cls = final_test_df.dropna(subset=["da-category", "Predicted_da-category"])
    df_cls["da-category"] = pd.to_numeric(df_cls["da-category"], errors='coerce').astype('Int64')
    df_cls["Predicted_da-category"] = pd.to_numeric(df_cls["Predicted_da-category"], errors='coerce').astype('Int64')
    df_cls = df_cls.dropna(subset=["da-category", "Predicted_da-category"]) # Re-drop after conversion

    overall_accuracy = np.nan
    site_stats_cls = pd.DataFrame(columns=["site", "accuracy"])
    if not df_cls.empty:
        try:
            overall_accuracy = accuracy_score(df_cls["da-category"], df_cls["Predicted_da-category"])
            # print(f"[INFO] Overall Classification Accuracy: {overall_accuracy:.4f}")
            site_stats_cls = (
                df_cls.groupby("site")
                .apply(lambda x: accuracy_score(x["da-category"], x["Predicted_da-category"]) if not x.empty else np.nan)
                .reset_index(name="accuracy")
            )
        except Exception as e:
            print(f"[ERROR] Calculating classification metrics failed: {e}")


    results["da-category"] = {
        "test_df": final_test_df,
        "site_stats": site_stats_cls.set_index("site")["accuracy"] if not site_stats_cls.empty else pd.Series(dtype=float),
        "overall_accuracy": overall_accuracy,
        "model": cls_model_base,
    }

    # print(f"[INFO] {method.upper()} evaluation complete for features: {features_to_use[:5]}...")
    return results


def prepare_all_predictions(
    data: pd.DataFrame,
    features_for_ml: List[str], # Features for ML methods
    features_for_lr: List[str], # Features for LR methods (could be same or different)
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
) -> Dict:
    """
    Runs evaluation for ML and optionally LR methods using specified feature sets.
    """
    predictions = {}
    print(f"\n[INFO] Preparing ML predictions using {len(features_for_ml)} features.")
    predictions["ml"] = train_and_evaluate(
        data, features_to_use=features_for_ml, method="ml",
        best_reg_params=best_reg_params, best_cls_params=best_cls_params
    )
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        print(f"\n[INFO] Preparing LR predictions using {len(features_for_lr)} features.")
        predictions["lr"] = train_and_evaluate(
            data, features_to_use=features_for_lr, method="lr"
        ) # LR models don't use best_reg_params from RF
    return predictions


# ---------------------------------------------------------
# Dash App Setup
# ---------------------------------------------------------
# create_dash_app function remains largely the same.
# It will receive predictions based on the *selected* optimal features.
def create_dash_app(predictions: Dict, data: pd.DataFrame):
    """
    Creates and configures the Dash application.
    """
    app = dash.Dash(__name__)

    # Sites list for dropdown
    sites_list = sorted(data["site"].unique().tolist())

    # Create forecast method options
    forecast_methods = [{"label": "Random Forest (ML)", "value": "ml"}]
    if CONFIG["ENABLE_LINEAR_LOGISTIC"] and "lr" in predictions: # Check if LR results exist
        forecast_methods.append({"label": "Linear/Logistic Regression (LR)", "value": "lr"})

    # Main layout
    app.layout = html.Div(
        [
            html.H1("Domoic Acid Forecast Dashboard"),
            html.Div(
                [
                    html.Label("Select Model Method:"),
                    dcc.Dropdown(
                        id="forecast-method-dropdown",
                        options=forecast_methods,
                        value="ml" if "ml" in predictions else (forecast_methods[0]["value"] if forecast_methods else None), # Default to ml if available
                        clearable=False,
                        style={"width": "30%", "marginLeft": "10px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.H3("Overall Analysis (Aggregated TimeSeriesSplit Folds)"),
                    dcc.Dropdown(
                        id="forecast-type-dropdown",
                        options=[
                            {"label": "DA Levels (Regression)", "value": "DA_Level"},
                            {"label": "DA Category (Classification)", "value": "da-category"},
                        ],
                        value="DA_Level",
                        style={"width": "50%", "marginBottom": "15px"},
                    ),
                    dcc.Dropdown(
                        id="site-dropdown",
                        options=[{"label": "All sites", "value": "All sites"}] + [
                            {"label": site, "value": site} for site in sites_list
                        ],
                        placeholder="Select site (or All sites)",
                        value="All sites", # Default to "All sites"
                        style={"width": "50%", "marginBottom": "15px"},
                    ),
                    dcc.Graph(id="analysis-graph"),
                ]
            ),
            dcc.Store(id="data-store", data={"sites": sites_list}),
        ]
    )

    @app.callback(
        Output("analysis-graph", "figure"),
        [
            Input("forecast-type-dropdown", "value"),
            Input("site-dropdown", "value"),
            Input("forecast-method-dropdown", "value"),
        ],
    )
    def update_graph(forecast_type, selected_site, forecast_method):
        if not forecast_method or forecast_method not in predictions:
            default_method = "ml" if "ml" in predictions else (forecast_methods[0]["value"] if forecast_methods else None)
            print(f"[WARN] Invalid or unavailable forecast method '{forecast_method}', defaulting to '{default_method}'.")
            forecast_method = default_method
            if not forecast_method: # No methods available at all
                 return px.line(title="No forecast methods available.")


        pred_data = predictions.get(forecast_method)
        if not pred_data:
            return px.line(title=f"No prediction data available for {forecast_method.upper()}")

        if (
            forecast_type not in pred_data
            or not pred_data[forecast_type]
            or "test_df" not in pred_data[forecast_type]
            or pred_data[forecast_type]["test_df"] is None
            or pred_data[forecast_type]["test_df"].empty
        ):
            return px.line(title=f"No data for {forecast_type} using {forecast_method.upper()}")

        results_dict = pred_data[forecast_type]
        df_plot_full = results_dict.get("test_df") # This is the df with all original columns + predictions

        # Plotting logic needs the correct actual and predicted columns
        performance_text = "Performance metrics unavailable"

        if forecast_type == "DA_Level":
            overall_r2 = results_dict.get("overall_r2", float("nan"))
            overall_mae = results_dict.get("overall_mae", float("nan"))
            y_axis_title = "Domoic Acid Levels"
            actual_col = "da"
            pred_col = "Predicted_da"
            # Filter df_plot to only necessary columns for melt, and drop rows where actual or pred is NaN for plotting
            df_plot = df_plot_full[["date", "site", actual_col, pred_col]].copy()
            df_plot.dropna(subset=[actual_col, pred_col], how='any', inplace=True)


            if selected_site is None or selected_site == "All sites":
                performance_text = f"Overall R² = {overall_r2:.3f}, MAE = {overall_mae:.3f}"
                df_plot_filtered = df_plot
            else:
                df_plot_filtered = df_plot[df_plot["site"] == selected_site]
                site_stats_df = results_dict.get("site_stats")
                if site_stats_df is not None and not site_stats_df.empty and selected_site in site_stats_df.index:
                    site_r2 = site_stats_df.loc[selected_site, "r2"]
                    site_mae = site_stats_df.loc[selected_site, "mae"]
                    performance_text = f"Site R² = {site_r2:.3f}, MAE = {site_mae:.3f}"
                else:
                     performance_text = f"Stats unavailable for site: {selected_site}"
            
            if df_plot_filtered.empty:
                 return px.line(title=f"No data to plot for {selected_site or 'All sites'} ({forecast_method.upper()}) - Regression")

            df_plot_melted = pd.melt(
                df_plot_filtered,
                id_vars=["date", "site"],
                value_vars=[actual_col, pred_col],
                var_name="Metric",
                value_name="Value",
            )
            metric_order = [actual_col, pred_col]


        elif forecast_type == "da-category":
            overall_accuracy = results_dict.get("overall_accuracy", float("nan"))
            y_axis_title = "Domoic Acid Category"
            actual_col = "da-category"
            pred_col = "Predicted_da-category"

            df_plot = df_plot_full[["date", "site", actual_col, pred_col]].copy()
            # Ensure types are appropriate for plotting and metrics
            df_plot[actual_col] = pd.to_numeric(df_plot[actual_col], errors='coerce').astype('Int64')
            df_plot[pred_col] = pd.to_numeric(df_plot[pred_col], errors='coerce').astype('Int64')
            df_plot.dropna(subset=[actual_col, pred_col], how='any', inplace=True)


            if selected_site is None or selected_site == "All sites":
                performance_text = f"Overall Accuracy = {overall_accuracy:.3f}"
                df_plot_filtered = df_plot
            else:
                df_plot_filtered = df_plot[df_plot["site"] == selected_site]
                site_stats_series = results_dict.get("site_stats") # This is a Series for accuracy
                if site_stats_series is not None and not site_stats_series.empty and selected_site in site_stats_series.index:
                    site_accuracy = site_stats_series.loc[selected_site]
                    performance_text = f"Site Accuracy = {site_accuracy:.3f}"
                else:
                    performance_text = f"Stats unavailable for site: {selected_site}"

            if df_plot_filtered.empty:
                 return px.line(title=f"No data to plot for {selected_site or 'All sites'} ({forecast_method.upper()}) - Classification")

            # Convert to string for categorical plotting after filtering and metric calculation
            df_plot_filtered[actual_col] = df_plot_filtered[actual_col].astype(str)
            df_plot_filtered[pred_col] = df_plot_filtered[pred_col].astype(str)

            df_plot_melted = pd.melt(
                df_plot_filtered,
                id_vars=["date", "site"],
                value_vars=[actual_col, pred_col],
                var_name="Metric",
                value_name="Value",
            )
            metric_order = [actual_col, pred_col] # Order for legend

        else:
            return px.line(title=f"Invalid forecast type: {forecast_type}")

        plot_title_site_part = selected_site if selected_site and selected_site != "All sites" else "All sites"
        plot_title = f"{y_axis_title} Forecast - {plot_title_site_part} ({forecast_method.upper()})"


        df_plot_melted.sort_values("date", inplace=True)
        fig_params = {
            "data_frame": df_plot_melted, "x": "date", "y": "Value",
            "title": plot_title, "category_orders": {"Metric": metric_order},
        }
        # For "All sites", color by site and make metric the line_dash
        # For a specific site, color by metric
        if selected_site == "All sites" or not selected_site:
            fig = px.line(**fig_params, color="site", line_dash="Metric")
        else:
            fig = px.line(**fig_params, color="Metric")


        fig.update_layout(
            yaxis_title=y_axis_title, xaxis_title="Date (Aggregated Test Sets)",
            legend_title_text="Metric/Site" if selected_site == "All sites" or not selected_site else "Metric",
            margin=dict(b=80),
            annotations=[{
                "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.2, # Adjusted y for more space
                "xanchor": "center", "yanchor": "top", "text": performance_text,
                "showarrow": False, "font": {"size": 12},
            }]
        )
        if forecast_type == "da-category":
            # Ensure y-axis categories are sorted numerically if they are numbers as strings
            cat_values = sorted(df_plot_melted["Value"].dropna().unique(), key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=cat_values)

        return fig

    return app

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading and preparing data...")
    data = load_and_prepare_data(CONFIG["DATA_FILE"])

    if data.empty:
        print("[ERROR] Data is empty after loading and preparation. Exiting.")
    else:
        print(f"[INFO] Data loaded successfully. Shape: {data.shape}")

        # --- Initial Hyperparameter Tuning using ALL features ---
        all_current_features = [col for col in data.columns if col not in ["date", "site", "da", "da-category"]]
        
        if not all_current_features:
            print("[ERROR] No features available after data loading (excluding date, site, da, da-category). Exiting.")
        else:
            print(f"[INFO] Initial hyperparameter tuning using all {len(all_current_features)} features: {all_current_features}")

            # Prepare data for initial GridSearchCV
            # The preprocessor for GridSearchCV needs to be created based on the X it will receive
            # create_numeric_transformer returns (transformer, X_transformed_feature_names_df)
            # For GridSearchCV, we pass X (features only) and the preprocessor separately to the pipeline.

            # For Regression GridSearch
            initial_transformer_reg, X_for_grid_reg = create_numeric_transformer(data, ["date", "site", "da", "da-category"])
            y_full_reg = data["da"].astype(float)

            best_rf_reg_params = {}
            if not X_for_grid_reg.empty:
                 best_rf_reg_params = run_grid_search(
                    base_model=RandomForestRegressor(random_state=42, n_jobs=1),
                    param_grid=PARAM_GRID_REG, X=X_for_grid_reg, y=y_full_reg, # Pass features DF
                    preprocessor=initial_transformer_reg, # Pass the transformer that knows how to process X_for_grid_reg
                    scoring="r2",
                    cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Regressor",
                )
            else:
                print("[WARN] X_for_grid_reg is empty. Skipping regressor hyperparameter tuning.")


            # For Classification GridSearch
            initial_transformer_cls, X_for_grid_cls = create_numeric_transformer(data, ["date", "site", "da", "da-category"])
            y_full_cls = data["da-category"].astype(int)
            
            best_rf_cls_params = {}
            if not X_for_grid_cls.empty:
                best_rf_cls_params = run_grid_search(
                    base_model=RandomForestClassifier(random_state=42, n_jobs=1),
                    param_grid=PARAM_GRID_CLS, X=X_for_grid_cls, y=y_full_cls, # Pass features DF
                    preprocessor=initial_transformer_cls, # Pass the transformer
                    scoring="accuracy",
                    cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Classifier",
                )
            else:
                print("[WARN] X_for_grid_cls is empty. Skipping classifier hyperparameter tuning.")


            # --- Feature Selection Loop ---
            feature_subsets_to_evaluate = {}
            if all_current_features: # Ensure there are features to select from
                feature_subsets_to_evaluate["all_features"] = list(all_current_features) # Ensure it's a list copy
                if len(all_current_features) > 1:
                    for feature_to_omit in all_current_features:
                        subset = [f for f in all_current_features if f != feature_to_omit]
                        feature_subsets_to_evaluate[f"all_except_{feature_to_omit}"] = subset
                    for feature_to_keep in all_current_features: # Keep only one
                        feature_subsets_to_evaluate[f"only_{feature_to_keep}"] = [feature_to_keep]
            else:
                print("[WARN] No features identified for feature selection process.")


            best_reg_subset_name = "all_features" # Default
            best_reg_score = -float('inf')
            best_cls_subset_name = "all_features" # Default
            best_cls_score = -float('inf')
            all_subset_performance_results = {}

            if not feature_subsets_to_evaluate or not all_current_features:
                 print("[WARN] No feature subsets to evaluate. Using all available features by default.")
                 # Ensure all_current_features is used if subsets are empty
                 if "all_features" not in feature_subsets_to_evaluate and all_current_features:
                     feature_subsets_to_evaluate["all_features"] = list(all_current_features)


            for subset_name, feature_subset in feature_subsets_to_evaluate.items():
                if not feature_subset:
                    print(f"[INFO] Skipping empty feature subset derived for: {subset_name}")
                    continue

                print(f"\n[INFO] Evaluating ML performance for feature subset '{subset_name}' ({len(feature_subset)} features)")
                subset_eval_results_ml = train_and_evaluate(
                    data,
                    features_to_use=feature_subset,
                    method="ml",
                    best_reg_params=best_rf_reg_params,
                    best_cls_params=best_rf_cls_params
                )
                all_subset_performance_results[subset_name] = {
                    "r2": subset_eval_results_ml["DA_Level"].get("overall_r2", -float('inf')),
                    "mae": subset_eval_results_ml["DA_Level"].get("overall_mae", float('inf')),
                    "accuracy": subset_eval_results_ml["da-category"].get("overall_accuracy", -float('inf'))
                }

                current_r2 = all_subset_performance_results[subset_name]["r2"]
                if pd.notna(current_r2) and current_r2 > best_reg_score:
                    best_reg_score = current_r2
                    best_reg_subset_name = subset_name

                current_acc = all_subset_performance_results[subset_name]["accuracy"]
                if pd.notna(current_acc) and current_acc > best_cls_score:
                    best_cls_score = current_acc
                    best_cls_subset_name = subset_name

            print("\n[INFO] Feature Selection Summary (ML Method):")
            for name, metrics in all_subset_performance_results.items():
                 print(f"  Subset '{name}': R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, Acc={metrics['accuracy']:.4f}, Features={len(feature_subsets_to_evaluate[name])}")

            print(f"\nBest ML Regression Feature Subset: '{best_reg_subset_name}' with R2: {best_reg_score:.4f}")
            print(f"  Features: {feature_subsets_to_evaluate.get(best_reg_subset_name)}")
            print(f"Best ML Classification Feature Subset: '{best_cls_subset_name}' with Accuracy: {best_cls_score:.4f}")
            print(f"  Features: {feature_subsets_to_evaluate.get(best_cls_subset_name)}")

            # For the dashboard, let's choose one optimal set. E.g., best for regression.
            # Or you could have a more complex selection criterion.
            final_ml_features_for_dash = feature_subsets_to_evaluate.get(best_reg_subset_name)
            if not final_ml_features_for_dash: # Fallback if something went wrong
                print("[WARN] Could not determine optimal ML feature set, falling back to all features.")
                final_ml_features_for_dash = list(all_current_features)

            # For Linear/Logistic Regression, we'll use all features or the same set.
            # A separate feature selection loop could be run for LR models if desired.
            final_lr_features_for_dash = list(all_current_features) # Or final_ml_features_for_dash


            print(f"\n[INFO] Preparing final predictions for dashboard...")
            predictions_for_dash = prepare_all_predictions(
                data,
                features_for_ml=final_ml_features_for_dash,
                features_for_lr=final_lr_features_for_dash, # Using all features for LR as an example
                best_reg_params=best_rf_reg_params,
                best_cls_params=best_rf_cls_params
            )

            print("\n[INFO] Creating Dash application...")
            app = create_dash_app(predictions_for_dash, data)

            print(f"[INFO] Starting Dash app on http://127.0.0.1:{CONFIG['PORT']}")
            app.run_server(debug=False, port=CONFIG['PORT'])