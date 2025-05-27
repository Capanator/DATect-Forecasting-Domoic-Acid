import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any, Optional
import random # For random anchor selection
from tqdm import tqdm # Import tqdm for the progress bar

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV # TimeSeriesSplit still used by GridSearchCV
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
# ---------------------------------------------------------x
CONFIG = {
    "ENABLE_LAG_FEATURES": False, # Set to True as in the provided script
    "ENABLE_LINEAR_LOGISTIC": True,
    "DATA_FILE": "final_output.parquet",
    "PORT": 8071,
    "NUM_RANDOM_ANCHORS_PER_SITE_EVAL": 500, # Number of random anchors per site for evaluation
    "N_SPLITS_TS_GRIDSEARCH": 10,    # Still used for GridSearchCV
    "MIN_TEST_DATE": "2008-01-01",  # Predictions will only be made for 'next_date' >= this
    "N_JOBS_EVAL": -1,
    "RANDOM_SEED": 42, # For reproducibility of random anchor selection
}
random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

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
# Data Processing Functions (largely unchanged)
# ---------------------------------------------------------
def create_numeric_transformer(
    df: pd.DataFrame, drop_cols: List[str]
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    Creates a numeric transformer pipeline and returns the features DataFrame.
    """
    X = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols)],
        remainder="passthrough",
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
    data.sort_values(["site", "date"], inplace=True) # Sort by site then date for anchor logic
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

    # Target Categorization - Updated to match the bins from the script provided in the prompt
    print("[INFO] Categorizing 'da'...")
    data["da-category"] = pd.cut(
        data["da"],
        bins=[-float("inf"), 5, 20, 40, float("inf")],
        labels=[0, 1, 2, 3], # Matches the 4 categories
        right=True,
    ).astype(pd.Int64Dtype())


    # Drop rows with NaN values in critical columns (lags, target)
    lag_cols = [f"da_lag_{lag}" for lag in [1,2,3]] if CONFIG["ENABLE_LAG_FEATURES"] else []
    critical_cols = lag_cols + ["da", "da-category"]
    initial_rows = len(data)
    data.dropna(subset=critical_cols, inplace=True)
    print(f"[INFO] Dropped {initial_rows - len(data)} rows due to NaNs in critical columns.")
    data.reset_index(drop=True, inplace=True)
    return data


# ---------------------------------------------------------
# Model Training Functions (safe_fit_predict, run_grid_search, get_model_configs unchanged)
# ---------------------------------------------------------
def safe_fit_predict(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model_type="regression"
):
    """
    Fits a model and returns predictions. Assumes y_train has no NaNs
    due to prior cleaning in load_and_prepare_data.
    """
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
    Runs GridSearchCV using a pipeline. TimeSeriesSplit is used for CV.
    """
    print(f"\n[INFO] Starting GridSearchCV for {model_type}...")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    # GridSearchCV still uses TimeSeriesSplit as it's a sound method for hyperparameter tuning in time series
    cv = TimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1, # Uses all available cores for GridSearchCV
        verbose=0, # Set to 0 or 1 if tqdm provides enough feedback for the main loop
        error_score="raise",
    )
    print(f"[INFO] Fitting GridSearchCV on {len(X)} samples...")
    grid_search.fit(X, y)
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
            "reg": RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **best_reg_params),
            "cls": RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **best_cls_params),
        },
        "lr": {
            "reg": LinearRegression(n_jobs=1),
            "cls": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
        },
    }
    return model_configs


# --- Helper function for processing a single random anchor forecast ---
def process_anchor_forecast(
    anchor_info: Tuple[str, pd.Timestamp], # (site, anchor_date)
    full_data: pd.DataFrame, # The complete dataset, already sorted by site, date
    reg_model_base: Any,
    cls_model_base: Any,
    reg_drop_cols: List[str],
    cls_drop_cols: List[str],
    min_target_date: pd.Timestamp,
    # anchor_num: int # No longer needed for logging here as tqdm handles iteration count
) -> Optional[pd.DataFrame]:
    """
    Processes a single random anchor forecast.
    Trains on data up to anchor_date for a site, predicts the single next point.
    Returns the test DataFrame (single row) with predictions, or None if skipped/error.
    """
    site, anchor_date = anchor_info
    
    site_data = full_data[full_data["site"] == site].copy()
    
    train_df = site_data[site_data["date"] <= anchor_date]
    potential_test_points = site_data[site_data["date"] > anchor_date]

    if train_df.empty or potential_test_points.empty:
        return None

    test_df_single_row = potential_test_points.iloc[[0]].copy() 
    target_prediction_date = test_df_single_row["date"].min()

    if target_prediction_date < min_target_date:
        return None
    
    # --- Regression ---
    transformer_reg, X_train_reg_all_cols = create_numeric_transformer(train_df, reg_drop_cols)
    X_train_reg_proc = transformer_reg.fit_transform(X_train_reg_all_cols)
    y_train_reg = train_df["da"]

    X_test_reg_all_cols = test_df_single_row.drop(columns=reg_drop_cols, errors="ignore")
    X_test_reg_all_cols = X_test_reg_all_cols.reindex(columns=X_train_reg_all_cols.columns, fill_value=0)
    X_test_reg_proc = transformer_reg.transform(X_test_reg_all_cols)

    reg_model = clone(reg_model_base)
    y_pred_reg = safe_fit_predict(reg_model, X_train_reg_proc, y_train_reg, X_test_reg_proc, model_type="regression")

    # --- Classification ---
    transformer_cls, X_train_cls_all_cols = create_numeric_transformer(train_df, cls_drop_cols)
    X_train_cls_proc = transformer_cls.fit_transform(X_train_cls_all_cols)
    y_train_cls = train_df["da-category"]

    X_test_cls_all_cols = test_df_single_row.drop(columns=cls_drop_cols, errors="ignore")
    X_test_cls_all_cols = X_test_cls_all_cols.reindex(columns=X_train_cls_all_cols.columns, fill_value=0)
    X_test_cls_proc = transformer_cls.transform(X_test_cls_all_cols)

    cls_model = clone(cls_model_base)
    y_pred_cls = safe_fit_predict(cls_model, X_train_cls_proc, y_train_cls, X_test_cls_proc, model_type="classification")

    test_df_single_row["Predicted_da"] = y_pred_reg[0] 
    test_df_single_row["Predicted_da-category"] = y_pred_cls[0]
    return test_df_single_row
# --- End Helper Function ---


def train_and_evaluate(
    data: pd.DataFrame,
    method="ml",
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
):
    """
    Trains models using Random Anchor Forecasts (parallel) and evaluates performance.
    """
    print(f"\n[INFO] Starting evaluation using method: {method} with PARALLEL Random Anchor Forecasts.")
    model_configs = get_model_configs(best_reg_params, best_cls_params)
    if method not in model_configs:
        raise ValueError(f"Method '{method}' not supported")

    reg_model_base = model_configs[method]["reg"]
    cls_model_base = model_configs[method]["cls"]

    data_sorted = data.copy() 
    min_target_date = pd.Timestamp(CONFIG["MIN_TEST_DATE"]) 
    n_jobs = CONFIG["N_JOBS_EVAL"]
    num_anchors_per_site = CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"]

    print(f"[INFO] Generating random anchor points. Num anchors per site: {num_anchors_per_site}. Predictions for dates >= {min_target_date.date()}")

    anchor_infos_to_process = []
    unique_sites = data_sorted["site"].unique()
    for site_val in unique_sites:
        site_dates = data_sorted[data_sorted["site"] == site_val]["date"].sort_values().unique()
        if len(site_dates) > 1:
            potential_anchor_dates = site_dates[:-1]
            num_to_sample = min(len(potential_anchor_dates), num_anchors_per_site)
            if num_to_sample > 0:
                sampled_anchor_dates = random.sample(list(potential_anchor_dates), num_to_sample)
                for ad_date in sampled_anchor_dates:
                    anchor_infos_to_process.append((site_val, pd.Timestamp(ad_date)))
    
    if not anchor_infos_to_process:
        print("[ERROR] No valid anchor points generated. Cannot proceed with evaluation.")
        return {"DA_Level": {}, "da-category": {}}
        
    print(f"[INFO] Generated {len(anchor_infos_to_process)} total random anchor points for processing.")
    
    common_cols = ["date", "site"] 
    reg_drop_cols = common_cols + ["da", "da-category"] 
    cls_drop_cols = common_cols + ["da", "da-category"]

    # --- Parallel Execution with tqdm progress bar ---
    print(f"[INFO] Starting parallel processing for {len(anchor_infos_to_process)} anchor forecasts...")
    
    # Wrap the generator with tqdm for a progress bar
    # The description for tqdm will be "Processing Anchors"
    # total specifies the total number of iterations for tqdm
    results_list = Parallel(n_jobs=n_jobs, verbose=1)( # Reduced verbose for Parallel if tqdm is primary
        delayed(process_anchor_forecast)(
            anchor_info, data_sorted, reg_model_base,
            cls_model_base, reg_drop_cols, cls_drop_cols, min_target_date
        )
        for anchor_info in tqdm(anchor_infos_to_process, desc="Processing Random Anchors", total=len(anchor_infos_to_process))
    )

    forecast_dfs_with_preds = [df for df in results_list if df is not None and not df.empty]
    processed_forecast_count = len(forecast_dfs_with_preds)
    skipped_forecast_count = len(anchor_infos_to_process) - processed_forecast_count
    print(f"[INFO] Parallel processing complete. Processed {processed_forecast_count}, skipped {skipped_forecast_count} anchor forecasts.")

    # --- Aggregation and Final Evaluation ---
    print(f"\n[INFO] Aggregating results across {processed_forecast_count} anchor forecasts...")
    final_test_df = pd.concat(forecast_dfs_with_preds).sort_values(["date", "site"])
    final_test_df = final_test_df.drop_duplicates(subset=["date", "site"], keep="last")

    results = {}
    last_reg_model = reg_model_base 
    last_cls_model = cls_model_base

    # Evaluate Regression
    df_reg = final_test_df.dropna(subset=["da", "Predicted_da"])
    overall_r2, overall_mae = np.nan, np.nan
    site_stats_reg = pd.DataFrame(columns=["site", "r2", "mae"])
    if not df_reg.empty:
        overall_r2 = r2_score(df_reg["da"], df_reg["Predicted_da"])
        overall_mae = mean_absolute_error(df_reg["da"], df_reg["Predicted_da"])
        print(f"[INFO] Overall Regression R2: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
        site_stats_reg = (
            df_reg.groupby("site")
            .apply(lambda x: pd.Series({
                "r2": r2_score(x["da"], x["Predicted_da"]) if len(x) > 1 else np.nan, 
                "mae": mean_absolute_error(x["da"], x["Predicted_da"]) if len(x) > 0 else np.nan,
            }))
            .reset_index()
        )

    results["DA_Level"] = {
        "test_df": final_test_df, 
        "site_stats": site_stats_reg.set_index("site"),
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "model": last_reg_model, 
    }

    # Evaluate Classification
    df_cls = final_test_df.dropna(subset=["da-category", "Predicted_da-category"])
    df_cls["da-category"] = pd.to_numeric(df_cls["da-category"], errors='coerce').astype('Int64')
    df_cls["Predicted_da-category"] = pd.to_numeric(df_cls["Predicted_da-category"], errors='coerce').astype('Int64')
    df_cls = df_cls.dropna(subset=["da-category", "Predicted_da-category"])

    overall_accuracy = np.nan
    site_stats_cls = pd.DataFrame(columns=["site", "accuracy"])
    if not df_cls.empty:
        overall_accuracy = accuracy_score(df_cls["da-category"], df_cls["Predicted_da-category"])
        print(f"[INFO] Overall Classification Accuracy: {overall_accuracy:.4f}")
        site_stats_cls = (
            df_cls.groupby("site")
            .apply(lambda x: accuracy_score(x["da-category"], x["Predicted_da-category"]) if not x.empty else np.nan)
            .reset_index(name="accuracy")
        )

    results["da-category"] = {
        "test_df": final_test_df, 
        "site_stats": site_stats_cls.set_index("site")["accuracy"],
        "overall_accuracy": overall_accuracy,
        "model": last_cls_model, 
    }

    print(f"[INFO] {method.upper()} random anchor forecast evaluation complete.")
    return results


def prepare_all_predictions(
    data: pd.DataFrame,
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
) -> Dict:
    """
    Runs evaluation for the ML method and optionally for LR if enabled, using random anchor forecasts.
    """
    predictions = {}
    predictions["ml"] = train_and_evaluate(
        data, method="ml", best_reg_params=best_reg_params, best_cls_params=best_cls_params
    )
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        predictions["lr"] = train_and_evaluate(data, method="lr") 
    return predictions


# ---------------------------------------------------------
# Dash App Setup (Text changes for titles)
# ---------------------------------------------------------
def create_dash_app(predictions: Dict, data: pd.DataFrame):
    """
    Creates and configures the Dash application.
    """
    app = dash.Dash(__name__)
    sites_list = sorted(data["site"].unique().tolist())
    forecast_methods = [{"label": "Random Forest (ML)", "value": "ml"}]
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        forecast_methods.append({"label": "Linear/Logistic Regression (LR)", "value": "lr"})

    app.layout = html.Div(
        [
            html.H1("Domoic Acid Forecast Dashboard (Random Anchor Evaluation)"), 
            html.Div(
                [
                    html.Label("Select Model Method:"),
                    dcc.Dropdown(
                        id="forecast-method-dropdown",
                        options=forecast_methods,
                        value="ml",
                        clearable=False,
                        style={"width": "30%", "marginLeft": "10px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.H3("Overall Analysis (Aggregated Random Anchor Forecasts)"),
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
        if forecast_method not in predictions:
            print(f"[WARN] Invalid forecast method '{forecast_method}', defaulting to 'ml'.")
            forecast_method = "ml"
        if forecast_method not in predictions or not predictions[forecast_method] or \
           forecast_type not in predictions[forecast_method] or \
           not predictions[forecast_method][forecast_type]:
            return px.line(title=f"No prediction data available for {forecast_method.upper()} and {forecast_type}")


        pred_data_type = predictions[forecast_method][forecast_type]
        if ( "test_df" not in pred_data_type or
            pred_data_type["test_df"] is None or
            pred_data_type["test_df"].empty ):
            return px.line(title=f"No data for {forecast_type} using {forecast_method.upper()}")

        results_dict = pred_data_type
        df_plot = results_dict.get("test_df").copy() 
        site_stats = results_dict.get("site_stats")
        performance_text = "Performance metrics unavailable"

        if forecast_type == "DA_Level":
            overall_r2 = results_dict.get("overall_r2", float("nan"))
            overall_mae = results_dict.get("overall_mae", float("nan"))
            y_axis_title = "Domoic Acid Levels"
            actual_col = "da"
            pred_col = "Predicted_da"
            metric_order = [actual_col, pred_col] 
            df_plot_melted = pd.melt(
                df_plot,
                id_vars=["date", "site"], 
                value_vars=[actual_col, pred_col],
                var_name="Metric",
                value_name="Value",
            ).dropna(subset=["Value"])

            if selected_site is None or selected_site == "All sites":
                performance_text = f"Overall R² = {overall_r2:.3f}, MAE = {overall_mae:.3f}"
            elif site_stats is not None and selected_site in site_stats.index:
                site_r2 = site_stats.loc[selected_site, "r2"]
                site_mae = site_stats.loc[selected_site, "mae"]
                performance_text = f"Site R² = {site_r2:.3f}, MAE = {site_mae:.3f}"
            else:
                performance_text = f"Stats unavailable for site: {selected_site}"

        elif forecast_type == "da-category":
            overall_accuracy = results_dict.get("overall_accuracy", float("nan"))
            y_axis_title = "Domoic Acid Category"
            actual_col = "da-category"
            pred_col = "Predicted_da-category"
            metric_order = [actual_col, pred_col]
            df_plot[actual_col] = pd.to_numeric(df_plot[actual_col], errors='coerce').astype('Int64').astype(str)
            df_plot[pred_col] = pd.to_numeric(df_plot[pred_col], errors='coerce').astype('Int64').astype(str)
            df_plot_melted = pd.melt(
                df_plot,
                id_vars=["date", "site"], 
                value_vars=[actual_col, pred_col],
                var_name="Metric",
                value_name="Value",
            ).dropna(subset=["Value"])

            if selected_site is None or selected_site == "All sites":
                performance_text = f"Overall Accuracy = {overall_accuracy:.3f}"
            elif site_stats is not None and selected_site in site_stats.index:
                site_accuracy = site_stats.loc[selected_site] 
                performance_text = f"Site Accuracy = {site_accuracy:.3f}"
            else:
                performance_text = f"Stats unavailable for site: {selected_site}"
        else:
            return px.line(title=f"Invalid forecast type: {forecast_type}")

        plot_title_suffix = f"(Aggregated Random Anchor Forecasts) - {selected_site or 'All sites'} ({forecast_method.upper()})"
        if selected_site and selected_site != "All sites":
            df_plot_melted = df_plot_melted[df_plot_melted["site"] == selected_site]
            plot_title = f"{y_axis_title} Forecast {plot_title_suffix}"
        else:
            plot_title = f"{y_axis_title} Forecast {plot_title_suffix}"

        if df_plot_melted.empty:
            return px.line(title=f"No data to plot for {selected_site or 'All sites'} ({forecast_method.upper()})")

        df_plot_melted.sort_values("date", inplace=True)
        fig_params = {
            "data_frame": df_plot_melted, "x": "date", "y": "Value",
            "title": plot_title, "category_orders": {"Metric": metric_order},
        }
        if selected_site == "All sites" or not selected_site:
            fig = px.line(**fig_params, color="site", line_dash="Metric")
        else: 
            fig = px.line(**fig_params, color="Metric")

        fig.update_layout(
            yaxis_title=y_axis_title, xaxis_title="Date (of Prediction from Random Anchors)", 
            legend_title_text="Metric/Site", margin=dict(b=80),
            annotations=[{
                "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.15,
                "xanchor": "center", "yanchor": "top", "text": performance_text,
                "showarrow": False, "font": {"size": 12},
            }]
        )
        if forecast_type == "da-category":
            cat_values = sorted(df_plot_melted["Value"].unique(), key=lambda x: str(x))
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
        print("[INFO] Data loaded successfully.")

        common_cols_gs = ["date", "site"] 
        reg_drop_cols_gs = common_cols_gs + ["da", "da-category"]
        cls_drop_cols_gs = common_cols_gs + ["da", "da-category"]

        temp_transformer_reg, X_full_reg = create_numeric_transformer(data, reg_drop_cols_gs)
        y_full_reg = data["da"].astype(float)
        temp_transformer_cls, X_full_cls = create_numeric_transformer(data, cls_drop_cols_gs)
        y_full_cls = data["da-category"].astype(int)

        if X_full_reg.empty or X_full_cls.empty:
            print("[ERROR] Feature set for GridSearchCV is empty. Exiting.")
        else:
            best_rf_reg_params = run_grid_search(
                base_model=RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
                param_grid=PARAM_GRID_REG, X=X_full_reg, y=y_full_reg,
                preprocessor=temp_transformer_reg, scoring="r2",
                cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Regressor",
            )
            best_rf_cls_params = run_grid_search(
                base_model=RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
                param_grid=PARAM_GRID_CLS, X=X_full_cls, y=y_full_cls,
                preprocessor=temp_transformer_cls, scoring="accuracy",
                cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Classifier",
            )

            print("\n[INFO] Preparing predictions using PARALLEL Random Anchor Forecast evaluation...")
            predictions = prepare_all_predictions(
                data, best_reg_params=best_rf_reg_params, best_cls_params=best_rf_cls_params
            )

            print("\n[INFO] Creating Dash application...")
            app = create_dash_app(predictions, data)

            print(f"[INFO] Starting Dash app on http://127.0.0.1:{CONFIG['PORT']}")
            app.run_server(debug=False, port=CONFIG['PORT'])