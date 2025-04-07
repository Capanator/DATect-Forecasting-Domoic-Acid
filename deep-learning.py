import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ---------------------------------------------------------
# PyTorch Imports and Model Definitions
# ---------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# --- Deep Neural Networks (Nonlinear) ---
class TorchRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(TorchRegressor, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class TorchClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=4):
        super(TorchClassifier, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# --- Linear Models ---
class TorchLinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(TorchLinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

class TorchLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(TorchLinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "ENABLE_LINEAR_LOGISTIC": True,
    "DATA_FILE": "final_output.parquet",
    "PORT": 8071,
    "N_SPLITS_TS_EVAL": 200,
    "N_SPLITS_TS_GRIDSEARCH": 5,
    "MIN_TEST_DATE": "2008-01-01",
    "N_JOBS_EVAL": -1,
    # Hyperparameters for PyTorch training
    "NUM_EPOCHS": 50,
    "LEARNING_RATE": 0.001,
    "BATCH_SIZE": 32,
}

# ---------------------------------------------------------
# Data Processing Functions
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
    data.sort_values(["date", "site"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Temporal Features
    day_of_year = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

    # Lag Features
    print("[INFO] Creating lag features...")
    data = add_lag_features(data, group_col="site", value_col="da", lags=[1, 2, 3])

    # Target Categorization
    print("[INFO] Categorizing 'da'...")
    data["da-category"] = pd.cut(
        data["da"],
        bins=[-float("inf"), 5, 20, 40, float("inf")],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(pd.Int64Dtype())

    # Drop rows with NaN values in critical columns (lags, target)
    critical_cols = [f"da_lag_{lag}" for lag in [1, 2, 3]] + ["da", "da-category"]
    initial_rows = len(data)
    data.dropna(subset=critical_cols, inplace=True)
    print(f"[INFO] Dropped {initial_rows - len(data)} rows due to NaNs in critical columns.")
    data.reset_index(drop=True, inplace=True)
    return data

# ---------------------------------------------------------
# PyTorch Training & Prediction Function
# ---------------------------------------------------------
def safe_fit_predict(
    model_class, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    model_type="regression",
    num_epochs: int = CONFIG["NUM_EPOCHS"],
    lr: float = CONFIG["LEARNING_RATE"],
    batch_size: int = CONFIG["BATCH_SIZE"],
):
    """
    Trains a PyTorch neural network and returns predictions along with the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    
    if model_type == "regression":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
        criterion = nn.MSELoss()
    else:
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()
    
    # Create a new model instance with the proper input dimension
    model = model_class(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        if model_type == "regression":
            predictions = predictions.cpu().numpy().flatten()
        else:
            predictions = predictions.cpu().numpy().argmax(axis=1)
    
    return predictions, model

# ---------------------------------------------------------
# Dummy Grid Search Function for PyTorch Models
# ---------------------------------------------------------
def run_grid_search(*args, **kwargs):
    """
    Grid search is not implemented for PyTorch models in this example.
    Default hyperparameters (from CONFIG) are used.
    """
    print("[INFO] Grid search not implemented for PyTorch models. Using default hyperparameters.")
    return {}

# ---------------------------------------------------------
# Model Configuration Functions
# ---------------------------------------------------------
def get_model_configs(
    best_reg_params: Dict = None, best_cls_params: Dict = None
) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of model configurations using PyTorch.
    Here, 'ml' corresponds to a deeper nonlinear model and 'lr' to a simple linear model.
    """
    best_reg_params = best_reg_params or {}
    best_cls_params = best_cls_params or {}
    model_configs = {
        "ml": {
            "reg": TorchRegressor,      # Deep nonlinear regression
            "cls": TorchClassifier,       # Deep nonlinear classification
        },
        "lr": {
            "reg": TorchLinearRegressor,  # Linear regression
            "cls": TorchLinearClassifier, # Linear classification (logistic)
        },
    }
    return model_configs

# ---------------------------------------------------------
# Helper function for parallel processing of a single fold
# ---------------------------------------------------------
def _process_fold(
    fold_num: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data_sorted: pd.DataFrame,
    reg_model_class: Any,
    cls_model_class: Any,
    reg_drop_cols: List[str],
    cls_drop_cols: List[str],
    min_test_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Processes a single fold. Returns test DataFrame with predictions, or None if skipped.
    """
    current_test_start_date = data_sorted.iloc[test_idx]["date"].min()
    if current_test_start_date < min_test_date:
        return None  # Skip this fold

    print(f"[INFO] Processing Fold {fold_num}...")
    train_df = data_sorted.iloc[train_idx].copy()
    test_df = data_sorted.iloc[test_idx].copy()

    # --- Regression ---
    transformer_reg, X_train_reg = create_numeric_transformer(train_df, reg_drop_cols)
    X_train_reg_proc = transformer_reg.fit_transform(X_train_reg)
    y_train_reg = train_df["da"]

    X_test_reg = test_df.drop(columns=reg_drop_cols, errors="ignore")
    X_test_reg = X_test_reg.reindex(columns=X_train_reg.columns, fill_value=0)
    X_test_reg_proc = transformer_reg.transform(X_test_reg)

    y_pred_reg, trained_reg_model = safe_fit_predict(
        reg_model_class, X_train_reg_proc, y_train_reg, X_test_reg_proc, model_type="regression"
    )

    # --- Classification ---
    transformer_cls, X_train_cls = create_numeric_transformer(train_df, cls_drop_cols)
    X_train_cls_proc = transformer_cls.fit_transform(X_train_cls)
    y_train_cls = train_df["da-category"]

    X_test_cls = test_df.drop(columns=cls_drop_cols, errors="ignore")
    X_test_cls = X_test_cls.reindex(columns=X_train_cls.columns, fill_value=0)
    X_test_cls_proc = transformer_cls.transform(X_test_cls)

    y_pred_cls, trained_cls_model = safe_fit_predict(
        cls_model_class, X_train_cls_proc, y_train_cls, X_test_cls_proc, model_type="classification"
    )

    # Store predictions
    test_df["Predicted_da"] = y_pred_reg
    test_df["Predicted_da-category"] = y_pred_cls
    return test_df

# ---------------------------------------------------------
# Model Training Functions (Parallel Evaluation)
# ---------------------------------------------------------
def train_and_evaluate(
    data: pd.DataFrame,
    method="ml",
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
):
    """
    Trains models using TimeSeriesSplit (in parallel) and evaluates performance.
    """
    print(f"\n[INFO] Starting evaluation using method: {method} with PARALLEL TimeSeriesSplit.")
    model_configs = get_model_configs(best_reg_params, best_cls_params)
    if method not in model_configs:
        raise ValueError(f"Method '{method}' not supported")

    reg_model_class = model_configs[method]["reg"]
    cls_model_class = model_configs[method]["cls"]

    data_sorted = data.copy()
    n_splits = CONFIG["N_SPLITS_TS_EVAL"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    min_test_date = pd.Timestamp(CONFIG["MIN_TEST_DATE"])
    n_jobs = CONFIG["N_JOBS_EVAL"]
    print(f"[INFO] Using TimeSeriesSplit with n_splits={n_splits}, n_jobs={n_jobs}. First test fold >= {min_test_date.date()}")

    common_cols = ["date", "site"]
    reg_drop_cols = common_cols + ["da", "da-category"]
    cls_drop_cols = common_cols + ["da", "da-category"]

    # --- Parallel Execution ---
    fold_args = [
        (i + 1, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(tscv.split(data_sorted))
    ]
    print(f"[INFO] Starting parallel processing for {len(fold_args)} folds...")
    results_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_fold)(
            fold_num, train_idx, test_idx, data_sorted, reg_model_class,
            cls_model_class, reg_drop_cols, cls_drop_cols, min_test_date,
        )
        for fold_num, train_idx, test_idx in fold_args
    )

    fold_test_dfs_with_preds = [df for df in results_list if df is not None]
    processed_fold_count = len(fold_test_dfs_with_preds)
    skipped_fold_count = len(fold_args) - processed_fold_count
    print(f"[INFO] Parallel processing complete. Processed {processed_fold_count}, skipped {skipped_fold_count} folds.")

    # --- Aggregation and Final Evaluation ---
    if not fold_test_dfs_with_preds:
        print(f"[ERROR] No valid folds processed. Cannot aggregate results.")
        return {"DA_Level": {}, "da-category": {}}

    print(f"\n[INFO] Aggregating results across {processed_fold_count} folds...")
    final_test_df = pd.concat(fold_test_dfs_with_preds).sort_values(["date", "site"])
    final_test_df = final_test_df.drop_duplicates(subset=["date", "site"], keep="last")

    results = {}
    last_reg_model = reg_model_class
    last_cls_model = cls_model_class

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
                "mae": mean_absolute_error(x["da"], x["Predicted_da"]) if len(x) > 1 else np.nan,
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

    print(f"[INFO] {method.upper()} parallel evaluation complete.")
    return results

def prepare_all_predictions(
    data: pd.DataFrame,
    best_reg_params: Dict = None,
    best_cls_params: Dict = None,
) -> Dict:
    """
    Runs evaluation for the DL method and, if enabled, for a simple linear model.
    """
    predictions = {}
    predictions["ml"] = train_and_evaluate(
        data, method="ml", best_reg_params=best_reg_params, best_cls_params=best_cls_params
    )
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        predictions["lr"] = train_and_evaluate(data, method="lr")
    return predictions

# ---------------------------------------------------------
# Dash App Setup
# ---------------------------------------------------------
def create_dash_app(predictions: Dict, data: pd.DataFrame):
    """
    Creates and configures the Dash application.
    """
    app = dash.Dash(__name__)

    analysis_layout = html.Div(
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
                placeholder="Select site (or All sites)",
                style={"width": "50%", "marginBottom": "15px"},
            ),
            dcc.Graph(id="analysis-graph"),
        ]
    )

    tabs_children = [dcc.Tab(label="Aggregated TS Analysis", children=[analysis_layout])]
    forecast_methods = [{"label": "Deep Learning (ML)", "value": "ml"}]
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        forecast_methods.append({"label": "Linear (LR)", "value": "lr"})
    sites_list = sorted(data["site"].unique().tolist())

    app.layout = html.Div(
        [
            html.H1("Domoic Acid Forecast Dashboard"),
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
            dcc.Tabs(id="tabs", children=tabs_children),
            dcc.Store(id="data-store", data={"sites": sites_list}),
        ]
    )

    @app.callback(
        Output("site-dropdown", "options"), [Input("data-store", "data")]
    )
    def update_site_dropdown(data_store):
        sites = data_store.get("sites", [])
        options = [{"label": "All sites", "value": "All sites"}] + [
            {"label": site, "value": site} for site in sites
        ]
        return options

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
        if forecast_method not in predictions or not predictions[forecast_method]:
            return px.line(title=f"No prediction data available for {forecast_method.upper()}")

        pred_data = predictions[forecast_method]
        if (
            forecast_type not in pred_data
            or not pred_data[forecast_type]
            or "test_df" not in pred_data[forecast_type]
            or pred_data[forecast_type]["test_df"] is None
            or pred_data[forecast_type]["test_df"].empty
        ):
            return px.line(title=f"No data for {forecast_type} using {forecast_method.upper()}")

        results_dict = pred_data[forecast_type]
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

        if selected_site and selected_site != "All sites":
            df_plot_melted = df_plot_melted[df_plot_melted["site"] == selected_site]
            plot_title = f"{y_axis_title} Forecast (Aggregated Test Sets) - {selected_site} ({forecast_method.upper()})"
        else:
            plot_title = f"{y_axis_title} Forecast (Aggregated Test Sets) - All sites ({forecast_method.upper()})"

        if df_plot_melted.empty:
            return px.line(title=f"No data to plot for {selected_site or 'All sites'} ({forecast_method.upper()})")

        df_plot_melted.sort_values("date", inplace=True)
        fig_params = {
            "data_frame": df_plot_melted, "x": "date", "y": "Value",
            "title": plot_title, "category_orders": {"Metric": metric_order},
        }
        fig = px.line(**fig_params, color="site" if selected_site == "All sites" or not selected_site else "Metric",
                      line_dash="Metric" if selected_site == "All sites" or not selected_site else None)

        fig.update_layout(
            yaxis_title=y_axis_title, xaxis_title="Date (Aggregated Test Sets)",
            legend_title_text="Metric/Site", margin=dict(b=80),
            annotations=[{
                "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.15,
                "xanchor": "center", "yanchor": "top", "text": performance_text,
                "showarrow": False, "font": {"size": 12},
            }]
        )
        if forecast_type == "da-category":
            cat_values = sorted(df_plot_melted["Value"].unique(), key=lambda x: int(x))
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

        # Prepare transformers for grid search (dummy for DL)
        common_cols = ["date", "site"]
        reg_drop_cols = common_cols + ["da", "da-category"]
        cls_drop_cols = common_cols + ["da", "da-category"]

        temp_transformer_reg, X_full_reg = create_numeric_transformer(data, reg_drop_cols)
        y_full_reg = data["da"].astype(float)
        temp_transformer_cls, X_full_cls = create_numeric_transformer(data, cls_drop_cols)
        y_full_cls = data["da-category"].astype(int)

        best_dl_reg_params = run_grid_search(
            base_model=TorchRegressor,
            param_grid={}, X=X_full_reg, y=y_full_reg,
            preprocessor=temp_transformer_reg, scoring="r2",
            cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Regressor",
        )
        best_dl_cls_params = run_grid_search(
            base_model=TorchClassifier,
            param_grid={}, X=X_full_cls, y=y_full_cls,
            preprocessor=temp_transformer_cls, scoring="accuracy",
            cv_splits=CONFIG["N_SPLITS_TS_GRIDSEARCH"], model_type="Classifier",
        )

        print("\n[INFO] Preparing predictions using PARALLEL TimeSeriesSplit evaluation...")
        predictions = prepare_all_predictions(
            data, best_reg_params=best_dl_reg_params, best_cls_params=best_dl_cls_params
        )

        print("\n[INFO] Creating Dash application...")
        app = create_dash_app(predictions, data)

        print(f"[INFO] Starting Dash app on http://127.0.0.1:{CONFIG['PORT']}")
        app.run_server(debug=False, port=CONFIG['PORT'])