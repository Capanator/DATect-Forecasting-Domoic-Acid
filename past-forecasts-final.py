import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import random
from typing import Dict

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

# Configuration
CONFIG = {
    "ENABLE_LAG_FEATURES": True,
    "ENABLE_LINEAR_LOGISTIC": True,
    "DATA_FILE": "final_output.parquet",
    "PORT": 8072,
    "NUM_RANDOM_ANCHORS_PER_SITE_EVAL": 100,
    "N_SPLITS_TS_GRIDSEARCH": 2,
    "MIN_TEST_DATE": "2008-01-01",
    "N_JOBS_EVAL": -1,
    "RANDOM_SEED": 42,
}
random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

# Parameter Grids
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

# Data Processing
def create_numeric_transformer(df, drop_cols):
    X = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    transformer = ColumnTransformer(
        [("num", numeric_pipeline, numeric_cols)],
        remainder="passthrough", verbose_feature_names_out=False
    )
    transformer.set_output(transform="pandas")
    return transformer, X

def add_lag_features(df, group_col, value_col, lags):
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df

def load_and_prepare_data(file_path):
    print(f"[INFO] Loading {file_path}")
    data = pd.read_parquet(file_path, engine="pyarrow")
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["site", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Temporal features
    day_of_year = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)

    # Lag features
    if CONFIG["ENABLE_LAG_FEATURES"]:
        print("[INFO] Creating lag features")
        data = add_lag_features(data, "site", "da", [1, 2, 3])

    # Target categorization
    data["da-category"] = pd.cut(
        data["da"],
        bins=[-float("inf"), 5, 20, 40, float("inf")],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(pd.Int64Dtype())

    # Clean data
    lag_cols = [f"da_lag_{lag}" for lag in [1,2,3]] if CONFIG["ENABLE_LAG_FEATURES"] else []
    initial_rows = len(data)
    data.dropna(subset=lag_cols + ["da", "da-category"], inplace=True)
    print(f"[INFO] Dropped {initial_rows - len(data)} rows")
    return data.reset_index(drop=True)

# Model Functions
def safe_fit_predict(model, X_train, y_train, X_test, model_type):
    if model_type == "classification":
        # y_train is expected to be a pandas Series.
        # .nunique() correctly counts distinct non-NA values.
        if y_train.nunique() < 2:
            print(f"[WARN] Not enough classes in y_train for classification model {type(model).__name__}. "
                  f"Found {y_train.nunique()} unique value(s): {y_train.unique().tolist()}. "
                  f"Required at least 2. X_test has {X_test.shape[0]} sample(s). Returning NaN predictions.")
            # Return an array of NaNs with the same shape as expected predictions
            return np.full(X_test.shape[0], np.nan)
        y_train_processed = y_train.astype(int)
    else: # regression
        y_train_processed = y_train.astype(float)

    model.fit(X_train, y_train_processed)
    return model.predict(X_test)

def run_grid_search(base_model, param_grid, X, y, preprocessor, scoring, cv_splits, model_type):
    print(f"\n[INFO] GridSearchCV for {model_type}")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring=scoring,
        cv=TimeSeriesSplit(n_splits=cv_splits), n_jobs=-1, verbose=0, error_score="raise"
    )
    grid_search.fit(X, y)
    print(f"[INFO] Best {scoring}: {grid_search.best_score_:.4f}\nParams: {grid_search.best_params_}")
    return {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}

def get_model_configs(best_reg_params=None, best_cls_params=None):
    return {
        "ml": {
            "reg": RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **(best_reg_params or {})),
            "cls": RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1, **(best_cls_params or {})),
        },
        "lr": {
            "reg": LinearRegression(n_jobs=1),
            "cls": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
        },
    }

def process_anchor_forecast(anchor_info, full_data, reg_model_base, cls_model_base, reg_drop_cols, cls_drop_cols, min_target_date):
    site, anchor_date = anchor_info
    site_data = full_data[full_data["site"] == site]
    train_df = site_data[site_data["date"] <= anchor_date]
    test_df_single_row = site_data[site_data["date"] > anchor_date].iloc[:1].copy()
    
    if train_df.empty or test_df_single_row.empty or test_df_single_row["date"].min() < min_target_date:
        return None

    # Regression processing
    transformer_reg, X_train_reg = create_numeric_transformer(train_df, reg_drop_cols)
    X_test_reg = test_df_single_row.drop(columns=reg_drop_cols, errors="ignore")
    X_test_reg = X_test_reg.reindex(columns=X_train_reg.columns, fill_value=0)
    reg_model = clone(reg_model_base)
    y_pred_reg = safe_fit_predict(
        reg_model, transformer_reg.fit_transform(X_train_reg), train_df["da"],
        transformer_reg.transform(X_test_reg), "regression"
    )[0]

    # Classification processing
    transformer_cls, X_train_cls = create_numeric_transformer(train_df, cls_drop_cols)
    X_test_cls = test_df_single_row.drop(columns=cls_drop_cols, errors="ignore")
    X_test_cls = X_test_cls.reindex(columns=X_train_cls.columns, fill_value=0)
    cls_model = clone(cls_model_base)
    y_pred_cls = safe_fit_predict(
        cls_model, transformer_cls.fit_transform(X_train_cls), train_df["da-category"],
        transformer_cls.transform(X_test_cls), "classification"
    )[0]

    test_df_single_row["Predicted_da"] = y_pred_reg
    test_df_single_row["Predicted_da-category"] = y_pred_cls
    return test_df_single_row

def train_and_evaluate(data, method="ml", best_reg_params=None, best_cls_params=None):
    print(f"\n[INFO] Evaluating {method} with parallel anchors")
    configs = get_model_configs(best_reg_params, best_cls_params)
    reg_model_base, cls_model_base = configs[method]["reg"], configs[method]["cls"]
    min_target_date = pd.Timestamp(CONFIG["MIN_TEST_DATE"])
    
    # Generate anchor points
    anchor_infos = []
    for site in data["site"].unique():
        dates = data[data["site"] == site]["date"].sort_values().unique()
        if len(dates) > 1:
            anchors = random.sample(list(dates[:-1]), min(len(dates)-1, CONFIG["NUM_RANDOM_ANCHORS_PER_SITE_EVAL"]))
            anchor_infos.extend([(site, pd.Timestamp(d)) for d in anchors])
    
    if not anchor_infos:
        print("[ERROR] No anchor points")
        return {"DA_Level": {}, "da-category": {}}
        
    # Parallel processing
    common_cols = ["date", "site"]
    results = Parallel(n_jobs=CONFIG["N_JOBS_EVAL"], verbose=1)(
        delayed(process_anchor_forecast)(
            ai, data, reg_model_base, cls_model_base, 
            common_cols + ["da", "da-category"], common_cols + ["da", "da-category"], min_target_date
        ) for ai in tqdm(anchor_infos, desc="Processing Anchors")
    )
    
    # Process results
    forecast_dfs = [df for df in results if df is not None]
    final_test_df = pd.concat(forecast_dfs).sort_values(["date", "site"]).drop_duplicates(["date", "site"])
    print(f"[INFO] Processed {len(forecast_dfs)} forecasts")

    # Evaluation metrics
    results_dict = {}
    # Regression evaluation
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

    results_dict["DA_Level"] = {
        "test_df": final_test_df, 
        "site_stats": site_stats_reg.set_index("site"),
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "model": reg_model_base, 
    }

    # Classification evaluation
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

    results_dict["da-category"] = {
        "test_df": final_test_df, 
        "site_stats": site_stats_cls.set_index("site")["accuracy"],
        "overall_accuracy": overall_accuracy,
        "model": cls_model_base, 
    }

    return results_dict

def prepare_all_predictions(data, best_reg_params=None, best_cls_params=None):
    predictions = {"ml": train_and_evaluate(data, "ml", best_reg_params, best_cls_params)}
    if CONFIG["ENABLE_LINEAR_LOGISTIC"]:
        predictions["lr"] = train_and_evaluate(data, "lr")
    return predictions

# Dash App (Original version with optimizations preserved)
def create_dash_app(predictions: Dict, data: pd.DataFrame):
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

# Main Execution
if __name__ == "__main__":
    print("[INFO] Loading data")
    data = load_and_prepare_data(CONFIG["DATA_FILE"])
    if data.empty:
        print("[ERROR] Data empty")
    else:
        # Grid search setup
        common_cols = ["date", "site"]
        reg_drop_cols = common_cols + ["da", "da-category"]
        cls_drop_cols = common_cols + ["da", "da-category"]
        
        trans_reg, X_reg = create_numeric_transformer(data, reg_drop_cols)
        trans_cls, X_cls = create_numeric_transformer(data, cls_drop_cols)
        
        best_rf_reg_params = run_grid_search(
            RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
            PARAM_GRID_REG, X_reg, data["da"].astype(float), trans_reg, "r2",
            CONFIG["N_SPLITS_TS_GRIDSEARCH"], "Regressor"
        )
        best_rf_cls_params = run_grid_search(
            RandomForestClassifier(random_state=CONFIG["RANDOM_SEED"], n_jobs=1),
            PARAM_GRID_CLS, X_cls, data["da-category"].astype(int), trans_cls, "accuracy",
            CONFIG["N_SPLITS_TS_GRIDSEARCH"], "Classifier"
        )

        # Prepare predictions
        predictions = prepare_all_predictions(data, best_rf_reg_params, best_rf_cls_params)
        
        # Start Dash app
        app = create_dash_app(predictions, data)
        print(f"[INFO] Starting server at http://127.0.0.1:{CONFIG['PORT']}")
        app.run_server(debug=False, port=CONFIG['PORT'])