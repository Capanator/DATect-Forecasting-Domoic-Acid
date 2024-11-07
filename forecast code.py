import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load and prepare data with feature engineering
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(['Site', 'Date'], inplace=True)

    # Spatial Feature Engineering
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['spatial_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    data = pd.get_dummies(data, columns=['spatial_cluster'], prefix='cluster')

    # Temporal Feature Engineering
    day_of_year = data['Date'].dt.dayofyear
    data['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)

    # Add Month as Categorical Variable
    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')

    # Include Year as a Feature
    data['Year'] = data['Date'].dt.year

    # Lag Features
    for lag in [1, 2, 3, 7, 14]:
        data[f'DA_Levels_lag_{lag}'] = data.groupby('Site')['DA_Levels'].shift(lag)

    # Interaction Terms
    for cluster in range(5):
        data[f'cluster_{cluster}_sin_day_of_year'] = data[f'cluster_{cluster}'] * data['sin_day_of_year']
        data[f'cluster_{cluster}_cos_day_of_year'] = data[f'cluster_{cluster}'] * data['cos_day_of_year']

    # Normalization
    cols_to_normalize = data.columns.difference(['Date', 'Site', 'DA_Levels'] + list(data.filter(regex='cluster_').columns))
    scaler = MinMaxScaler()
    data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])

    # Categorize DA Levels
    def categorize_da_levels(da_levels):
        if da_levels <= 5:
            return 0
        elif 5 < da_levels <= 20:
            return 1
        elif 20 < da_levels <= 40:
            return 2
        else:
            return 3

    data['DA_Category'] = data['DA_Levels'].apply(categorize_da_levels)

    # Imputation
    numerical_cols = data.columns.drop(['Date', 'Site'])
    imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    return data

# Train models and make predictions for DA_Level and DA_Category
def train_and_predict(data):
    # DA_Level prediction
    data_reg = data.drop(['DA_Category'], axis=1)
    train_set_reg, test_set_reg = train_test_split(data_reg, test_size=0.2, random_state=42)
    X_train_reg = train_set_reg.drop(['DA_Levels', 'Date', 'Site'], axis=1)
    y_train_reg = train_set_reg['DA_Levels']
    X_test_reg = test_set_reg.drop(['DA_Levels', 'Date', 'Site'], axis=1)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = rf_regressor.predict(X_test_reg)
    test_set_reg['Predicted_DA_Levels'] = y_pred_reg
    test_set_reg.sort_values(['Date'], inplace=True)
    
    # Calculate R² and RMSE per site and overall
    overall_r2_reg = r2_score(test_set_reg['DA_Levels'], test_set_reg['Predicted_DA_Levels'])
    overall_rmse_reg = np.sqrt(mean_squared_error(test_set_reg['DA_Levels'], test_set_reg['Predicted_DA_Levels']))
    site_stats_reg = test_set_reg.groupby('Site').apply(
        lambda x: pd.Series({
            'r2': r2_score(x['DA_Levels'], x['Predicted_DA_Levels']),
            'rmse': np.sqrt(mean_squared_error(x['DA_Levels'], x['Predicted_DA_Levels']))
        })
    )

    # DA_Category prediction
    data_cls = data.drop(['DA_Levels'], axis=1)
    train_set_cls, test_set_cls = train_test_split(data_cls, test_size=0.2, random_state=42)
    X_train_cls = train_set_cls.drop(['DA_Category', 'Date', 'Site'], axis=1)
    y_train_cls = train_set_cls['DA_Category']
    X_test_cls = test_set_cls.drop(['DA_Category', 'Date', 'Site'], axis=1)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_cls, y_train_cls)
    
    y_pred_cls = rf_classifier.predict(X_test_cls)
    test_set_cls['Predicted_DA_Category'] = y_pred_cls
    test_set_cls.sort_values(['Date'], inplace=True)
    
    # Calculate accuracy per site and overall
    overall_accuracy_cls = accuracy_score(test_set_cls['DA_Category'], test_set_cls['Predicted_DA_Category'])
    site_stats_cls = test_set_cls.groupby('Site').apply(
        lambda x: accuracy_score(x['DA_Category'], x['Predicted_DA_Category'])
    )

    return {
        "DA_Level": (test_set_reg, site_stats_reg, overall_r2_reg, overall_rmse_reg),
        "DA_Category": (test_set_cls, site_stats_cls, overall_accuracy_cls)
    }

# Initialize Dash app
app = dash.Dash(__name__)

# Load data and train models
file_path = 'final_output.csv'  # Update this path to your dataset location
data = load_and_prepare_data(file_path)
predictions = train_and_predict(data)

# Define layout with dynamic options
app.layout = html.Div([
    html.H1("Domoic Acid Analysis Dashboard"),
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
        options=[{'label': 'All Sites', 'value': 'All Sites'}] + 
                 [{'label': site, 'value': site} for site in data['Site'].unique()],
        value='All Sites',
        style={'width': '50%'}
    ),
    dcc.Graph(id='analysis-graph')
])

# Callback to update the graph and display site-specific stats
@app.callback(
    Output('analysis-graph', 'figure'),
    [Input('forecast-type-dropdown', 'value'),
     Input('site-dropdown', 'value')]
)
def update_graph(selected_forecast_type, selected_site):
    if selected_forecast_type == 'DA_Level':
        df_plot, site_stats, overall_r2, overall_rmse = predictions['DA_Level']
        y_axis_title = 'Domoic Acid Levels'
        y_columns = ['DA_Levels', 'Predicted_DA_Levels']
        
        # Display overall stats if "All Sites" is selected
        if selected_site == 'All Sites':
            performance_text = f"Overall R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f}"
        elif selected_site in site_stats.index:
            site_r2 = site_stats.loc[selected_site, 'r2']
            site_rmse = site_stats.loc[selected_site, 'rmse']
            performance_text = f"R² = {site_r2:.2f}, RMSE = {site_rmse:.2f}"
    else:
        df_plot, site_stats, overall_accuracy = predictions['DA_Category']
        y_axis_title = 'Domoic Acid Category'
        y_columns = ['DA_Category', 'Predicted_DA_Category']
        
        # Display overall stats if "All Sites" is selected
        if selected_site == 'All Sites':
            performance_text = f"Overall Accuracy = {overall_accuracy:.2f}"
        elif selected_site in site_stats.index:
            site_accuracy = site_stats.loc[selected_site]
            performance_text = f"Accuracy = {site_accuracy:.2f}"
    
    # Filter data based on selected site
    if selected_site != 'All Sites':
        df_plot = df_plot[df_plot['Site'] == selected_site]
    
    # Plot results
    fig = px.line(
        df_plot,
        x='Date',
        y=y_columns,
        color='Site' if selected_site == 'All Sites' else None,
        title=f"{y_axis_title} Forecast - {selected_site}"
    )
    fig.update_layout(
        yaxis_title=y_axis_title,
        xaxis_title='Date',
        annotations=[
            dict(
                xref='paper', yref='paper', x=0.5, y=-0.2,
                xanchor='center', yanchor='top',
                text=performance_text,
                showarrow=False
            )
        ]
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
