"""
Test script for the unified forecasting pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pipeline import DAForecastPipeline, DAForecastConfig

def create_test_data():
    """Create synthetic test data."""
    np.random.seed(42)
    
    sites = ['Site_A', 'Site_B', 'Site_C']
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='W')
    
    data = []
    for site in sites:
        for date in dates:
            # Create synthetic oceanographic data
            base_da = np.random.lognormal(1.5, 1.0)  # Log-normal distribution for DA
            
            record = {
                'site': site,
                'date': date,
                'da': base_da,
                'chlor_a': np.random.normal(5, 2),
                'sst': np.random.normal(15, 3),
                'par': np.random.normal(40, 10),
                'fluorescence': np.random.normal(0.1, 0.05),
                'k490': np.random.normal(0.08, 0.02),
                'pdo': np.random.normal(0, 1),
                'oni': np.random.normal(0, 1),
                'beuti': np.random.normal(10, 5),
                'streamflow_1': np.random.normal(100, 50),
                'streamflow_2': np.random.normal(200, 100),
            }
            data.append(record)
    
    return pd.DataFrame(data)

def test_pipeline():
    """Test the unified pipeline."""
    print("Creating test data...")
    data = create_test_data()
    
    print(f"Test data shape: {data.shape}")
    print(f"Sites: {data['site'].unique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    config = DAForecastConfig.create_config(
        enable_lag_features=True,
        random_state=42,
        n_jobs=1  # Use single core for testing
    )
    
    pipeline = DAForecastPipeline(config)
    
    # Fit pipeline
    print("Fitting pipeline...")
    pipeline.fit(data)
    
    # Test single forecast
    print("\nTesting single forecast...")
    forecast_date = pd.Timestamp('2015-06-01')
    site = 'Site_A'
    
    try:
        result = pipeline.forecast_single(data, forecast_date, site)
        print(f"Single forecast successful for {site} on {forecast_date.date()}")
        print(f"Predicted DA: {result.get('Predicted_da', 'N/A')}")
        print(f"Predicted Category: {result.get('Predicted_da_category', 'N/A')}")
        print(f"Actual DA: {result.get('Actual_da', 'N/A')}")
    except Exception as e:
        print(f"Error in single forecast: {e}")
    
    # Test retrospective evaluation (small sample)
    print("\nTesting retrospective evaluation...")
    try:
        results_df = pipeline.evaluate_retrospective(
            data, 
            n_anchors_per_site=10,  # Small sample for testing
            min_test_date="2012-01-01"
        )
        print(f"Retrospective evaluation successful: {len(results_df)} forecasts")
        
        # Evaluate performance
        performance = pipeline.evaluate_performance(results_df)
        print(f"Overall performance: {performance}")
        
    except Exception as e:
        print(f"Error in retrospective evaluation: {e}")
    
    print("\nPipeline test completed!")

if __name__ == "__main__":
    test_pipeline()