#!/usr/bin/env python3
"""
Dashboard Functionality Test
============================

Test the dashboard components without actually launching the servers.
Validates that dashboards can be initialized and core functionality works.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

def test_realtime_dashboard():
    """Test real-time dashboard initialization and core functions."""
    print("🔄 Testing Real-time Dashboard...")
    
    try:
        from forecasting.dashboard.realtime import RealtimeDashboard
        from forecasting.core.forecast_engine import ForecastEngine
        
        # Check if data exists
        if not pd.io.common.file_exists('final_output.parquet'):
            print("❌ Data file missing - cannot test dashboard")
            return False
            
        # Initialize dashboard
        dashboard = RealtimeDashboard('final_output.parquet')
        print("✅ RealtimeDashboard initialized successfully")
        
        # Test forecast engine
        engine = ForecastEngine('final_output.parquet')
        
        # Load data for testing
        data = pd.read_parquet('final_output.parquet')
        recent_date = data['date'].max() - timedelta(days=30)
        test_site = data['site'].iloc[0]
        
        # Test single forecast generation
        result = engine.generate_single_forecast(
            'final_output.parquet',
            recent_date,
            test_site,
            'regression',
            'xgboost'
        )
        
        if result:
            print(f"✅ Test forecast generated for {test_site} on {recent_date.date()}")
            print(f"   Predicted DA: {result.get('predicted_da', 'N/A'):.2f} μg/g")
        else:
            print("⚠️  Test forecast returned None (insufficient data)")
            
        return True
        
    except Exception as e:
        print(f"❌ Real-time dashboard test failed: {e}")
        return False


def test_retrospective_dashboard():
    """Test retrospective dashboard with mock data."""
    print("🔄 Testing Retrospective Dashboard...")
    
    try:
        from forecasting.dashboard.retrospective import RetrospectiveDashboard
        
        # Create mock results data
        mock_results = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='W'),
            'site': ['TestSite'] * 50,
            'da': [5.0 + i * 0.1 for i in range(50)],
            'Predicted_da': [5.2 + i * 0.1 for i in range(50)],
            'anchor_date': pd.date_range('2022-12-01', periods=50, freq='W')
        })
        
        # Initialize dashboard
        dashboard = RetrospectiveDashboard(mock_results)
        print("✅ RetrospectiveDashboard initialized successfully")
        
        # Test metrics calculation (simulate what the callback would do)
        from sklearn.metrics import r2_score, mean_absolute_error
        
        valid_data = mock_results.dropna(subset=['da', 'Predicted_da'])
        if not valid_data.empty:
            r2 = r2_score(valid_data['da'], valid_data['Predicted_da'])
            mae = mean_absolute_error(valid_data['da'], valid_data['Predicted_da'])
            print(f"✅ Mock metrics calculated: R²={r2:.3f}, MAE={mae:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrospective dashboard test failed: {e}")
        return False


def main():
    """Run dashboard functionality tests."""
    print("📈 Testing Dashboard Functionality")
    print("=" * 40)
    
    success = True
    
    # Test real-time dashboard
    success &= test_realtime_dashboard()
    print()
    
    # Test retrospective dashboard  
    success &= test_retrospective_dashboard()
    print()
    
    if success:
        print("✅ All dashboard tests passed!")
        return 0
    else:
        print("❌ Some dashboard tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())