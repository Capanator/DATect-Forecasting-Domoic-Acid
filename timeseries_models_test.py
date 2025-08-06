#!/usr/bin/env python3
"""
Time Series Forecasting Models for DA Prediction
================================================

Tests specialized time series models including ARIMA, Prophet, NeuralProphet,
and state-of-the-art transformer models.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_absolute_error
import time

# Try importing time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Statsmodels not installed. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Prophet not installed. Install with: pip install prophet")

try:
    from neuralprophet import NeuralProphet
    HAS_NEURALPROPHET = True
except ImportError:
    HAS_NEURALPROPHET = False
    print("NeuralProphet not installed. Install with: pip install neuralprophet")

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    print("pmdarima not installed. Install with: pip install pmdarima")

try:
    from darts import TimeSeries
    from darts.models import (
        TFTModel, NBEATSModel, TCNModel, 
        RNNModel, BlockRNNModel, TransformerModel,
        NHiTSModel, DLinearModel, NLinearModel
    )
    HAS_DARTS = True
except ImportError:
    HAS_DARTS = False
    print("Darts not installed. Install with: pip install darts")

try:
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.tbats import TBATS
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.ensemble import EnsembleForecaster
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    print("sktime not installed. Install with: pip install sktime")

try:
    from gluonts.dataset.common import ListDataset
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.trainer import Trainer
    HAS_GLUONTS = True
except ImportError:
    HAS_GLUONTS = False
    print("GluonTS not installed. Install with: pip install gluonts")

import config


def prepare_timeseries_data():
    """Prepare data for time series models."""
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Focus on one site for pure time series
    site_data = data[data['site'] == 'Kalaloch'].copy()
    site_data = site_data.sort_values('date')
    
    # Create time series features
    site_data = site_data.set_index('date')
    
    # Get DA values and features
    y = site_data['da'].dropna()
    
    # Get exogenous variables (external features)
    exog_features = ['pdo', 'oni', 'beuti', 'streamflow', 'modis-chla', 'modis-sst']
    X = site_data[exog_features].fillna(method='ffill').fillna(method='bfill')
    
    # Split into train/test (80/20)
    split_idx = int(len(y) * 0.8)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    return y_train, y_test, X_train, X_test


def test_arima_models():
    """Test ARIMA family models."""
    if not HAS_STATSMODELS:
        return {}
        
    print("\n" + "="*60)
    print("TESTING ARIMA FAMILY MODELS")
    print("="*60)
    
    y_train, y_test, X_train, X_test = prepare_timeseries_data()
    results = {}
    
    # Test ARIMA
    print("\nTesting ARIMA...")
    try:
        start_time = time.time()
        model = ARIMA(y_train, order=(5,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test))
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['ARIMA'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test SARIMAX (with exogenous variables)
    print("\nTesting SARIMAX with exogenous variables...")
    try:
        start_time = time.time()
        model = SARIMAX(y_train, exog=X_train, order=(2,1,2), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['SARIMAX'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test Exponential Smoothing
    print("\nTesting Exponential Smoothing...")
    try:
        start_time = time.time()
        model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test))
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['ExpSmoothing'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    return results


def test_prophet_models():
    """Test Prophet and NeuralProphet."""
    results = {}
    
    if HAS_PROPHET:
        print("\n" + "="*60)
        print("TESTING PROPHET")
        print("="*60)
        
        y_train, y_test, X_train, X_test = prepare_timeseries_data()
        
        # Prepare data for Prophet
        train_df = pd.DataFrame({
            'ds': y_train.index,
            'y': y_train.values
        })
        
        # Add exogenous variables
        for col in X_train.columns:
            train_df[col] = X_train[col].values
        
        print("\nTesting Facebook Prophet...")
        try:
            start_time = time.time()
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Add regressors
            for col in X_train.columns:
                model.add_regressor(col)
            
            model.fit(train_df)
            
            # Create future dataframe
            future = pd.DataFrame({
                'ds': y_test.index
            })
            for col in X_test.columns:
                future[col] = X_test[col].values
            
            forecast = model.predict(future)
            train_time = time.time() - start_time
            
            y_pred = forecast['yhat'].values
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results['Prophet'] = {'r2': r2, 'mae': mae, 'time': train_time}
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.2f} μg/g")
            print(f"  Train time: {train_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
    
    if HAS_NEURALPROPHET:
        print("\nTesting NeuralProphet...")
        try:
            y_train, y_test, X_train, X_test = prepare_timeseries_data()
            
            # Prepare data
            train_df = pd.DataFrame({
                'ds': y_train.index,
                'y': y_train.values
            })
            
            start_time = time.time()
            
            model = NeuralProphet(
                n_forecasts=len(y_test),
                yearly_seasonality=True,
                weekly_seasonality=False,
                epochs=100,
                batch_size=32,
                learning_rate=0.01
            )
            
            model.fit(train_df, freq='W')
            
            future = model.make_future_dataframe(train_df, periods=len(y_test))
            forecast = model.predict(future)
            train_time = time.time() - start_time
            
            y_pred = forecast['yhat1'].tail(len(y_test)).values
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results['NeuralProphet'] = {'r2': r2, 'mae': mae, 'time': train_time}
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.2f} μg/g")
            print(f"  Train time: {train_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
    
    return results


def test_auto_arima():
    """Test Auto-ARIMA."""
    if not HAS_PMDARIMA:
        return {}
        
    print("\n" + "="*60)
    print("TESTING AUTO-ARIMA")
    print("="*60)
    
    y_train, y_test, X_train, X_test = prepare_timeseries_data()
    results = {}
    
    print("\nTesting Auto-ARIMA with automatic parameter selection...")
    try:
        start_time = time.time()
        
        model = pm.auto_arima(
            y_train,
            X=X_train,
            seasonal=True,
            m=12,  # 12 months seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=10
        )
        
        forecast = model.predict(n_periods=len(y_test), X=X_test)
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['AutoARIMA'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  Best order: {model.order}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    return results


def test_sktime_models():
    """Test sktime models."""
    if not HAS_SKTIME:
        return {}
        
    print("\n" + "="*60)
    print("TESTING SKTIME MODELS")
    print("="*60)
    
    y_train, y_test, X_train, X_test = prepare_timeseries_data()
    results = {}
    
    # Test AutoETS
    print("\nTesting AutoETS...")
    try:
        start_time = time.time()
        model = AutoETS(auto=True, sp=12, n_jobs=-1)
        model.fit(y_train)
        forecast = model.predict(fh=range(1, len(y_test)+1))
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['AutoETS'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test Theta
    print("\nTesting Theta Forecaster...")
    try:
        start_time = time.time()
        model = ThetaForecaster(sp=12)
        model.fit(y_train)
        forecast = model.predict(fh=range(1, len(y_test)+1))
        train_time = time.time() - start_time
        
        r2 = r2_score(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        
        results['Theta'] = {'r2': r2, 'mae': mae, 'time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
    
    return results


def test_multivariate_models():
    """Test multivariate time series models."""
    print("\n" + "="*60)
    print("TESTING MULTIVARIATE TIME SERIES")
    print("="*60)
    
    results = {}
    
    # Load all sites data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Create multivariate time series (multiple sites)
    sites = ['Kalaloch', 'Quinault', 'Copalis']
    mv_data = data[data['site'].isin(sites)].pivot(index='date', columns='site', values='da')
    mv_data = mv_data.dropna()
    
    # Split data
    split_idx = int(len(mv_data) * 0.8)
    train_data = mv_data[:split_idx]
    test_data = mv_data[split_idx:]
    
    if HAS_STATSMODELS:
        print("\nTesting Vector Autoregression (VAR)...")
        try:
            start_time = time.time()
            model = VAR(train_data)
            model_fit = model.fit(maxlags=10)
            forecast = model_fit.forecast(train_data.values[-model_fit.k_ar:], steps=len(test_data))
            train_time = time.time() - start_time
            
            # Evaluate for first site
            y_true = test_data.iloc[:, 0].values
            y_pred = forecast[:, 0]
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            results['VAR'] = {'r2': r2, 'mae': mae, 'time': train_time}
            print(f"  R² Score (Kalaloch): {r2:.4f}")
            print(f"  MAE: {mae:.2f} μg/g")
            print(f"  Train time: {train_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
    
    return results


def main():
    """Run comprehensive time series model testing."""
    print("="*60)
    print("TIME SERIES FORECASTING MODELS COMPARISON")
    print("="*60)
    print("Testing specialized time series models...")
    print("Baseline Random Forest R²: ~0.78")
    print("Current best (XGBoost): R² = 0.84")
    
    all_results = {}
    
    # Test different model families
    if HAS_STATSMODELS:
        arima_results = test_arima_models()
        all_results.update(arima_results)
    
    prophet_results = test_prophet_models()
    all_results.update(prophet_results)
    
    if HAS_PMDARIMA:
        auto_arima_results = test_auto_arima()
        all_results.update(auto_arima_results)
    
    if HAS_SKTIME:
        sktime_results = test_sktime_models()
        all_results.update(sktime_results)
    
    mv_results = test_multivariate_models()
    all_results.update(mv_results)
    
    # Summary
    print("\n" + "="*60)
    print("TIME SERIES MODELS SUMMARY")
    print("="*60)
    print("Baseline RF R²: 0.7818")
    print("Current best (XGBoost): 0.8394")
    print("\nTime Series Model Results:")
    
    for model_name, result in all_results.items():
        if result and 'r2' in result:
            r2 = result['r2']
            improvement = ((r2 - 0.7818) / 0.7818) * 100
            
            if r2 > 0.8394:  # Better than XGBoost
                status = "🏆"
            elif r2 > 0.7818:  # Better than RF
                status = "✅"
            else:
                status = "❌"
                
            print(f"{status} {model_name}: R²={r2:.4f} ({improvement:+.1f}%)")
    
    # Find best time series model
    best_model = None
    best_r2 = 0
    
    for model_name, result in all_results.items():
        if result and result.get('r2', 0) > best_r2:
            best_model = model_name
            best_r2 = result['r2']
    
    if best_r2 > 0.8394:
        print(f"\n🏆 NEW CHAMPION: {best_model} (R²={best_r2:.4f})")
        print("This time series model beats XGBoost!")
    elif best_r2 > 0.7818:
        print(f"\n✅ Best time series model: {best_model} (R²={best_r2:.4f})")
        print("Better than RF but not as good as XGBoost")
    else:
        print("\n❌ No time series model beat Random Forest")
        print("Tree-based models remain superior for this dataset")


if __name__ == "__main__":
    main()