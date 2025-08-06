#!/usr/bin/env python3
"""
Deep Learning and Advanced Model Testing for DA Forecasting
===========================================================

Tests LSTM, 1D-CNN, Transformers, and advanced ensemble methods.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. Install with: pip install tensorflow-macos tensorflow-metal")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")

import config


def create_lstm_model(input_shape):
    """Create LSTM model for time series regression."""
    if not HAS_TF:
        return None
    
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_cnn_model(input_shape):
    """Create 1D CNN model for regression."""
    if not HAS_TF:
        return None
    
    model = models.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_transformer_model(input_shape):
    """Create simple Transformer model for regression."""
    if not HAS_TF:
        return None
    
    inputs = keras.Input(shape=input_shape)
    
    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    positions = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + positions
    
    # Transformer block
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(x, x)
    x = layers.Dropout(0.1)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed forward
    ffn_output = layers.Dense(128, activation="relu")(x)
    ffn_output = layers.Dense(input_shape[1])(ffn_output)
    x = layers.Dropout(0.1)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


class PyTorchLSTM(nn.Module):
    """PyTorch LSTM model."""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def test_deep_learning_models():
    """Test various deep learning models."""
    print("="*60)
    print("DEEP LEARNING MODEL TESTING")
    print("="*60)
    
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Get subset for testing
    sites = data['site'].unique()[:3]
    test_data = data[data['site'].isin(sites)].copy()
    
    # Prepare features
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'site', 'da', 'pn', 'da-category']]
    
    test_data_clean = test_data.dropna(subset=['da'])
    X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
    y = test_data_clean.loc[X.index, 'da']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    results = {}
    
    # Test TensorFlow models
    if HAS_TF:
        print("\n" + "-"*40)
        print("TENSORFLOW MODELS")
        print("-"*40)
        
        # Reshape for LSTM/CNN (samples, timesteps, features)
        X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Test LSTM
        print("\nTesting LSTM...")
        lstm_model = create_lstm_model((1, X_train.shape[1]))
        
        early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        start_time = time.time()
        lstm_model.fit(X_train_3d, y_train, epochs=50, batch_size=32, 
                      validation_split=0.2, verbose=0, callbacks=[early_stop])
        train_time = time.time() - start_time
        
        y_pred = lstm_model.predict(X_test_3d, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results['LSTM'] = {'r2': r2, 'mae': mae, 'train_time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
        
        # Test 1D-CNN
        print("\nTesting 1D-CNN...")
        cnn_model = create_cnn_model((1, X_train.shape[1]))
        
        start_time = time.time()
        cnn_model.fit(X_train_3d, y_train, epochs=50, batch_size=32,
                     validation_split=0.2, verbose=0, callbacks=[early_stop])
        train_time = time.time() - start_time
        
        y_pred = cnn_model.predict(X_test_3d, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results['1D-CNN'] = {'r2': r2, 'mae': mae, 'train_time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
        
        # Test Transformer
        print("\nTesting Transformer...")
        transformer_model = create_transformer_model((1, X_train.shape[1]))
        
        start_time = time.time()
        transformer_model.fit(X_train_3d, y_train, epochs=50, batch_size=32,
                            validation_split=0.2, verbose=0, callbacks=[early_stop])
        train_time = time.time() - start_time
        
        y_pred = transformer_model.predict(X_test_3d, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results['Transformer'] = {'r2': r2, 'mae': mae, 'train_time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    
    # Test PyTorch models
    if HAS_TORCH:
        print("\n" + "-"*40)
        print("PYTORCH MODELS")
        print("-"*40)
        
        # Convert to PyTorch tensors
        X_train_torch = torch.FloatTensor(X_train).unsqueeze(1)
        y_train_torch = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_test_torch = torch.FloatTensor(X_test).unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Test PyTorch LSTM
        print("\nTesting PyTorch LSTM...")
        model = PyTorchLSTM(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        train_time = time.time() - start_time
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_torch).numpy().flatten()
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results['PyTorch LSTM'] = {'r2': r2, 'mae': mae, 'train_time': train_time}
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f} μg/g")
        print(f"  Train time: {train_time:.2f}s")
    
    return results


def test_ensemble_stacking():
    """Test advanced ensemble stacking methods."""
    from sklearn.ensemble import StackingRegressor, VotingRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    import xgboost as xgb
    import lightgbm as lgb
    
    print("\n" + "="*60)
    print("ADVANCED ENSEMBLE TESTING")
    print("="*60)
    
    # Load data
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    
    # Get subset
    sites = data['site'].unique()[:3]
    test_data = data[data['site'].isin(sites)].copy()
    
    # Prepare features
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'site', 'da', 'pn', 'da-category']]
    
    test_data_clean = test_data.dropna(subset=['da'])
    X = test_data_clean[feature_cols].fillna(test_data_clean[feature_cols].median())
    y = test_data_clean.loc[X.index, 'da']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=8, random_state=42, verbose=-1))
    ]
    
    # Test Stacking
    print("\nTesting Stacking Ensemble...")
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=xgb.XGBRegressor(n_estimators=50, random_state=42),
        cv=5
    )
    
    start_time = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = stacking.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} μg/g")
    print(f"  Train time: {train_time:.2f}s")
    
    # Test Voting
    print("\nTesting Voting Ensemble...")
    voting = VotingRegressor(estimators=base_models)
    
    start_time = time.time()
    voting.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = voting.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} μg/g")
    print(f"  Train time: {train_time:.2f}s")
    
    return {'stacking': {'r2': r2}, 'voting': {'r2': r2}}


def main():
    """Run comprehensive deep learning tests."""
    
    # Test deep learning models
    if HAS_TF or HAS_TORCH:
        dl_results = test_deep_learning_models()
    else:
        print("No deep learning libraries available. Skipping DL tests.")
        dl_results = {}
    
    # Test ensemble methods
    ensemble_results = test_ensemble_stacking()
    
    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON WITH RANDOM FOREST BASELINE")
    print("="*60)
    print("Random Forest baseline R²: ~0.78")
    print("\nModels exceeding baseline:")
    
    all_results = {**dl_results, **ensemble_results}
    best_model = None
    best_score = 0.78  # RF baseline
    
    for name, result in all_results.items():
        if result.get('r2', 0) > 0.78:
            improvement = ((result['r2'] - 0.78) / 0.78) * 100
            print(f"  ✅ {name}: R²={result['r2']:.4f} (+{improvement:.1f}%)")
            if result['r2'] > best_score:
                best_model = name
                best_score = result['r2']
    
    if best_model:
        print(f"\n🏆 BEST MODEL: {best_model} (R²={best_score:.4f})")
    else:
        print("\nNo deep learning model exceeded Random Forest baseline.")


if __name__ == "__main__":
    main()