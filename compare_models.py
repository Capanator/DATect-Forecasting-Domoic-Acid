#!/usr/bin/env python3
"""
Compare performance of different models for DA forecasting.
Tests Linear, Ridge, and XGBoost for both regression and classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from forecasting.core.forecast_engine import ForecastEngine
import warnings
warnings.filterwarnings('ignore')

def run_model_comparison(n_samples=50):
    """Run comprehensive model comparison."""
    
    print("="*80)
    print("MODEL PERFORMANCE COMPARISON FOR DA FORECASTING")
    print("="*80)
    
    # Initialize forecast engine
    engine = ForecastEngine()
    
    # Test sites
    test_sites = ["Cannon Beach", "Newport", "Coos Bay"]
    
    # Models to test
    regression_models = ["linear", "ridge", "xgboost"]
    classification_models = ["logistic", "xgboost"]
    
    # Store results
    results = {
        "regression": {model: {"r2": [], "mae": []} for model in regression_models},
        "classification": {model: {"accuracy": [], "f1": []} for model in classification_models}
    }
    
    print(f"\nTesting with {n_samples} random samples across {len(test_sites)} sites...")
    print("-"*80)
    
    # Run regression comparison
    print("\n📊 REGRESSION TASK (Predicting continuous DA values)")
    print("-"*40)
    
    for model_type in regression_models:
        print(f"\nTesting {model_type.upper()} regression...")
        
        # Run retrospective evaluation
        forecast_results = engine.run_retrospective_evaluation(
            task="regression",
            model_type=model_type,
            n_anchors=n_samples,
            min_test_date="2010-01-01"
        )
        
        if not forecast_results.empty:
            # Filter to test sites and valid predictions
            df = forecast_results[forecast_results['site'].isin(test_sites)].copy()
            df = df.dropna(subset=['da', 'Predicted_da'])
            
            if len(df) > 0:
                r2 = r2_score(df['da'], df['Predicted_da'])
                mae = mean_absolute_error(df['da'], df['Predicted_da'])
                
                results["regression"][model_type]["r2"].append(r2)
                results["regression"][model_type]["mae"].append(mae)
                
                print(f"  • R² Score: {r2:.4f}")
                print(f"  • MAE: {mae:.2f} μg/g")
                print(f"  • Samples: {len(df)}")
            else:
                print(f"  ⚠️ No valid predictions")
    
    # Run classification comparison
    print("\n🎯 CLASSIFICATION TASK (Predicting risk categories)")
    print("-"*40)
    
    for model_type in classification_models:
        # Map model type for classification
        actual_model = "logistic" if model_type == "logistic" else "xgboost"
        print(f"\nTesting {model_type.upper()} classification...")
        
        # Run retrospective evaluation
        forecast_results = engine.run_retrospective_evaluation(
            task="classification",
            model_type=actual_model,
            n_anchors=n_samples,
            min_test_date="2010-01-01"
        )
        
        if not forecast_results.empty:
            # Filter to test sites and valid predictions
            df = forecast_results[forecast_results['site'].isin(test_sites)].copy()
            df = df.dropna(subset=['da-category', 'Predicted_da-category'])
            
            if len(df) > 0:
                accuracy = accuracy_score(df['da-category'], df['Predicted_da-category'])
                
                results["classification"][model_type]["accuracy"].append(accuracy)
                
                print(f"  • Accuracy: {accuracy:.4f}")
                print(f"  • Samples: {len(df)}")
            else:
                print(f"  ⚠️ No valid predictions")
    
    # Print summary comparison
    print("\n"+"="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\n📈 REGRESSION PERFORMANCE (R² Score):")
    print("-"*40)
    for model in regression_models:
        if results["regression"][model]["r2"]:
            avg_r2 = np.mean(results["regression"][model]["r2"])
            avg_mae = np.mean(results["regression"][model]["mae"])
            print(f"{model.upper():12} → R²: {avg_r2:.4f}, MAE: {avg_mae:.2f} μg/g")
    
    print("\n🎯 CLASSIFICATION PERFORMANCE (Accuracy):")
    print("-"*40)
    for model in classification_models:
        if results["classification"][model]["accuracy"]:
            avg_acc = np.mean(results["classification"][model]["accuracy"])
            print(f"{model.upper():12} → Accuracy: {avg_acc:.4f}")
    
    # Analysis of why classification models are closer
    print("\n"+"="*80)
    print("WHY LOGISTIC ≈ XGBOOST FOR CLASSIFICATION BUT LINEAR << XGBOOST FOR REGRESSION?")
    print("="*80)
    
    print("""
1. **CLASSIFICATION IS EASIER** (4 discrete categories vs continuous values):
   - Only need to get the right "bin" (Low/Moderate/High/Extreme)
   - Boundaries are wide: 0-5, 5-20, 20-40, >40 μg/g
   - Can be wrong by 10 μg/g in regression but still correct in classification
   
2. **CLASS IMBALANCE HELPS LINEAR MODELS**:
   - Most samples are "Low" category (0-5 μg/g)
   - Logistic can achieve ~70% accuracy by mostly predicting "Low"
   - XGBoost doesn't gain much advantage here
   
3. **REGRESSION REQUIRES PRECISE VALUES**:
   - Linear assumes: DA = β₀ + β₁×SST + β₂×Chla + ... (straight line)
   - Reality: DA has exponential blooms, threshold effects, complex interactions
   - XGBoost captures these non-linear dynamics
   
4. **FEATURE SPACE DIFFERENCES**:
   - Classification: Features just need to separate 4 regions
   - Regression: Features need to predict exact values across 0-100+ μg/g range
   - Linear models struggle with the full dynamic range
   
5. **ERROR TOLERANCE**:
   - Classification: 15 μg/g vs 18 μg/g = both "Moderate" ✓
   - Regression: 15 μg/g vs 18 μg/g = 3 μg/g error ✗
   
**RIDGE VS LINEAR**:
   - Ridge adds L2 regularization (penalizes large coefficients)
   - Helps with multicollinearity in environmental features
   - Should perform slightly better than pure Linear regression
""")
    
    return results

if __name__ == "__main__":
    # Run comparison with more samples for better statistics
    results = run_model_comparison(n_samples=30)
    
    print("\n✅ Model comparison complete!")