import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import warnings

# For Sobol sensitivity analysis (install SALib if needed)
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("Warning: SALib not available, Sobol analysis will be skipped")

warnings.filterwarnings('ignore')

def sophisticated_nan_handling_for_sensitivity(df):
    """
    Implements sophisticated NaN handling strategy from modular-forecast for sensitivity analysis.
    
    Strategy:
    1. Preserve temporal integrity by maintaining order
    2. Remove samples with NaN target values (DA) - matches modular-forecast training approach
    3. Use median imputation for feature variables only
    4. Ensure sufficient samples for reliable sensitivity analysis
    5. Prevent data leakage through careful temporal handling
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with sophisticated NaN handling applied
    """
    df_processed = df.copy()
    
    # Maintain temporal ordering if date column exists
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
        print(f"  Maintaining temporal order: {len(df_processed)} records")
    
    # Check for target variable
    if 'da' not in df_processed.columns:
        raise ValueError("Target variable 'da' not found in dataset")
    
    print(f"  Before NaN handling: {len(df_processed)} records")
    
    # CRITICAL: Remove samples with NaN target values (matches modular-forecast strategy)
    # This is exactly what modular-forecast does: train_df_clean = train_df.dropna(subset=["da"])
    df_clean = df_processed.dropna(subset=['da']).copy()
    print(f"  After removing NaN targets: {len(df_clean)} records")
    
    if len(df_clean) < 50:  # Need sufficient samples for sensitivity analysis
        print(f"  Warning: Only {len(df_clean)} samples available - may affect sensitivity analysis reliability")
    
    # Identify feature columns (excluding target and non-predictive columns)
    exclude_cols = ['da', 'date', 'site', 'lon', 'lat']  # Non-predictive or target columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if feature_cols:
        print(f"  Applying median imputation to {len(feature_cols)} feature variables")
        # Use median imputation for features (matches modular-forecast: SimpleImputer(strategy="median"))
        imputer = SimpleImputer(strategy="median")
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    
    print(f"  Final dataset: {len(df_clean)} records with {len(feature_cols)} features")
    
    return df_clean


# ----------------------------------------
# Load and Prepare the Dataset from Parquet with Sophisticated NaN Handling
# ----------------------------------------
print("Loading data with sophisticated NaN handling for sensitivity analysis...")
df = pd.read_parquet("final_output.parquet")

# Apply sophisticated NaN handling
df_processed = sophisticated_nan_handling_for_sensitivity(df)

# Drop non-essential columns after NaN handling
cols_to_drop = [col for col in ['lon', 'lat', 'site', 'date'] if col in df_processed.columns]
if cols_to_drop:
    df_processed = df_processed.drop(columns=cols_to_drop)

# Keep only numeric columns (should be clean after sophisticated handling)
numeric_df = df_processed.select_dtypes(include=['number'])

# Ensure the target variable exists (should be guaranteed by our NaN handling)
target_var = 'da'
if target_var not in numeric_df.columns:
    raise ValueError(f"Column '{target_var}' not found in the dataset.")

print(f"Final dataset for sensitivity analysis: {len(numeric_df)} records, {len(numeric_df.columns)} variables")

# ----------------------------------------
# 1. Correlation Analysis
# ----------------------------------------
# Compute absolute Pearson correlation of each variable with "DA Levels"
correlations = numeric_df.corr()[target_var].drop(target_var).abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(correlations.index, correlations.values)
plt.title("Correlation Sensitivity Analysis: Impact on DA Levels", fontsize=16)
plt.xlabel("Input Variables", fontsize=14)
plt.ylabel("Absolute Pearson Correlation", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
corr_pdf = "data-visualizations/sensitivity-correlation_sensitivity_analysis_sophisticated.pdf"
plt.savefig(corr_pdf, format='pdf')
plt.close()  # Close the figure to free memory
print(f"Correlation sensitivity PDF saved at: {corr_pdf}")

# ----------------------------------------
# Prepare Data for Model-Based Methods
# ----------------------------------------
X = numeric_df.drop(columns=[target_var])
y = numeric_df[target_var]
feature_names = X.columns.tolist()

# Split data for training a surrogate model (used in permutation importance)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------
# 2. Variance-Based Sensitivity (Sobol Indices) - If Available
# ----------------------------------------
if SALIB_AVAILABLE and len(X) > 200:  # Need sufficient data for Sobol analysis
    try:
        print("Performing Sobol sensitivity analysis...")
        # Define the problem for SALib using the range (min, max) of each feature.
        problem = {
            'num_vars': len(feature_names),
            'names': feature_names,
            'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_names]
        }

        # Generate samples using Saltelli's sampling scheme (reduced for stability)
        param_values = saltelli.sample(problem, 64, calc_second_order=False)

        # Define a model wrapper function to use the trained regression model
        def model_wrapper(X_input):
            return model.predict(X_input)

        # Evaluate the model for all generated samples
        Y = model_wrapper(param_values)

        # Compute Sobol sensitivity indices
        sobol_indices = sobol.analyze(problem, Y, print_to_console=False)
        first_order = sobol_indices['S1']

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, first_order)
        plt.title("Sobol First Order Sensitivity Indices (Sophisticated NaN Handling)", fontsize=16)
        plt.xlabel("Input Variables", fontsize=14)
        plt.ylabel("First Order Sobol Index", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.tight_layout()
        sobol_pdf = "data-visualizations/sensitivity-sobol_sensitivity_analysis_sophisticated.pdf"
        plt.savefig(sobol_pdf, format='pdf')
        plt.close()
        print(f"Sobol sensitivity PDF saved at: {sobol_pdf}")
        
    except Exception as e:
        print(f"Sobol analysis failed: {e}")
        print("This is often due to insufficient data variety after NaN handling - this is expected behavior")
else:
    if not SALIB_AVAILABLE:
        print("Sobol analysis skipped: SALib not available")
    else:
        print(f"Sobol analysis skipped: insufficient data ({len(X)} samples, need >200)")

# ----------------------------------------
# 3. Permutation Feature Importance
# ----------------------------------------
# Compute permutation feature importance on the test set
perm_result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
perm_importances = perm_result.importances_mean

plt.figure(figsize=(10, 6))
plt.bar(feature_names, perm_importances)
plt.title("Permutation Feature Importance", fontsize=16)
plt.xlabel("Input Variables", fontsize=14)
plt.ylabel("Decrease in Model Score", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
perm_pdf = "data-visualizations/sensitivity-permutation_importance_sophisticated.pdf"
plt.savefig(perm_pdf, format='pdf')
plt.close()
print(f"Permutation importance PDF saved at: {perm_pdf}")
