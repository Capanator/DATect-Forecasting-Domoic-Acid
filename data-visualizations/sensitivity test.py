import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For model training and permutation importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# For Sobol sensitivity analysis (install SALib if needed)
from SALib.sample import saltelli
from SALib.analyze import sobol

# ----------------------------------------
# Load and Prepare the Dataset from Parquet
# ----------------------------------------
df = pd.read_parquet("final_output.parquet")

# Drop non-essential columns if they exist
cols_to_drop = [col for col in ['lon', 'lat', 'site'] if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Ensure the target variable "DA Levels" exists
target_var = 'da'
if target_var not in numeric_df.columns:
    raise ValueError(f"Column '{target_var}' not found in the dataset.")

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
corr_pdf = "data-visualizations/sensitivity-correlation_sensitivity_analysis.pdf"
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
# 2. Variance-Based Sensitivity (Sobol Indices)
# ----------------------------------------
# Define the problem for SALib using the range (min, max) of each feature.
problem = {
    'num_vars': len(feature_names),
    'names': feature_names,
    'bounds': [[float(X[col].min()), float(X[col].max())] for col in feature_names]
}

# Generate samples using Saltelli's sampling scheme
param_values = saltelli.sample(problem, 128, calc_second_order=True)

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
plt.title("Sobol First Order Sensitivity Indices", fontsize=16)
plt.xlabel("Input Variables", fontsize=14)
plt.ylabel("First Order Sobol Index", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
sobol_pdf = "data-visualizations/sensitivity-sobol_sensitivity_analysis.pdf"
plt.savefig(sobol_pdf, format='pdf')
plt.close()
print(f"Sobol sensitivity PDF saved at: {sobol_pdf}")

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
perm_pdf = "data-visualizations/sensitivity-permutation_importance.pdf"
plt.savefig(perm_pdf, format='pdf')
plt.close()
print(f"Permutation importance PDF saved at: {perm_pdf}")
