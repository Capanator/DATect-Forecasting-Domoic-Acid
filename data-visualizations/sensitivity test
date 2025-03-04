import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For model training and permutation importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# For Sobol and Morris sensitivity analysis (install SALib if needed)
from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris as morris_analyze

# ----------------------------------------
# Load and Prepare the Dataset
# ----------------------------------------
df = pd.read_csv("data-visualizations/final_output.csv")

# Drop non-essential columns if they exist
cols_to_drop = [col for col in ['lon', 'lat', 'Site'] if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Ensure the target variable "DA Levels" exists
target_var = 'DA Levels'
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
corr_pdf = "data-visualizations/correlation_sensitivity_analysis.pdf"
plt.savefig(corr_pdf, format='pdf', dpi=300)
plt.show()
print(f"Correlation sensitivity PDF saved at: {corr_pdf}")

# ----------------------------------------
# Prepare Data for Model-Based Methods
# ----------------------------------------
# Define features and target for the following tests
X = numeric_df.drop(columns=[target_var])
y = numeric_df[target_var]
feature_names = X.columns.tolist()

# Split the data for training a surrogate model (used in permutation importance)
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
sobol_pdf = "data-visualizations/sobol_sensitivity_analysis.pdf"
plt.savefig(sobol_pdf, format='pdf', dpi=300)
plt.show()
print(f"Sobol sensitivity PDF saved at: {sobol_pdf}")

# ----------------------------------------
# 3. Morris Screening Method
# ----------------------------------------
# Define parameters for the Morris method
num_levels = 4   # number of levels in the grid
num_trajectories = 10  # number of trajectories (can be adjusted)

# Generate samples for the Morris method
morris_samples = morris_sample.sample(problem, N=num_trajectories, num_levels=num_levels, optimal_trajectories=None)

# Evaluate the model on the Morris samples
Y_morris = model_wrapper(morris_samples)

# Analyze the Morris results
morris_results = morris_analyze.analyze(problem, morris_samples, Y_morris, num_levels=num_levels, print_to_console=False)

# Use mu_star: the mean of the absolute elementary effects as sensitivity
mu_star = morris_results['mu_star']

plt.figure(figsize=(10, 6))
plt.bar(feature_names, mu_star)
plt.title("Morris Screening Method: mu* Sensitivity", fontsize=16)
plt.xlabel("Input Variables", fontsize=14)
plt.ylabel("mu* (Mean Absolute Elementary Effects)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
morris_pdf = "data-visualizations/morris_sensitivity_analysis.pdf"
plt.savefig(morris_pdf, format='pdf', dpi=300)
plt.show()
print(f"Morris sensitivity PDF saved at: {morris_pdf}")

# ----------------------------------------
# 4. Permutation Feature Importance
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
perm_pdf = "data-visualizations/permutation_importance.pdf"
plt.savefig(perm_pdf, format='pdf', dpi=300)
plt.show()
print(f"Permutation importance PDF saved at: {perm_pdf}")

# ----------------------------------------
# 5. Regression-Based Sensitivity Analysis
# ----------------------------------------
# Standardize the features (zero mean, unit variance) for regression-based sensitivity
X_standardized = (X_train - X_train.mean()) / X_train.std()
model_std = LinearRegression()
model_std.fit(X_standardized, y_train)

# Get the absolute value of standardized coefficients
std_coefficients = np.abs(model_std.coef_)

plt.figure(figsize=(10, 6))
plt.bar(feature_names, std_coefficients)
plt.title("Regression-Based Sensitivity (Standardized Coefficients)", fontsize=16)
plt.xlabel("Input Variables", fontsize=14)
plt.ylabel("Absolute Standardized Coefficient", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
regression_pdf = "data-visualizations/regression_sensitivity_analysis.pdf"
plt.savefig(regression_pdf, format='pdf', dpi=300)
plt.show()
print(f"Regression-based sensitivity PDF saved at: {regression_pdf}")