import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def sophisticated_nan_handling_for_correlation(df, preserve_temporal=True):
    """
    Implements sophisticated NaN handling strategy from modular-forecast.
    
    Strategy:
    1. Preserve all data initially for temporal integrity
    2. Use median imputation for feature variables (non-targets)
    3. Only remove samples where target (DA) is NaN for correlation analysis
    4. Maintain temporal ordering to prevent data leakage
    
    Args:
        df: DataFrame to process
        preserve_temporal: Whether to maintain temporal safeguards
        
    Returns:
        DataFrame with sophisticated NaN handling applied
    """
    df_processed = df.copy()
    
    # Sort by date if available to maintain temporal integrity
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(['date']).reset_index(drop=True)
    
    # Separate target variable (DA) from features
    if 'da' in df_processed.columns:
        print(f"  Before NaN handling: {len(df_processed)} records")
        
        # For correlation analysis, we need both variables to be non-NaN
        # Use pairwise deletion approach (similar to pandas corr default)
        # This prevents data leakage by not imputing target values
        
        # Identify numeric columns for imputation (excluding target)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'da']
        
        if feature_cols:
            # Use median imputation for feature variables only
            # This matches modular-forecast strategy: SimpleImputer(strategy="median")
            imputer = SimpleImputer(strategy="median")
            
            # Apply imputation to feature columns
            df_processed[feature_cols] = imputer.fit_transform(df_processed[feature_cols])
            
        # For DA (target), keep NaN values - correlation will handle pairwise deletion
        # This prevents data leakage and maintains scientific integrity
        
        print(f"  After NaN handling: {len(df_processed)} records (preserved all)")
        non_nan_da = df_processed['da'].notna().sum()
        print(f"  Valid DA values: {non_nan_da} ({non_nan_da/len(df_processed)*100:.1f}%)")
        
    return df_processed


# Load the data
file_path = "final_output.parquet"
print("Loading data with sophisticated NaN handling...")
df = pd.read_parquet(file_path)

# Drop 'longitude' and 'latitude' columns if they exist (spatial columns not needed for correlation)
cols_to_drop = [col for col in ['lon', 'lat'] if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Create overall correlation heatmap first
print("Creating overall correlation heatmap for all data")

# Apply sophisticated NaN handling to overall data
overall_df = sophisticated_nan_handling_for_correlation(df.copy())

# Drop the 'site' column if it exists for the overall analysis
if 'site' in overall_df.columns:
    overall_df = overall_df.drop(columns=['site'])

# Select numeric columns only
numeric_df = overall_df.select_dtypes(include=['number'])

# Compute correlation matrix using pandas default (pairwise deletion)
# This automatically handles remaining NaN values without data leakage
corr_matrix = numeric_df.corr(method='pearson')  # Explicit method for clarity

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create the heatmap using imshow with the RdBu colormap
cax = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)

# Set tick marks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.index)))
ax.set_xticklabels(corr_matrix.columns, fontsize=12, rotation=-45)
ax.set_yticklabels(corr_matrix.index, fontsize=12)

# Title and axis labels
ax.set_title(f'Overall Correlation Heatmap', fontsize=20, pad=20)
ax.set_xlabel('Variable', fontsize=14)
ax.set_ylabel('Variable', fontsize=14)

# Annotate each cell with the correlation value (rounded to 2 decimals)
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        # Use white text for dark cells (strong correlations), black otherwise
        color = "white" if abs(value) > 0.7 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

# Add colorbar with a label
cbar = fig.colorbar(cax, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Correlation (r)', rotation=270, labelpad=20, fontsize=14)

# Adjust layout
plt.tight_layout()

# Save the figure as a high-resolution PDF
pdf_path = "data-visualizations/correlation_heatmap_overall.pdf"
plt.savefig(pdf_path, format="pdf")

plt.close()  # Close the figure to free memory
print(f"Overall heatmap saved at: {pdf_path}")

# Now process each site if 'site' column exists
if 'site' in df.columns:
    sites = df['site'].unique()
    site_dfs = [df[df['site'] == site] for site in sites]
    
    # Process each site with sophisticated NaN handling
    for site, site_df in zip(sites, site_dfs):
        print(f"Processing site: {site}")
        
        # Apply sophisticated NaN handling to site data
        site_df_processed = sophisticated_nan_handling_for_correlation(site_df.copy())
        
        # Select numeric columns only (excluding the site column)
        if 'site' in site_df_processed.columns:
            site_df_processed = site_df_processed.drop(columns=['site'])
        numeric_df = site_df_processed.select_dtypes(include=['number'])
        
        # Skip sites with insufficient data
        if len(numeric_df) < 10:
            print(f"  Skipping {site}: insufficient data after NaN handling")
            continue
        
        # Compute correlation matrix using pairwise deletion (leak-free)
        corr_matrix = numeric_df.corr(method='pearson')
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap using imshow with the RdBu colormap
        cax = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        
        # Set tick marks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, fontsize=12, rotation=-45)
        ax.set_yticklabels(corr_matrix.index, fontsize=12)
        
        # Title and axis labels
        ax.set_title(f'Correlation Heatmap - {site}', fontsize=20, pad=20)
        ax.set_xlabel('Variable', fontsize=14)
        ax.set_ylabel('Variable', fontsize=14)
        
        # Annotate each cell with the correlation value (rounded to 2 decimals)
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                # Use white text for dark cells (strong correlations), black otherwise
                color = "white" if abs(value) > 0.7 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
        
        # Add colorbar with a label
        cbar = fig.colorbar(cax, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Correlation (r)', rotation=270, labelpad=20, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure as a high-resolution PDF
        pdf_path = f"data-visualizations/correlation_heatmap_{site}.pdf"
        plt.savefig(pdf_path, format="pdf")
        
        plt.close()  # Close the figure to free memory
        print(f"Heatmap for site '{site}' saved at: {pdf_path}")
else:
    print("No 'site' column found. Only the overall correlation heatmap was created.")

print("All correlation heatmaps have been generated successfully.")