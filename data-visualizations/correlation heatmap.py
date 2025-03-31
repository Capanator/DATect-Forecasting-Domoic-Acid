import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = "final_output.parquet"
df = pd.read_parquet(file_path)

# Drop 'longitude' and 'latitude' columns if they exist
cols_to_drop = [col for col in ['longitude', 'latitude'] if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Create overall correlation heatmap first
print("Creating overall correlation heatmap for all data")

# Make a copy of the dataframe
overall_df = df.copy()

# Drop the 'Site' column if it exists for the overall analysis
if 'Site' in overall_df.columns:
    overall_df = overall_df.drop(columns=['Site'])

# Select numeric columns only
numeric_df = overall_df.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

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

# Now process each site if 'Site' column exists
if 'Site' in df.columns:
    sites = df['Site'].unique()
    site_dfs = [df[df['Site'] == site] for site in sites]
    
    # Process each site
    for site, site_df in zip(sites, site_dfs):
        print(f"Processing site: {site}")
        
        # Select numeric columns only (excluding the site column)
        site_df = site_df.drop(columns=['Site'])
        numeric_df = site_df.select_dtypes(include=['number'])
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
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
    print("No 'Site' column found. Only the overall correlation heatmap was created.")

print("All correlation heatmaps have been generated successfully.")