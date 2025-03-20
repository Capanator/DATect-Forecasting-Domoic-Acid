import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "final_output.parquet"
df = pd.read_parquet(file_path)

# Drop 'longitude' and 'latitude' columns if they exist
cols_to_drop = [col for col in ['longitude', 'latitude'] if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Select numeric columns only
numeric_df = df.select_dtypes(include=['number'])

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
ax.set_title('Correlation Heatmap', fontsize=20, pad=20)
ax.set_xlabel('Variable', fontsize=14)
ax.set_ylabel('Variable', fontsize=14)

# Annotate each cell with the correlation value (rounded to 2 decimals)
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        # Use white text for correlation 1 (or very close to it), black otherwise
        color = "white" if np.isclose(value, 1.0) else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

# Add colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.ax.tick_params(labelsize=12)

# Adjust layout
plt.tight_layout()

# Save the figure as a high-resolution PDF
pdf_path = "data-visualizations/correlation_heatmap.pdf"
plt.savefig(pdf_path, format="pdf")
plt.show()

print(f"High-resolution PDF saved at: {pdf_path}")