import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the dataset
file_path = "data-visualizations/final_output.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Define normalization to shift red appearance aggressively at 50 ppm
norm = mcolors.Normalize(vmin=df["DA Levels"].min(), vmax=50)

# Create the bubble plot
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    df["Date"], df["lat"], s=df["DA Levels"] * 5, c=df["DA Levels"], 
    cmap=plt.cm.coolwarm, norm=norm, alpha=0.6, edgecolors="black", linewidth=0.5, marker="x"
)

# Add colorbar with customized labels
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("DA Levels (ppm)")
cbar.set_ticks([0, 10, 20, 30, 40, 50])
cbar.ax.set_yticklabels(["0", "10", "20", "30", "40", "≥50"])  # Adjust labels

# Set labels and title
ax.set_xlabel("Date")
ax.set_ylabel("Latitude")
ax.set_title("Time Series of DA Levels by Latitude (Bubble Size Represents DA Levels)")
ax.tick_params(axis='x', rotation=45)
ax.grid(True, linestyle="--", alpha=0.5)

# Export as PDF
pdf_filename = "data-visualizations/DA_Levels_Bubble_Plot.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
plt.show()

print(f"Plot saved as {pdf_filename}")
