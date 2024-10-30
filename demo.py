import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data = pd.read_excel("Pistachio_16_Features_Dataset.xlsx")
data.to_csv("Pistachio_16_features.csv", index=False)

# Basic Statistics Calculation
stats_summary = {}
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:  # Skip non-numeric columns
        stats_summary[column] = {
            "mean": data[column].mean(),
            "median": data[column].median(),
            "mode": data[column].mode()[0] if not data[column].mode().empty else None
        }

# Display the statistics
print("Basic Statistics for Each Column:")
for column, stats in stats_summary.items():
    print(f"{column}: Mean = {stats['mean']}, Median = {stats['median']}, Mode = {stats['mode']}")

# Visualizations
# 1. Histogram for each numeric feature
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# 2. Boxplot for each numeric feature
plt.figure(figsize=(15, 10))
sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Features")
plt.show()

# 3. Pairplot to show relationships among features (if dataset is not too large)
if data.shape[0] <= 10000:  
    sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
    plt.show()