import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Box Plot for Energy vs Rank Category
plt.figure(figsize=(8, 6))
sns.boxplot(x='Rank Category Ordinal', y='Energy', data=data)
plt.title('Box Plot: Energy vs Rank Category')
plt.xlabel('Rank Category')
plt.ylabel('Energy')
plt.show()

# Violin Plot for Energy vs Rank Category
plt.figure(figsize=(8, 6))
sns.violinplot(x='Rank Category Ordinal', y='Energy', data=data, inner="quartile")
plt.title('Violin Plot: Energy vs Rank Category')
plt.xlabel('Rank Category')
plt.ylabel('Energy')
plt.show()

# Scatter Plot with Jitter for Energy vs Rank Category
plt.figure(figsize=(8, 6))
sns.stripplot(x='Rank Category Ordinal', y='Energy', data=data, jitter=True, alpha=0.5)
plt.title('Scatter Plot: Energy vs Rank Category (with Jitter)')
plt.xlabel('Rank Category')
plt.ylabel('Energy')
plt.show()
