import kagglehub
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

peng= kagglehub.dataset_download("srisyra02/penguin-species-dataset")
print(peng)

sns.set_theme(style="whitegrid")

csv_file= os.path.join(peng, "Penguin Species Prediction Dataset.csv")
fg= pd.read_csv(csv_file)

print(fg.shape)

#EXPLORATORY DATA ANALYSIS ON PENGUIN SPECIES DATASET
print("\n--------First 10 rows of the dataset-------")
time.sleep(1)
print(fg.head(10))

print("\n--------Data Info-------")
time.sleep(1)
print(fg.info())

print("\n--------Data Types-------")
time.sleep(1)
print(fg.dtypes)

#comparison operations

print(fg.isnull().sum())  # Check for missing values in the dataset

# Create the shadow feature
fg['flipper_length_missing'] = fg['flipper_length_mm'].isna()
print(fg.groupby('sex')['flipper_length_missing'].value_counts())
#this is check how missing values are distributed across the dataset, and their relationship with the sex of the penguins. 
#It helps to understand if there is any bias in the missing data based on gender.


#DATA CLEANING OPERATIONS
filled = fg['flipper_length_mm'].fillna(0)
print(filled.mean())

#the cleaned dataset (without missing)
clean = fg.dropna(subset=['flipper_length_mm'])
print(clean['flipper_length_mm'].mean())


#SUMMARY STATISTICS FOR THE DATASET
print("\n--------Summary Statistics-------")
time.sleep(1)

print("\n--------Data Description-------")
time.sleep(1)
print(fg.describe())
print(fg.describe(include="object"))  # text columns: count, unique, top, freq

num_cols = fg.select_dtypes(include="number").columns
cat_cols = fg.select_dtypes(exclude="number").columns

#median of numeric columns
print("\n--------Median of Numeric Columns-------")
print(fg[num_cols].median())
time.sleep(1)

#mode of numeric columns
print("\n--------Mode of Numeric Columns-------")
print(fg[num_cols].mode().iloc[0])
time.sleep(1)

#variance, skewness, and kurtosis of numeric columns
print("\n--------Variance, Skewness, and Kurtosis of Numeric Columns-------")

print("------Variance------")
print(fg[num_cols].var())
print("\n------Skewness------")
print(fg[num_cols].skew())    # asymmetry of distribution
print("\n------Kurtosis------")
print(fg[num_cols].kurt())    # tail heaviness

#descriptive statistics for numeric columns
print("\n--------Descriptive Statistics for Numeric Columns-------")
time.sleep(1)
print(fg[num_cols].describe())      # stats only on numeric columns

print("\n--------Correlation Matrix for Numeric Columns-------")
print(fg[num_cols].corr())         # correlation matrix only on numeric column


#Some additional insights from the dataset
print("\n--------Relationship between Culemen Lenght, Island of Penguins-------")
grouped_island= fg.groupby('island')[['culmen_length_mm', 'body_mass_g']].mean()
print(grouped_island)


# --- 5. VISUALIZE ---
# ==========================================
# DATA VISUALIZATIONS FOR PENGUNIN DATASET
# ==========================================

# 1. HISTOGRAMS (Distribution of each feature)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Numeric Features', fontsize=16)

for ax, col in zip(axes.flatten(), num_cols[:4]):
    sns.histplot(data=fg, x=col, kde=True, ax=ax, color='steelblue')
    ax.set_title(f'Histogram & KDE of {col}')

for ax in axes.flatten()[len(num_cols[:4]):]:
    ax.remove()

plt.tight_layout()
plt.show()


# 2. BOX PLOTS (Detecting Outliers & Class Distribution)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Boxplots across Penguin Species', fontsize=14)

for ax, col in zip(axes.flatten(), num_cols[:4]):
    sns.boxplot(data=fg, x='species', y=col, ax=ax, color='steelblue')
    ax.set_title(f'{col} by Species')

for ax in axes.flatten()[len(num_cols[:4]):]:
    ax.remove()

plt.tight_layout()
plt.show()




# Pairplot across all numerical features
sns.pairplot(fg[list(num_cols) + ['species']], hue='species')
plt.suptitle('Pairplot of Penguin Features', y=1.00)
plt.show()


# 4. CORRELATION HEATMAP
plt.figure(figsize=(8, 6))
sns.heatmap(fg[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# 5. COUNT PLOT (Categorical Distribution)
plt.figure(figsize=(6, 4))
sns.countplot(data=fg, x='species', palette='viridis')
plt.title('Sample Count per Penguin Species')
plt.show()

sns.violinplot(data=fg, x='species', y=num_cols[0], palette='Set2')
plt.show()