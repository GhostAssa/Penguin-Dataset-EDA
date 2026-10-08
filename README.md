# Penguins: Missing Data Mechanisms & Exploratory Data Analysis

A practical exploration of the **Penguins dataset** investigating missing data mechanisms (MCAR vs. MAR), the dangers of naive imputation, and conditional associations across penguin species and island habitats.

---

## 📌 Project Overview

When handling missing data, default operations like zero-filling (`fillna(0)`) or row dropping (`dropna()`) are frequently applied without auditing their structural impact on feature distributions. This project demonstrates how missing data carrying underlying signal (**Missing At Random - MAR**) alters foundational summary statistics, and how cross-tabulation can prove conditional dependencies before choosing an imputation strategy.

### Key Objectives
* Evaluate structural shifts in continuous metrics (`body_mass_g`, `flipper_length_mm`, `culmen_length_mm`, `culmen_depth_mm`) across different missing value treatments.
* Construct **Shadow Matrices** (binary missingness flags) to cross-tabulate missing values against categorical attributes (`species`, `island`).
* Identify conditional missingness patterns across penguin species and locations.
* Apply an Object-Oriented layout model (`fig, ax`) to build crisp data visualizations using `matplotlib` and `seaborn`.

---

## 🔬 Experimental Findings

### 1. The Danger of Zero Imputation (`fillna(0)`)
Filling continuous physical features with `0` introduces non-existent physical outliers (e.g., a "0g penguin") into the distribution:

* **Mean Shift:** In this dataset, zero-imputation pulled the global average body mass down by **~2.01 units**.
* **Variance Distortion:** Artificially inflating the total count with zero values distorts feature variance and skews downstream machine learning models.

### 2. Listwise Deletion Bias (`dropna()`)
Calculating summary metrics with `dropna()` alters the average relative to the observed sample default (`skipna=True`). Because physical traits vary drastically between species (e.g., *Gentoo* vs. *Adelie* vs. *Chinstrap*), listwise deletion selectively removes records from specific subgroups, shifting the overall population average.

### 3. Proving Missing At Random (MAR) via Shadow Matrices
By constructing a binary missingness indicator (flagging missing entries as true/false) and cross-tabulating against `species` and `island`, the missing entries were proved to be non-uniformly distributed:
* Cross-tabulation revealed that missing records were concentrated within specific species and island combinations.
* Group comparison confirmed a clear association between categorical attributes and missingness—formally confirming a **Missing At Random (MAR)** mechanism.

---

## 📊 Summary Comparison Matrix

| Method | Pandas Syntax | Practical Operation | Effect on Average | Impact on Data Spread |
| :--- | :--- | :--- | :--- | :--- |
| **Observed Only** | `df['body_mass_g'].mean()` | Evaluates observed records exclusively | Unbiased for observed sample | Preserves observed spread |
| **Zero-Fill** | `df['body_mass_g'].fillna(0).mean()` | Evaluates total count with zero-value additions | Decreases average (~2.01 lower) | Artificially inflates spread |
| **Listwise Deletion** | `df.dropna().mean()` | Removes entire rows with missing data | Shifts average due to subgroup removal | Alters group variance proportions |
| **Group-wise Imputation** | `df.groupby(['species', 'island']).transform(...)` | Replaces missing entries with subgroup median | Preserves subgroup distribution | Minimizes artificial distortion |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`

---

## 📁 Repository Structure

```text
├── data/
│   └── palmer_penguins.csv       # Raw Palmer Penguins dataset
├── notebooks/
│   └── eda_missing_data.ipynb    # Main Analysis Notebook (Shadow matrix, distributions)
├── scripts/
│   ├── missing_analysis.py       # Helper functions for cross-tabulation & shadow matrix generation
│   └── visualization_theme.py    # Custom Seaborn/Matplotlib styling configurations
├── README.md                     # Project documentation
└── requirements.txt              # Environment dependencies
