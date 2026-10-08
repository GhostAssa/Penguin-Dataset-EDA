# Penguins: Missing Data Mechanisms & Exploratory Data Analysis

A statistical exploration of the **Palmer Penguins dataset** investigating missing data mechanisms (MCAR vs. MAR), the mathematical dangers of naive imputation, and conditional associations across penguin species and island habitats.

---

## 📌 Project Overview

When handling missing data, default operations like zero-filling (`fillna(0)`) or row dropping (`dropna()`) are frequently applied without auditing their structural impact on feature distributions. This project demonstrates how missing data carrying underlying signal (**Missing At Random - MAR**) alters foundational summary statistics, and how statistical tests (Chi-Square test of independence) and cross-tabulation can prove conditional dependencies before choosing an imputation strategy.

### Key Objectives
* Evaluate structural shifts in continuous metrics (`body_mass_g`, `flipper_length_mm`, `culmen_length_mm`, `culmen_depth_mm`) across different missing value treatments.
* Construct **Shadow Matrices** (binary missingness flags) to cross-tabulate missing values against categorical attributes (`species`, `island`).
* Prove conditional missingness using Chi-Square tests of independence ($\chi^2$).
* Apply an Object-Oriented layout model (`fig, ax`) to build crisp data visualizations using `matplotlib` and `seaborn`.

---

## 🔬 Experimental Findings

### 1. The Mathematical Danger of Zero Imputation (`fillna(0)`)
Filling continuous physical features with `0` introduces non-existent physical outliers (e.g., a "0g penguin") into the distribution:

$$\bar{X}_{\text{zero}} = \frac{\sum_{i=1}^{N_{\text{obs}}} X_i + 0 \times N_{\text{missing}}}{N_{\text{obs}} + N_{\text{missing}}}$$

* **Mean Shift:** In this dataset, zero-imputation pulled the global average body mass down by **~2.01 units**.
* **Variance Explosion:** Artificially inflating the denominator while keeping the numerator constant distorts feature variance and skews downstream machine learning models.

### 2. Listwise Deletion Bias (`dropna()`)
Calculating summary metrics with `dropna()` alters the mean relative to the observed sample default (`skipna=True`). Because physical traits vary drastically between species (e.g., *Gentoo* vs. *Adelie* vs. *Chinstrap*), listwise deletion selectively removes records from specific subgroups, shifting the global population mean.

### 3. Proving Missing At Random (MAR) via Shadow Matrices
By constructing a binary missingness indicator ($M_i = 1$ if $X_i$ is `NaN`, else $0$) and cross-tabulating against `species` and `island`, the missing entries were proved to be non-uniformly distributed:
* Cross-tabulation revealed that missing records were concentrated within specific species and island combinations.
* Running a Chi-Square test of independence ($\chi^2$) confirmed a statistically significant association between categorical attributes and missingness—formally confirming a **Missing At Random (MAR)** mechanism.

---

## 📊 Summary Comparison Matrix

| Method | Pandas Syntax | Mathematical Operation | Effect on Mean ($\bar{X}$) | Variance ($\sigma^2$) Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Observed Only** | `df['body_mass_g'].mean()` | Evaluates $N_{\text{obs}}$ exclusively | Unbiased for observed sample | Preserves observed spread |
| **Zero-Fill** | `df['body_mass_g'].fillna(0).mean()` | Evaluates $N_{\text{total}}$ with $0$-value additions | Decreases mean ($\approx -2.01$) | Artificially inflates variance |
| **Listwise Deletion** | `df.dropna().mean()` | Removes entire rows with $\ge 1$ `NaN` | Shifts mean due to MAR subgroup removal | Alters group variance proportions |
| **Group-wise Imputation** | `df.groupby(['species', 'island']).transform(...)` | Replaces `NaN` with subgroup median | Preserves subgroup distribution | Minimizes artificial distortion |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing:** `pandas`, `numpy`
* **Statistical Testing:** `scipy.stats` (`chi2_contingency`)
* **Data Visualization:** `seaborn`, `matplotlib`

---

## 📁 Repository Structure

```text
├── data/
│   └── pal
