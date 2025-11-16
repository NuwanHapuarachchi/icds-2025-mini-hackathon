# Data Quality Assessment Report
## Drug Category Prediction - ICDS 2025 Mini-Hackathon

---

## Executive Summary

This comprehensive data quality assessment analyzed 1,500 training records across 14 features to identify data quality issues and establish cleaning procedures. The dataset achieved an **overall quality score of 93.4/100 (Good)**, with minimal missing values (0.15% of all cells), low outlier prevalence (0.67%), and three extreme anomalies requiring treatment. The primary data quality concern is **class imbalance** (2.86:1 ratio) which will require stratified sampling and class weighting strategies.

**Key Findings:**
- **3 extreme anomalies** detected (Escore=50, Cscore=-10, Impulsive=10)
- **31 missing values** across 10 features (<2% missingness)
- **70 outliers** identified by IQR method (0.67% of data points)
- **0 duplicate records**
- **No data leakage** concerns
- **Class imbalance** detected (46% Hallucinogens, 38% Stimulants, 16% Depressants)

---

## Table of Contents

1. [Missing Values Analysis](#1-missing-values-analysis)
2. [Outlier Detection](#2-outlier-detection)
3. [Anomaly Detection](#3-anomaly-detection)
4. [Data Consistency Checks](#4-data-consistency-checks)
5. [Duplicate Detection](#5-duplicate-detection)
6. [Data Leakage Validation](#6-data-leakage-validation)
7. [Class Balance Analysis](#7-class-balance-analysis)
8. [Data Quality Scorecard](#8-data-quality-scorecard)
9. [Recommendations](#9-recommendations)

---

## 1. Missing Values Analysis

### 1.1 Missingness Summary

| Feature | Missing Count | Missing % | Pattern | Recommended Action |
|---------|--------------|-----------|---------|-------------------|
| Education | 20 | 1.33% | MCAR | Mode imputation or predictive imputation |
| Age | 2 | 0.13% | MCAR | Mode imputation |
| Country | 2 | 0.13% | MCAR | Mode imputation |
| Gender | 1 | 0.07% | MCAR | Mode imputation |
| Nscore | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| Escore | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| Oscore | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| Ascore | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| Cscore | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| Impulsive | 1 | 0.07% | MCAR | KNN imputation (k=5) |
| SS | 0 | 0.00% | - | No action needed |

**Total Missing:** 31 values (0.148% of all 21,000 data cells)

### 1.2 Missingness Pattern

**Assessment:** Missing Completely at Random (MCAR)

**Evidence:**
- Low missing rates across all features (<2%)
- No systematic patterns correlating with drug category
- Random distribution across records
- No clustering of missing values

**Implications:**
- Simple imputation strategies are appropriate
- Minimal impact on model performance expected
- No need for complex missing data mechanisms

**Visualization:** See `visualizations/dqa_missing_values.png`

---

## 2. Outlier Detection

### 2.1 IQR Method Results (1.5 × IQR Rule)

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % | Min Value | Max Value |
|---------|-------------|-------------|---------------|-----------|-----------|-----------|
| **Ascore** | -2.40 | 2.39 | **24** | 1.60% | -3.46 | 3.46 |
| **Nscore** | -2.64 | 2.59 | **16** | 1.07% | -3.46 | 3.27 |
| **Escore** | -2.69 | 2.64 | **12** | 0.80% | -3.27 | **50.00** |
| **Impulsive** | -2.57 | 2.39 | **7** | 0.47% | -2.56 | **10.00** |
| **Oscore** | -2.88 | 2.88 | **6** | 0.40% | -2.86 | 2.90 |
| **Cscore** | -2.77 | 2.87 | **5** | 0.33% | **-10.00** | 3.46 |
| **SS** | -2.46 | 2.70 | **0** | 0.00% | -2.08 | 1.92 |

**Total IQR Outliers:** 70 (0.67% of 10,500 continuous data points)

### 2.2 Z-Score Method Results (|Z| > 3)

| Feature | Outlier Count | Outlier % |
|---------|--------------|-----------|
| **Nscore** | 4 | 0.27% |
| **Ascore** | 4 | 0.27% |
| **Cscore** | 3 | 0.20% |
| **Impulsive** | 1 | 0.07% |
| **Escore** | 1 | 0.07% |
| **Oscore** | 0 | 0.00% |
| **SS** | 0 | 0.00% |

**Total Z-Score Outliers:** 13 (0.12% of continuous data points)

### 2.3 Assessment

**Prevalence:** Low outlier rates (<2%) across all features

**Distribution:**
- Agreeableness (Ascore) has the highest outlier rate at 1.6%
- Sensation Seeking (SS) has zero outliers (well-behaved distribution)
- Most personality traits show <1% outlier prevalence

**Recommendation:**
- **Keep outliers** for tree-based models (they handle outliers naturally)
- **Use RobustScaler** for linear/distance-based models
- **Do not remove** outliers (below 2% threshold)
- Three extreme values (Escore=50, Cscore=-10, Impulsive=10) require special treatment (see Anomaly Detection)

**Visualization:** See `visualizations/dqa_outliers_boxplots.png`

---

## 3. Anomaly Detection

### 3.1 Extreme Anomalies Identified (|Z-Score| > 5)

| Feature | Record ID | Value | Z-Score | Type | Action Required |
|---------|-----------|-------|---------|------|-----------------|
| **Escore** | 1108 | **50.00** | +30.81 | Extreme High | ⚠️ Replace with NaN → Impute |
| **Impulsive** | 24 | **10.00** | +10.11 | Extreme High | ⚠️ Replace with NaN → Impute |
| **Cscore** | 1250 | **-10.00** | -9.69 | Extreme Low | ⚠️ Replace with NaN → Impute |

**Total Anomalies:** 3 (0.02% of continuous data points)

### 3.2 Anomaly Assessment

**Escore = 50.00:**
- **Context:** Extraversion standardized score expected range [-3.27, 3.27]
- **Z-Score:** 30.81 standard deviations above mean
- **Interpretation:** Likely data entry error (entered "50" instead of standardized value)
- **Impact:** Severe distortion in Escore distribution (skewness=19.54, kurtosis=601.72)
- **Action:** Replace with NaN, apply KNN imputation

**Cscore = -10.00:**
- **Context:** Conscientiousness standardized score expected range [-3.46, 3.46]
- **Z-Score:** 9.69 standard deviations below mean
- **Interpretation:** Likely data entry error (entered "-10" instead of standardized value)
- **Impact:** Moderate distortion in Cscore distribution (kurtosis=5.40)
- **Action:** Replace with NaN, apply KNN imputation

**Impulsive = 10.00:**
- **Context:** Impulsiveness standardized score expected range [-2.56, 2.90]
- **Z-Score:** 10.11 standard deviations above mean
- **Interpretation:** Likely raw score entered instead of standardized value
- **Impact:** Moderate distortion in Impulsive distribution (kurtosis=6.36)
- **Action:** Replace with NaN, apply KNN imputation

### 3.3 Treatment Strategy

**Step 1: Flag Anomalies**
```python
df.loc[df['Escore'] > 10, 'Escore'] = np.nan
df.loc[df['Cscore'] < -5, 'Cscore'] = np.nan
df.loc[df['Impulsive'] > 5, 'Impulsive'] = np.nan
```

**Step 2: Apply KNN Imputation**
```python
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df[continuous_features] = knn_imputer.fit_transform(df[continuous_features])
```

**Visualization:** See `reports/dqa/dqa_anomalies.csv` for detailed records

---

## 4. Data Consistency Checks

### 4.1 Categorical Value Ranges

| Feature | Unique Count | Expected Count | Min Value | Max Value | Status |
|---------|-------------|---------------|-----------|-----------|--------|
| **Age** | 6 | 6 | -0.95197 | 2.59171 | ✓ Consistent |
| **Gender** | 2 | 2 | -0.48246 | 0.48246 | ✓ Consistent |
| **Education** | 9 | 9 | -2.43591 | 1.98437 | ✓ Consistent |
| **Country** | 7 | 7 | -0.57009 | 0.96082 | ✓ Consistent |
| **Ethnicity** | 7 | 7 | -1.10702 | 1.90725 | ✓ Consistent |

**Assessment:** All categorical features have expected number of unique values

**Encoding Validation:**
- All categorical features use numerical encoding (standardized values)
- No unexpected categories or encoding errors
- Value ranges match attribute documentation

### 4.2 Standardization Range Violations

| Feature | Min Value | Max Value | Expected Range | Violation Type |
|---------|-----------|-----------|----------------|----------------|
| **Escore** | -3.27 | **50.00** | [-4.84, 4.89] | ⚠️ Upper bound (30× SD) |
| **Cscore** | **-10.00** | 3.46 | [-3.10, 3.09] | ⚠️ Lower bound (10× SD) |
| **Impulsive** | -2.56 | **10.00** | [-2.96, 2.97] | ⚠️ Upper bound (10× SD) |

**Assessment:**
- Three features violate expected standardization range (±5 SD)
- All violations are due to extreme anomalies identified above
- Other 4 continuous features are within normal range (±3 SD)

**Action:** Address anomalies as outlined in Section 3

---

## 5. Duplicate Detection

### 5.1 Duplicate Assessment Results

**Exact Duplicate Rows:** 0  
**Feature Duplicates (excluding ID):** 0  
**Status:** ✓ No duplicates detected

### 5.2 Validation Details

**Checks Performed:**
1. Exact row duplicates (all columns)
2. Feature duplicates (excluding ID column)
3. Partial duplicates (same demographics + personality)

**Conclusion:** No duplicate records exist in the dataset

---

## 6. Data Leakage Validation

### 6.1 Target Variable Check

**Status:** ✓ Target variable 'drug_category' present in training data

**Validation:**
- Target is properly separated from features
- No derived features from target
- No suspicious feature names containing "drug"

### 6.2 Temporal Leakage Check

**Status:** ✓ No temporal features detected

**Validation:**
- All features represent measurements at time of assessment
- No future information in features
- No time-series data that could leak future events

### 6.3 Data Splitting Recommendation

**Stratified Sampling Required:**
- Use `stratify=y` in train_test_split
- Maintain class distribution across splits
- Prevents information leakage across folds
- Ensures representative validation sets

**Example:**
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 7. Class Balance Analysis

### 7.1 Class Distribution

| Drug Category | Count | Percentage |
|--------------|-------|------------|
| **Hallucinogens** | 691 | 46.07% |
| **Stimulants** | 567 | 37.80% |
| **Depressants** | 242 | 16.13% |

**Total Records:** 1,500

### 7.2 Imbalance Assessment

**Imbalance Ratio (max/min):** 2.86

**Analysis:**
- **Majority Class:** Hallucinogens (46.07%)
- **Minority Class:** Depressants (16.13%)
- **Ratio:** 2.86:1 (Hallucinogens to Depressants)

**Severity:** ⚠️ Moderate to Significant Imbalance

**Expected Impact:**
- Models may be biased toward predicting Hallucinogens
- Depressants (minority class) may be under-predicted
- Accuracy metric alone will be misleading
- Need balanced evaluation metrics (macro F1-score)

### 7.3 Recommended Strategies

**1. Evaluation Metrics:**
- **Primary:** Macro F1-Score (equal weight to all classes)
- **Secondary:** Weighted F1-Score, per-class precision/recall
- **Monitor:** Confusion matrix for per-class performance
- **Avoid:** Overall accuracy as primary metric

**2. Sampling Strategies:**
- **Stratified K-Fold CV:** Maintain class distribution in folds
- **Stratified Train/Val Split:** Preserve proportions
- **Consider SMOTE:** Synthetic Minority Over-sampling for Depressants

**3. Model-Level Solutions:**
- **Class Weights:** Inverse frequency weighting
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
  ```
- **Threshold Tuning:** Adjust decision thresholds per class
- **Ensemble Methods:** Combine multiple models with different balancing

**Visualization:** See `visualizations/dqa_class_balance.png`

---

## 8. Data Quality Scorecard

### 8.1 Overall Quality Assessment

| Dimension | Score (out of 100) | Status |
|-----------|-------------------|---------|
| **Missing Values** | 99.85 | Excellent |
| **Duplicates** | 100.00 | Excellent |
| **Outliers** | 99.33 | Good |
| **Anomalies** | 99.97 | Good |
| **Consistency** | 57.14 | Fair |
| **OVERALL** | **93.39** | **Good** |

### 8.2 Scorecard Interpretation

**Strengths:**
- ✓ Minimal missing values (99.85% completeness)
- ✓ Zero duplicate records (100% uniqueness)
- ✓ Low outlier prevalence (99.33% within normal range)
- ✓ Very few anomalies (99.97% data points are valid)

**Areas for Improvement:**
- ⚠️ Consistency score affected by 3 extreme anomalies
- ⚠️ Range violations in Escore, Cscore, Impulsive
- ⚠️ Class imbalance requires attention

**Overall Assessment:** **Good quality (93.4/100)**

The dataset is in good condition for modeling with targeted cleaning needed for anomalies and missing values. No major quality issues that would prevent model development.

**Visualization:** See `visualizations/dqa_scorecard.png`

---

## 9. Recommendations

### 9.1 Priority Actions

**PRIORITY 1: Handle Extreme Anomalies** ⚠️ CRITICAL
- **Issue:** 3 extreme values detected (|Z| > 5)
- **Features Affected:** Escore (1 value), Cscore (1 value), Impulsive (1 value)
- **Action:** Replace with NaN and apply KNN imputation
- **Expected Impact:** Restore normal distributions, reduce kurtosis/skewness
- **Timeline:** Immediate (before any modeling)

**PRIORITY 2: Missing Value Treatment** ⚠️ HIGH
- **Issue:** 31 missing values across 10 features
- **Categorical Features:** Mode imputation (simple, effective for <2% missing)
- **Continuous Features:** KNN imputation (k=5, preserves feature relationships)
- **Expected Impact:** 100% data completeness, minimal distribution shifts
- **Timeline:** Immediate (as part of cleaning pipeline)

**PRIORITY 3: Outlier Management** ℹ️ MEDIUM
- **Issue:** 70 outliers detected (0.67% of data points)
- **For Tree-Based Models:** Keep outliers (Random Forest, XGBoost handle naturally)
- **For Linear Models:** Use RobustScaler or Winsorization
- **Expected Impact:** Improved model robustness
- **Timeline:** Model-specific (during preprocessing)

**PRIORITY 4: Class Imbalance Handling** ⚠️ HIGH
- **Issue:** Imbalance ratio of 2.86:1 (Hallucinogens to Depressants)
- **Stratification:** Use stratified sampling in all splits
- **Class Weights:** Apply inverse frequency weights in models
- **SMOTE (Optional):** Consider for minority class augmentation
- **Expected Impact:** Balanced per-class performance
- **Timeline:** Model training phase

**PRIORITY 5: Data Validation** ✓ LOW
- **Issue:** None (no duplicates or data leakage)
- **Action:** Final validation after cleaning
- **Timeline:** Post-cleaning validation step

### 9.2 Data Cleaning Pipeline

**Recommended Workflow:**

```
1. Load Raw Data
   ↓
2. Flag Anomalies (Escore>10, Cscore<-5, Impulsive>5) → NaN
   ↓
3. Apply KNN Imputation (continuous features)
   ↓
4. Apply Mode Imputation (categorical features)
   ↓
5. Validate: Check distributions, ranges, missing values
   ↓
6. Save Cleaned Data
   ↓
7. Proceed to Feature Engineering
```

**Expected Outcomes:**
- 100% data completeness
- Normal distributions restored
- All values within expected ranges
- Ready for feature engineering

### 9.3 Next Steps

**Immediate Actions:**
- [ ] Execute data cleaning pipeline (see `scripts/data_cleaning.py`)
- [ ] Validate cleaned data quality
- [ ] Document all transformations

**Feature Engineering Phase:**
- [ ] Create interaction features (Oscore×SS, Cscore×Impulsive)
- [ ] Generate polynomial features for non-linear relationships
- [ ] Encode categorical variables
- [ ] Create domain-specific risk profiles

**Model Development Phase:**
- [ ] Implement stratified train/validation splits
- [ ] Apply class weights to models
- [ ] Use macro F1-score as primary evaluation metric
- [ ] Conduct cross-validation experiments

---

## Summary Statistics

### Continuous Features Overview

| Feature | Mean | Std | Min | Max | Skewness | Kurtosis | Missing % |
|---------|------|-----|-----|-----|----------|----------|-----------|
| Nscore | 0.01 | 1.00 | -3.46 | 3.27 | -0.02 | -0.01 | 0.07% |
| **Escore** | 0.02 | 1.62 | -3.27 | **50.00** | **19.54** | **601.72** | 0.07% |
| Oscore | 0.00 | 0.99 | -2.86 | 2.90 | 0.04 | -0.14 | 0.07% |
| Ascore | -0.02 | 0.99 | -3.46 | 3.46 | 0.02 | 0.07 | 0.07% |
| **Cscore** | -0.00 | 1.03 | **-10.00** | 3.46 | **-0.58** | **5.40** | 0.07% |
| **Impulsive** | 0.01 | 0.99 | -2.56 | **10.00** | **0.77** | **6.36** | 0.07% |
| SS | 0.00 | 0.97 | -2.08 | 1.92 | -0.05 | -0.45 | 0.00% |

**Note:** Bold values indicate features affected by extreme anomalies

---

## Deliverables

**Reports Generated (8 files):**
1. `data/dqa_missing_values.csv` - Missing value summary
2. `data/dqa_outliers_iqr.csv` - IQR outlier detection results
3. `data/dqa_outliers_zscore.csv` - Z-score outlier detection results
4. `data/dqa_anomalies.csv` - Extreme anomaly records
5. `data/dqa_consistency.csv` - Categorical consistency checks
6. `data/dqa_class_balance.csv` - Target variable distribution
7. `data/dqa_summary_statistics.csv` - Comprehensive statistics
8. `data/dqa_scorecard.csv` - Quality scorecard scores

**Visualizations Generated (4 files):**
1. `dqa_missing_values.png` - Missing value patterns
2. `dqa_outliers_boxplots.png` - Outlier visualization
3. `dqa_class_balance.png` - Class distribution charts
4. `dqa_scorecard.png` - Quality scorecard visualization

**Scripts:**
- `scripts/data_quality_assessment.py` - Full assessment pipeline

---

**Report Generated:** 2025-11-16  
**Dataset Version:** Training set (1,500 records)  
**Overall Quality Score:** 93.4/100 (Good)  
**Status:** Ready for data cleaning and feature engineering

**Next Action:** Execute data cleaning pipeline to address anomalies and missing values
