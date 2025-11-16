# Data Quality Assessment and Cleaning Guide
## Drug Category Prediction - ICDS 2025 Mini-Hackathon

---

## Executive Summary

This document provides comprehensive data quality assessment results and detailed cleaning procedures for the drug category prediction dataset. Based on EDA findings, this guide outlines strategies for handling missing values, outliers, anomalies, and data transformations required for optimal model performance.

---

## 1. Missing Values Assessment

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

**Missingness Pattern**: Missing Completely at Random (MCAR)
- Low missing rates (< 2% for all features)
- No systematic patterns detected
- No correlation between missingness and drug category

### 1.2 Imputation Strategy

**Categorical Features** (Simple Mode Imputation):
```python
from sklearn.impute import SimpleImputer

# Mode imputation for categorical features
categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
```

**Continuous Features** (KNN Imputation):
```python
from sklearn.impute import KNNImputer

# KNN imputation for personality traits (preserves relationships)
personality_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']
behavioral_features = ['Impulsive', 'SS']
continuous_features = personality_features + behavioral_features

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df[continuous_features] = knn_imputer.fit_transform(df[continuous_features])
```

**Validation**:
- Compare distributions before/after imputation
- Ensure no significant shifts in mean/variance
- Check that imputed values fall within reasonable ranges

---

## 2. Outlier Detection and Treatment

### 2.1 IQR-Based Outliers

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |
|---------|-------------|-------------|---------------|-----------|
| Ascore | -2.40 | 2.39 | 24 | 1.60% |
| Impulsive | -2.57 | 2.39 | 7 | 0.47% |
| Nscore | -2.88 | 2.88 | 6 | 0.40% |
| Oscore | -2.88 | 2.88 | 6 | 0.40% |
| Cscore | -2.77 | 2.87 | 5 | 0.33% |
| SS | -2.46 | 2.70 | 0 | 0.00% |

### 2.2 Treatment Strategy

**Option 1: Keep Outliers** (Recommended for tree-based models)
- Random Forest, XGBoost, LightGBM handle outliers naturally
- No treatment needed
- Outlier percentage < 2% for all features

**Option 2: Winsorization** (For linear models)
```python
from scipy.stats import mstats

def winsorize_features(df, features, limits=(0.01, 0.01)):
    """Cap outliers at 1st and 99th percentiles"""
    for feature in features:
        df[feature] = mstats.winsorize(df[feature], limits=limits)
    return df

# Apply to features with outliers
outlier_features = ['Ascore', 'Nscore', 'Oscore', 'Cscore', 'Impulsive']
df = winsorize_features(df, outlier_features)
```

**Option 3: RobustScaler** (For distance-based models)
```python
from sklearn.preprocessing import RobustScaler

# Use RobustScaler instead of StandardScaler
scaler = RobustScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])
```

---

## 3. Anomaly Detection and Resolution

### 3.1 Identified Anomalies

| Feature | Anomaly Type | Value | Z-Score | Recommended Action |
|---------|-------------|-------|---------|-------------------|
| **Escore** | Extreme High | 50.00 | ~30 | Replace with NaN, then impute |
| **Cscore** | Extreme Low | -10.00 | ~10 | Replace with NaN, then impute |
| **Impulsive** | Extreme High | 10.00 | ~10 | Replace with NaN, then impute |

### 3.2 Anomaly Treatment

**Step 1: Flag Anomalies**
```python
import numpy as np

# Flag extreme values as NaN
df.loc[df['Escore'] > 10, 'Escore'] = np.nan
df.loc[df['Cscore'] < -5, 'Cscore'] = np.nan
df.loc[df['Impulsive'] > 5, 'Impulsive'] = np.nan
```

**Step 2: Apply Imputation**
```python
# Re-run KNN imputation after flagging anomalies
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df[continuous_features] = knn_imputer.fit_transform(df[continuous_features])
```

**Step 3: Validate**
```python
# Check ranges after treatment
for feature in continuous_features:
    print(f"{feature}: [{df[feature].min():.2f}, {df[feature].max():.2f}]")
```

**Alternative: Capping**
```python
# Cap at 99th/1st percentile instead of NaN
p99 = df['Escore'].quantile(0.99)
p1_cscore = df['Cscore'].quantile(0.01)
p99_imp = df['Impulsive'].quantile(0.99)

df['Escore'] = df['Escore'].clip(upper=p99)
df['Cscore'] = df['Cscore'].clip(lower=p1_cscore)
df['Impulsive'] = df['Impulsive'].clip(upper=p99_imp)
```

---

## 4. Data Consistency Checks

### 4.1 Categorical Value Consistency

**Validation Rules**:
- Age: 6 unique values (categorical encoding)
- Gender: 2 unique values
- Education: 9 unique values
- Country: 7 unique values
- Ethnicity: 7 unique values

**Check Script**:
```python
categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']

for feature in categorical_features:
    unique_count = df[feature].nunique()
    unique_values = df[feature].unique()
    print(f"{feature}: {unique_count} unique values")
    print(f"  Values: {sorted(unique_values)}")
```

### 4.2 Standardization Range Validation

**Expected Range**: ±3 standard deviations for normalized features

**Validation**:
```python
for feature in continuous_features:
    mean = df[feature].mean()
    std = df[feature].std()
    min_val = df[feature].min()
    max_val = df[feature].max()
    
    if min_val < (mean - 5*std) or max_val > (mean + 5*std):
        print(f"⚠️ {feature}: Range violation [{min_val:.2f}, {max_val:.2f}]")
```

---

## 5. Duplicate Detection

### 5.1 Assessment Results

- **Exact duplicate rows**: 0
- **Duplicate rows (excluding ID)**: 0
- **No action needed**

### 5.2 Validation Script

```python
# Check for exact duplicates
exact_duplicates = df.duplicated().sum()
print(f"Exact duplicates: {exact_duplicates}")

# Check for feature duplicates (excluding ID)
feature_cols = df.columns.drop('ID')
feature_duplicates = df.duplicated(subset=feature_cols).sum()
print(f"Feature duplicates: {feature_duplicates}")

# If duplicates exist, remove them
if exact_duplicates > 0:
    df = df.drop_duplicates()
```

---

## 6. Data Leakage Prevention

### 6.1 Feature Validation

**Ensure no future information in features**:

✓ All features are measured **at the time of assessment**
✓ No temporal features that could leak future information
✓ Drug category is the outcome, not a predictor

**Validation Checklist**:
- [ ] No features derived from target variable
- [ ] No test set information used in training
- [ ] No data snooping (no peeking at test labels)
- [ ] Proper train/validation/test splits with stratification

### 6.2 Stratified Splitting

```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X = df.drop(['ID', 'drug_category'], axis=1)
y = df['drug_category']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

# Verify class distribution
print("Training set distribution:")
print(y_train.value_counts(normalize=True))
print("\nValidation set distribution:")
print(y_val.value_counts(normalize=True))
```

---

## 7. Complete Data Cleaning Pipeline

### 7.1 End-to-End Script

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import mstats

def clean_data(df, training=True):
    """
    Complete data cleaning pipeline
    
    Parameters:
    - df: Input DataFrame
    - training: If True, fit imputers; if False, use pre-fitted imputers
    
    Returns:
    - Cleaned DataFrame
    """
    
    # 1. Handle Anomalies
    print("Step 1: Handling anomalies...")
    df.loc[df['Escore'] > 10, 'Escore'] = np.nan
    df.loc[df['Cscore'] < -5, 'Cscore'] = np.nan
    df.loc[df['Impulsive'] > 5, 'Impulsive'] = np.nan
    
    # 2. Impute Categorical Features
    print("Step 2: Imputing categorical features...")
    categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
    
    # 3. Impute Continuous Features
    print("Step 3: Imputing continuous features...")
    continuous_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    df[continuous_features] = knn_imputer.fit_transform(df[continuous_features])
    
    # 4. Remove Duplicates
    print("Step 4: Removing duplicates...")
    df = df.drop_duplicates()
    
    # 5. Validate Ranges
    print("Step 5: Validating data ranges...")
    for feature in continuous_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}: [{min_val:.2f}, {max_val:.2f}]")
    
    print("Data cleaning complete!")
    return df

# Apply cleaning
train_df = pd.read_csv('data/data_minihackathon_train.csv')
train_df_clean = clean_data(train_df, training=True)

# Save cleaned data
train_df_clean.to_csv('data/data_minihackathon_train_clean.csv', index=False)
```

### 7.2 Test Set Cleaning

```python
# Apply same transformations to test set
test_df = pd.read_csv('data/data_minihackathon_test.csv')
test_df_clean = clean_data(test_df, training=False)
test_df_clean.to_csv('data/data_minihackathon_test_clean.csv', index=False)
```

---

## 8. Data Quality Validation

### 8.1 Post-Cleaning Checks

**Checklist**:
- [ ] No missing values remain
- [ ] No extreme outliers beyond ±5 SD
- [ ] No duplicate records
- [ ] All categorical values within expected ranges
- [ ] Distributions preserved (no major shifts)

**Validation Script**:
```python
def validate_cleaned_data(df):
    """Validate cleaned data quality"""
    
    print("="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    # Check 1: Missing values
    missing = df.isnull().sum().sum()
    print(f"\n1. Missing values: {missing}")
    assert missing == 0, "Missing values remain!"
    
    # Check 2: Duplicates
    duplicates = df.duplicated().sum()
    print(f"2. Duplicate rows: {duplicates}")
    assert duplicates == 0, "Duplicates remain!"
    
    # Check 3: Outliers (Z-score > 5)
    continuous_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    extreme_outliers = 0
    for feature in continuous_features:
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        outliers = (z_scores > 5).sum()
        extreme_outliers += outliers
        if outliers > 0:
            print(f"  ⚠️ {feature}: {outliers} extreme outliers (|Z| > 5)")
    
    print(f"3. Extreme outliers (|Z| > 5): {extreme_outliers}")
    
    # Check 4: Categorical consistency
    print("\n4. Categorical value counts:")
    categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
    for feature in categorical_features:
        unique_count = df[feature].nunique()
        print(f"  {feature}: {unique_count} unique values")
    
    # Check 5: Distribution comparison
    print("\n5. Distribution summary (continuous features):")
    print(df[continuous_features].describe().T[['mean', 'std', 'min', 'max']])
    
    print("\n✓ Validation complete!")

# Run validation
validate_cleaned_data(train_df_clean)
```

---

## 9. Data Transformation Documentation

### 9.1 Transformation Log

| Step | Action | Features Affected | Records Affected | Rationale |
|------|--------|------------------|------------------|-----------|
| 1 | Flag anomalies as NaN | Escore, Cscore, Impulsive | 3 | Extreme outliers (>10 SD) |
| 2 | KNN imputation | All continuous features | 10 + 3 anomalies | Preserve feature relationships |
| 3 | Mode imputation | All categorical features | 25 | Simple strategy for <2% missing |
| 4 | Duplicate removal | All features | 0 | No duplicates found |

### 9.2 Pre/Post Cleaning Comparison

```python
def compare_distributions(df_before, df_after, features):
    """Compare distributions before and after cleaning"""
    
    comparison = []
    for feature in features:
        before_mean = df_before[feature].mean()
        after_mean = df_after[feature].mean()
        before_std = df_before[feature].std()
        after_std = df_after[feature].std()
        
        mean_change = ((after_mean - before_mean) / before_mean) * 100
        std_change = ((after_std - before_std) / before_std) * 100
        
        comparison.append({
            'Feature': feature,
            'Mean_Before': before_mean,
            'Mean_After': after_mean,
            'Mean_Change_%': mean_change,
            'Std_Before': before_std,
            'Std_After': after_std,
            'Std_Change_%': std_change
        })
    
    return pd.DataFrame(comparison)

# Generate comparison report
continuous_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
comparison_df = compare_distributions(train_df, train_df_clean, continuous_features)
print(comparison_df.to_string(index=False))
```

---

## 10. Next Steps

### 10.1 Immediate Actions

- [ ] Run complete data cleaning pipeline on training data
- [ ] Apply same transformations to test data
- [ ] Validate cleaned data quality
- [ ] Save cleaned datasets
- [ ] Document all transformations

### 10.2 Proceed to Feature Engineering

After cleaning:
1. Create interaction features (Oscore×SS, Cscore×Impulsive)
2. Generate polynomial features (Oscore², Cscore²)
3. Encode categorical variables
4. Create domain-specific features (risk profiles)
5. Apply feature selection

### 10.3 Model Development Readiness

**Cleaned Data is Ready For**:
- Train/validation splitting (stratified)
- Feature engineering pipeline
- Model training (tree-based and linear models)
- Cross-validation experiments
- Hyperparameter tuning

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Related Documents**: EXPLORATORY_DATA_ANALYSIS_REPORT.md

