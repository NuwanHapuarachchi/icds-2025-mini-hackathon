# Data Quality Assessment & Cleaning - Complete Report
## ICDS 2025 Mini-Hackathon - Drug Category Prediction

---

## Status: ✅ COMPLETED

**Date:** 2025-11-16  
**Overall Data Quality Score:** 93.4/100 (Good)  
**Data Completeness:** 100% (after cleaning)  
**Ready for:** Feature Engineering

---

## Executive Summary

Comprehensive data quality assessment and cleaning has been completed for the drug category prediction dataset. The analysis examined 1,500 training records across 14 features, identifying and addressing 3 extreme anomalies, 31 missing values, and 70 outliers. The cleaned dataset is now 100% complete with all values within expected ranges, ready for feature engineering and model development.

**Key Achievements:**
- ✅ Identified and fixed 3 extreme anomalies (Escore=50, Cscore=-10, Impulsive=10)
- ✅ Imputed 31 missing values using KNN and mode imputation
- ✅ Validated 100% data completeness
- ✅ Confirmed no duplicate records
- ✅ Verified no data leakage concerns
- ✅ Cleaned both training and test datasets
- ✅ Generated comprehensive reports and visualizations

---

## Table of Contents

1. [Data Quality Assessment Results](#1-data-quality-assessment-results)
2. [Data Cleaning Pipeline](#2-data-cleaning-pipeline)
3. [Quality Metrics](#3-quality-metrics)
4. [Next Steps: Feature Engineering](#4-next-steps-feature-engineering)
5. [Next Steps: Feature Selection](#5-next-steps-feature-selection)
6. [Recommendations for Model Development](#6-recommendations-for-model-development)

---

## 1. Data Quality Assessment Results

### 1.1 Quality Scorecard

| Dimension | Score (out of 100) | Status | Details |
|-----------|-------------------|---------|---------|
| **Missing Values** | 99.85 | Excellent | Only 31 missing values (0.15% of cells) |
| **Duplicates** | 100.00 | Excellent | Zero duplicate records detected |
| **Outliers** | 99.33 | Good | 70 outliers (0.67% of data points) |
| **Anomalies** | 99.97 | Good | 3 extreme values requiring treatment |
| **Consistency** | 57.14 | Fair | Range violations from extreme anomalies |
| **OVERALL** | **93.39** | **Good** | Ready for feature engineering |

### 1.2 Missing Values Analysis

**Total Missing:** 31 values across 10 features

| Feature | Missing Count | Missing % | Strategy Applied |
|---------|--------------|-----------|-----------------|
| Education | 20 | 1.33% | Mode imputation |
| Age | 2 | 0.13% | Mode imputation |
| Country | 2 | 0.13% | Mode imputation |
| Gender | 1 | 0.07% | Mode imputation |
| Nscore | 1 | 0.07% | KNN imputation (k=5) |
| Escore | 1 + 1 anomaly | 0.13% | KNN imputation (k=5) |
| Oscore | 1 | 0.07% | KNN imputation (k=5) |
| Ascore | 1 | 0.07% | KNN imputation (k=5) |
| Cscore | 1 + 1 anomaly | 0.13% | KNN imputation (k=5) |
| Impulsive | 1 + 1 anomaly | 0.13% | KNN imputation (k=5) |

**Result:** ✅ 100% data completeness achieved

### 1.3 Extreme Anomalies Detected

| Feature | Record ID | Anomaly Value | Z-Score | Normal Range | Action Taken |
|---------|-----------|---------------|---------|--------------|--------------|
| **Escore** | 1108 | 50.00 | +30.81 | [-3.27, 3.27] | Replaced with NaN → KNN imputed |
| **Cscore** | 1250 | -10.00 | -9.69 | [-3.46, 3.46] | Replaced with NaN → KNN imputed |
| **Impulsive** | 24 | 10.00 | +10.11 | [-2.56, 2.90] | Replaced with NaN → KNN imputed |

**Impact:** Fixed severe distribution distortions (Escore kurtosis reduced from 601.72 to -0.03)

### 1.4 Outlier Summary

**IQR Method (1.5 × IQR):** 70 outliers (0.67% of continuous data points)

| Feature | Outlier Count | Outlier % | Treatment |
|---------|--------------|-----------|-----------|
| Ascore | 24 | 1.60% | Kept (below 2% threshold) |
| Nscore | 16 | 1.07% | Kept |
| Escore | 12 | 0.80% | Kept |
| Impulsive | 7 | 0.47% | Kept |
| Oscore | 6 | 0.40% | Kept |
| Cscore | 5 | 0.33% | Kept |
| SS | 0 | 0.00% | None needed |

**Decision:** Outliers retained for tree-based models; RobustScaler recommended for linear models

### 1.5 Class Balance Analysis

| Drug Category | Count | Percentage | Classification |
|--------------|-------|------------|----------------|
| **Hallucinogens** | 691 | 46.07% | Majority class |
| **Stimulants** | 567 | 37.80% | Intermediate |
| **Depressants** | 242 | 16.13% | Minority class |

**Imbalance Ratio:** 2.86:1 (Hallucinogens to Depressants)

**Recommendation:** Use stratified sampling, class weights, and macro F1-score for evaluation

---

## 2. Data Cleaning Pipeline

### 2.1 Cleaning Steps Executed

**Step 1: Handle Extreme Anomalies**
- Flagged Escore > 10 as NaN (1 value)
- Flagged Cscore < -5 as NaN (1 value)
- Flagged Impulsive > 5 as NaN (1 value)
- **Total anomalies flagged:** 3

**Step 2: Impute Categorical Features**
- Applied mode imputation to Age, Gender, Education, Country, Ethnicity
- **Total values imputed:** 25

**Step 3: Impute Continuous Features**
- Applied KNN imputation (k=5, distance-weighted) to personality traits and behavioral measures
- **Total values imputed:** 9 (original) + 3 (anomalies) = 12

**Step 4: Validate Cleaned Data**
- ✅ Zero missing values
- ✅ All features within ±5 SD range
- ✅ No duplicate records
- ✅ Normal distributions restored

**Step 5: Apply to Test Set**
- Same transformations applied to test set (377 records)
- Both datasets cleaned consistently

### 2.2 Distribution Changes (Before vs After)

| Feature | Mean Change | Std Change | Status |
|---------|------------|-----------|--------|
| Nscore | 10.1% | 0.03% | ✓ Minimal |
| **Escore** | 155.4% | 39.5% | ⚠️ Significant (anomaly fixed) |
| Oscore | 18.2% | 0.01% | ✓ Minimal |
| Ascore | 2.3% | 0.02% | ✓ Minimal |
| **Cscore** | 236.9% | 3.2% | ⚠️ Significant (anomaly fixed) |
| **Impulsive** | 113.7% | 3.4% | ⚠️ Significant (anomaly fixed) |
| SS | 0.0% | 0.0% | ✓ No change |

**Note:** Large changes in Escore, Cscore, and Impulsive are **expected and desired** due to fixing extreme anomalies

### 2.3 Post-Cleaning Statistics

| Feature | Mean | Std | Min | Max | Skewness | Kurtosis |
|---------|------|-----|-----|-----|----------|----------|
| Nscore | 0.01 | 1.00 | -3.46 | 3.27 | -0.02 | -0.01 |
| Escore | -0.01 | 0.98 | -3.27 | 3.27 | -0.00 | -0.03 |
| Oscore | 0.00 | 0.99 | -2.86 | 2.90 | 0.04 | -0.14 |
| Ascore | -0.02 | 0.99 | -3.46 | 3.46 | 0.02 | 0.07 |
| Cscore | 0.00 | 1.00 | -3.16 | 3.46 | 0.00 | -0.17 |
| Impulsive | -0.00 | 0.95 | -2.56 | 2.90 | 0.10 | -0.29 |
| SS | 0.00 | 0.97 | -2.08 | 1.92 | -0.05 | -0.45 |

**Assessment:** ✅ All distributions are now normalized with skewness and kurtosis near zero

---

## 3. Quality Metrics

### 3.1 Data Quality Improvements

| Metric | Before Cleaning | After Cleaning | Improvement |
|--------|----------------|----------------|-------------|
| Missing Values | 31 (0.15%) | 0 (0%) | +100% complete |
| Extreme Anomalies | 3 | 0 | +100% fixed |
| Range Violations | 3 features | 0 features | +100% compliant |
| Escore Kurtosis | 601.72 | -0.03 | +99.995% improvement |
| Data Completeness | 99.85% | 100% | +0.15% |

### 3.2 Files Generated

**Cleaned Data Files:**
1. ✅ `data/data_minihackathon_train_clean.csv` (1,500 records, 14 features)
2. ✅ `data/data_minihackathon_test_clean.csv` (377 records, 13 features)

**Reports (10 files):**
1. `reports/dqa/dqa_missing_values.csv`
2. `reports/dqa/dqa_outliers_iqr.csv`
3. `reports/dqa/dqa_outliers_zscore.csv`
4. `reports/dqa/dqa_anomalies.csv`
5. `reports/dqa/dqa_consistency.csv`
6. `reports/dqa/dqa_class_balance.csv`
7. `reports/dqa/dqa_summary_statistics.csv`
8. `reports/dqa/dqa_scorecard.csv`
9. `reports/dqa/data_cleaning_comparison.csv`
10. `reports/dqa/data_cleaning_summary.csv`

**Visualizations (6 files):**
1. `visualizations/dqa_missing_values.png`
2. `visualizations/dqa_outliers_boxplots.png`
3. `visualizations/dqa_class_balance.png`
4. `visualizations/dqa_scorecard.png`
5. `visualizations/data_cleaning_comparison.png`
6. `visualizations/data_cleaning_boxplots.png`

**Documentation:**
1. `reports/dqa/DATA_QUALITY_ASSESSMENT_REPORT.md` (comprehensive DQA report)

**Scripts:**
1. `scripts/data_quality_assessment.py` (assessment pipeline)
2. `scripts/data_cleaning.py` (cleaning pipeline)

---

## 4. Next Steps: Feature Engineering

### 4.1 Domain-Specific Features

**Psychological Risk Indicators:**

1. **Risk-Taking Behavior** = `Impulsive × SS`
   - Rationale: Combined measure of impulsivity and novelty-seeking
   - Expected to predict hallucinogen use

2. **Exploratory Tendency** = `Oscore × SS`
   - Rationale: Openness combined with sensation seeking
   - Strong predictor for hallucinogen use (based on EDA)

3. **Self-Control Index** = `Cscore / (Impulsive + 2)`
   - Rationale: Conscientiousness moderated by impulsivity
   - Expected to predict stimulant use (organized, low impulsivity)

4. **Anxious Impulsivity** = `Nscore × Impulsive`
   - Rationale: Emotional instability with poor impulse control
   - May predict substance use for emotional regulation

5. **Emotional Instability** = `Nscore - Cscore`
   - Rationale: High neuroticism + low conscientiousness
   - Indicates emotional dysregulation

6. **Exploratory Profile** = `Oscore + SS - Cscore`
   - Rationale: Novelty-seeking minus constraint
   - Composite measure for hallucinogen preference

### 4.2 Interaction Terms

**High-Priority Interactions (based on EDA effect sizes):**

```python
# Age-based interactions (age shows strong patterns)
'Age_x_Oscore': Age × Oscore  # Openness effects vary by age
'Age_x_Nscore': Age × Nscore  # Neuroticism across lifespan
'Age_x_SS': Age × SS  # Sensation seeking by age group

# Personality interactions
'Oscore_x_SS': Oscore × SS  # Exploratory behavior
'Cscore_x_Impulsive': Cscore × Impulsive  # Self-control
'Nscore_x_Impulsive': Nscore × Impulsive  # Anxious impulsivity
'Ascore_x_Cscore': Ascore × Cscore  # Social constraint
```

### 4.3 Polynomial Features

**Non-Linear Transformations:**

```python
# Quadratic terms for features with large effect sizes
'Oscore_squared': Oscore²  # Capture non-linear openness effects
'SS_squared': SS²  # Capture threshold effects in sensation seeking
'Cscore_squared': Cscore²  # Non-linear conscientiousness
'Impulsive_squared': Impulsive²  # Extreme impulsivity patterns

# Cubic terms for strongest predictors
'Oscore_cubed': Oscore³
'SS_cubed': SS³
```

### 4.4 Aggregation Features

**Big Five Composite Scores:**

```python
# Personality summary statistics
'personality_mean': mean(Nscore, Escore, Oscore, Ascore, Cscore)
'personality_std': std(Nscore, Escore, Oscore, Ascore, Cscore)
'personality_range': max(...) - min(...)

# Behavioral composite
'behavioral_risk': 0.4×Impulsive + 0.6×SS  # Weighted by effect size

# Substance use risk score (evidence-based weights)
'substance_risk': 0.35×SS + 0.30×Oscore + 0.20×Impulsive + 0.15×(-Cscore)
```

### 4.5 Encoding Strategy

**Categorical Features:**

| Feature | Encoding Strategy | Rationale |
|---------|------------------|-----------|
| **Age** | Ordinal (1-6) | Natural ordering (18-24 → 65+) |
| **Education** | Ordinal (1-9) | Natural ordering (low → high) |
| **Gender** | Binary (0/1) | Only 2 categories |
| **Country** | One-Hot or Target Encoding | 7 categories, no natural order |
| **Ethnicity** | One-Hot or Target Encoding | 7 categories, no natural order |

**Recommendations:**
- Use **one-hot encoding** for Country and Ethnicity if feature count is acceptable
- Use **target encoding** (with cross-validation) if dimensionality becomes an issue
- **Drop Escore** if feature selection confirms low importance (p=0.57 from EDA)

### 4.6 Feature Transformations

**For Linear Models:**
```python
# Log transformations for skewed features (if needed)
'Impulsive_log': np.log(Impulsive + constant)  # Positive skew

# Box-Cox transformations
'Nscore_boxcox': boxcox(Nscore + constant)

# Scaling strategies
RobustScaler()  # For features with remaining outliers
StandardScaler()  # For well-behaved features
```

**For Tree-Based Models:**
- No scaling required
- Keep original features + engineered features

---

## 5. Next Steps: Feature Selection

### 5.1 Low-Variance Feature Removal

**Check:**
- Features with near-zero variance
- Features with >95% same value

**Expected:** No issues (all features have good variance from EDA)

### 5.2 Correlation-Based Selection

**Rules:**
- Remove features with correlation > 0.95 with another feature
- Keep domain-important features in case of ties

**From EDA:**
- ✅ No multicollinearity detected (all correlations < 0.15)
- All Big Five traits are independent

### 5.3 Statistical Feature Selection

**Methods to Apply:**

1. **Mutual Information** (for non-linear relationships)
   ```python
   from sklearn.feature_selection import mutual_info_classif
   mi_scores = mutual_info_classif(X, y)
   ```

2. **Chi-Square Test** (for categorical features)
   ```python
   from sklearn.feature_selection import chi2
   chi_scores = chi2(X_encoded, y)
   ```

3. **ANOVA F-test** (for continuous features)
   ```python
   from sklearn.feature_selection import f_classif
   f_scores = f_classif(X, y)
   ```

### 5.4 Model-Based Feature Selection

**Random Forest Feature Importance:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = rf.feature_importances_
```

**L1 Regularization (Lasso):**
```python
from sklearn.linear_model import LogisticRegressionCV

lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=5)
lasso.fit(X_train, y_train)
selected_features = X.columns[lasso.coef_[0] != 0]
```

**Recursive Feature Elimination (RFE):**
```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=rf, n_features_to_select=20)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]
```

### 5.5 Hybrid Approach (Recommended)

**Step 1: Domain-Guided Pre-Selection**
- **Must-Keep Features** (from EDA):
  - Sensation Seeking (SS): η² = 0.228
  - Openness (Oscore): η² = 0.176
  - Impulsiveness: η² = 0.110
  - Conscientiousness (Cscore): η² = 0.070
  - Age: χ² = 285.59
  
- **Consider Dropping:**
  - Extraversion (Escore): p = 0.57 (not significant)

**Step 2: Statistical Filtering**
- Remove features with p-value > 0.05
- Remove features with effect size < 0.01

**Step 3: Model-Based Ranking**
- Use Random Forest importance
- Use Mutual Information scores
- Combine rankings

**Step 4: Final Selection**
- Target: 20-25 features total
- Include top domain features
- Include top statistical features
- Include top engineered features

---

## 6. Recommendations for Model Development

### 6.1 Train/Validation Split Strategy

```python
from sklearn.model_selection import train_test_split

# Stratified split (maintain class distribution)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Critical for imbalanced data
)

# Verify class distribution
print("Training distribution:", y_train.value_counts(normalize=True))
print("Validation distribution:", y_val.value_counts(normalize=True))
```

### 6.2 Class Imbalance Handling

**Option 1: Class Weights (Recommended)**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# For XGBoost
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
```

**Option 2: SMOTE (Optional - for minority class)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Option 3: Threshold Tuning**
- Adjust decision thresholds per class
- Use validation set to optimize thresholds

### 6.3 Recommended Models

**Priority 1: Tree-Based Ensemble Models**

1. **XGBoost** (Primary recommendation)
   ```python
   from xgboost import XGBClassifier
   
   xgb = XGBClassifier(
       n_estimators=200,
       max_depth=6,
       learning_rate=0.1,
       scale_pos_weight=class_weights,
       random_state=42
   )
   ```

2. **LightGBM**
   ```python
   from lightgbm import LGBMClassifier
   
   lgbm = LGBMClassifier(
       n_estimators=200,
       max_depth=6,
       learning_rate=0.1,
       class_weight='balanced',
       random_state=42
   )
   ```

3. **CatBoost**
   ```python
   from catboost import CatBoostClassifier
   
   catboost = CatBoostClassifier(
       iterations=200,
       depth=6,
       learning_rate=0.1,
       auto_class_weights='Balanced',
       random_state=42,
       verbose=0
   )
   ```

4. **Random Forest**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   rf = RandomForestClassifier(
       n_estimators=200,
       max_depth=10,
       class_weight='balanced',
       random_state=42
   )
   ```

**Priority 2: Baseline Models**

1. **Logistic Regression** (baseline)
2. **Decision Tree** (interpretability)
3. **Dummy Classifier** (naive baseline)

### 6.4 Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring='f1_macro'  # Macro F1 for imbalanced data
)

print(f"CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 6.5 Evaluation Metrics

**Primary Metric:**
- **Macro F1-Score** (equal weight to all classes)

**Secondary Metrics:**
- Weighted F1-Score
- Per-class Precision, Recall, F1
- Confusion Matrix
- ROC-AUC (one-vs-rest)

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Predictions
y_pred = model.predict(X_val)

# Macro F1 (primary)
macro_f1 = f1_score(y_val, y_pred, average='macro')

# Detailed report
print(classification_report(y_val, y_pred))

# Confusion matrix
print(confusion_matrix(y_val, y_pred))
```

### 6.6 Hyperparameter Tuning

**GridSearchCV:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    xgb,
    param_grid,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**RandomizedSearchCV (faster):**
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1]
}

random_search = RandomizedSearchCV(
    xgb,
    param_distributions,
    n_iter=50,
    cv=skf,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

### 6.7 Expected Performance

**Baseline (Dummy Classifier):**
- Accuracy: ~46% (majority class)

**Realistic Target:**
- **Accuracy:** 70-80%
- **Macro F1-Score:** 0.65-0.75
- **Per-Class F1:**
  - Hallucinogens: 0.75-0.85 (majority, high SS/Oscore)
  - Stimulants: 0.70-0.80 (clear patterns, high Cscore)
  - Depressants: 0.50-0.65 (minority, weak patterns)

**Optimistic Target (with extensive tuning):**
- **Accuracy:** 80-85%
- **Macro F1-Score:** 0.75-0.80

---

## Summary

### Completed Work ✅

1. **Data Quality Assessment**
   - Missing values analysis
   - Outlier detection (IQR and Z-score)
   - Anomaly detection and flagging
   - Data consistency validation
   - Duplicate detection
   - Data leakage verification
   - Class balance analysis
   - Quality scorecard (93.4/100)

2. **Data Cleaning**
   - Fixed 3 extreme anomalies
   - Imputed 31 missing values
   - Validated 100% completeness
   - Cleaned training set (1,500 records)
   - Cleaned test set (377 records)
   - Generated 10 comprehensive reports
   - Created 6 visualizations

### Ready for Next Phase ✅

**Feature Engineering:**
- Domain-specific features (risk profiles)
- Interaction terms (Oscore×SS, Age×Personality)
- Polynomial features (quadratic, cubic)
- Aggregation features (composite scores)
- Categorical encoding (one-hot, target)

**Feature Selection:**
- Low-variance removal
- Correlation analysis
- Statistical tests (MI, chi-square, ANOVA)
- Model-based selection (RF importance, L1, RFE)
- Hybrid domain + statistical approach

**Model Development:**
- Stratified train/validation splits
- Class weight implementation
- XGBoost, LightGBM, CatBoost models
- Hyperparameter tuning
- Macro F1-score evaluation
- Cross-validation experiments

---

**Status:** ✅ **Data Quality Assessment & Cleaning COMPLETE**  
**Next Action:** Begin Feature Engineering Phase  
**Timeline:** Ready to proceed immediately

**Files Ready:**
- `data/data_minihackathon_train_clean.csv`
- `data/data_minihackathon_test_clean.csv`

**Documentation:**
- Full DQA report in `reports/dqa/DATA_QUALITY_ASSESSMENT_REPORT.md`
- All scripts, reports, and visualizations available
