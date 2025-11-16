# NEXT AGENT REFERENCE GUIDE
## Data Quality Assessment & Cleaning - Complete Reference
### ICDS 2025 Mini-Hackathon - Drug Category Prediction

---

## 🎯 Quick Status

**Phase Completed:** ✅ Data Quality Assessment & Cleaning  
**Data Quality Score:** 93.4/100 (Good)  
**Data Completeness:** 100%  
**Ready For:** Feature Engineering & Model Development

**Clean Data Files:**
- ✅ `data/data_minihackathon_train_clean.csv` (1,500 records, 14 features)
- ✅ `data/data_minihackathon_test_clean.csv` (377 records, 13 features)

---

## 📊 Critical Findings Summary

### Data Quality Issues Identified & Fixed

| Issue Type | Count | Status | Impact |
|-----------|-------|--------|---------|
| **Extreme Anomalies** | 3 | ✅ Fixed | Escore=50, Cscore=-10, Impulsive=10 replaced with KNN imputation |
| **Missing Values** | 31 | ✅ Fixed | Mode (categorical) & KNN (continuous) imputation applied |
| **Outliers (IQR)** | 70 | ✅ Kept | 0.67% of data - retained for tree models |
| **Duplicates** | 0 | ✅ None | No action needed |
| **Data Leakage** | 0 | ✅ None | All features validated |

### Class Distribution (Imbalanced)

| Class | Count | % | Strategy |
|-------|-------|---|----------|
| Hallucinogens | 691 | 46.07% | Majority class |
| Stimulants | 567 | 37.80% | Intermediate |
| Depressants | 242 | 16.13% | Minority - needs class weights |

**Imbalance Ratio:** 2.86:1 → **MUST use stratified sampling + class weights**

---

## 🔑 Key Features for Modeling (From EDA)

### Top Predictors by Effect Size

| Rank | Feature | Effect Size (η²) | P-Value | Strength |
|------|---------|-----------------|---------|----------|
| 1 | **Sensation Seeking (SS)** | 0.228 | <10⁻⁷⁷ | 🔴 CRITICAL |
| 2 | **Openness (Oscore)** | 0.176 | <10⁻⁶³ | 🔴 CRITICAL |
| 3 | **Impulsiveness** | 0.110 | <10⁻³⁸ | 🟡 High |
| 4 | **Conscientiousness (Cscore)** | 0.070 | <10⁻²⁴ | 🟡 Medium |
| 5 | **Age** (categorical) | χ²=285.59 | <10⁻⁵⁵ | 🔴 CRITICAL |
| 6 | Agreeableness (Ascore) | 0.036 | <10⁻¹² | 🟢 Small |
| 7 | Neuroticism (Nscore) | 0.015 | <10⁻⁵ | 🟢 Small |
| 8 | ❌ **Extraversion (Escore)** | 0.001 | 0.57 | ⚠️ NOT significant - consider dropping |

### Feature Patterns

**Hallucinogen Users:**
- ⬆️ High Openness (Oscore: 0.419)
- ⬆️ Very high Sensation Seeking (SS: 0.467)
- ⬆️ High Impulsiveness (0.326)
- ⬇️ Low Conscientiousness (-0.264)
- 👥 Predominantly young (72% are 18-24 years old)

**Stimulant Users:**
- ⬇️ Low Openness (Oscore: -0.491)
- ⬇️ Very low Sensation Seeking (SS: -0.543)
- ⬇️ Low Impulsiveness (-0.393)
- ⬆️ High Conscientiousness (0.335)
- 👥 Predominantly older (60-80% are 35+ years old)

**Depressants Users:**
- ⚠️ Weakest patterns (minority class, hardest to predict)
- Near-zero differences across all features

---

## 🛠️ Data Cleaning Applied

### Transformations Log

**Step 1: Anomaly Flagging**
```python
# 3 extreme values flagged as NaN
Escore > 10 → NaN (1 record: ID 1108, value=50.00, Z=30.81)
Cscore < -5 → NaN (1 record: ID 1250, value=-10.00, Z=-9.69)
Impulsive > 5 → NaN (1 record: ID 24, value=10.00, Z=10.11)
```

**Step 2: Categorical Imputation**
```python
# Mode imputation (25 values)
Age: 2 missing → mode
Gender: 1 missing → mode
Education: 20 missing → mode
Country: 2 missing → mode
```

**Step 3: Continuous Imputation**
```python
# KNN imputation (k=5, distance-weighted, 12 values total)
# Original missing: 9 values
# Anomalies: 3 values
# All personality traits + behavioral measures
```

**Result:** ✅ 100% data completeness

### Distribution Changes

| Feature | Mean Change | Std Change | Status |
|---------|-------------|------------|--------|
| **Escore** | 155.4% ⚠️ | 39.5% ⚠️ | Fixed anomaly - distributions normalized |
| **Cscore** | 236.9% ⚠️ | 3.2% | Fixed anomaly - distributions normalized |
| **Impulsive** | 113.7% ⚠️ | 3.4% | Fixed anomaly - distributions normalized |
| Nscore | 10.1% | 0.03% | ✅ Minimal change |
| Oscore | 18.2% | 0.01% | ✅ Minimal change |
| Ascore | 2.3% | 0.02% | ✅ Minimal change |
| SS | 0.0% | 0.0% | ✅ No change |

⚠️ **Large changes in Escore, Cscore, Impulsive are EXPECTED and DESIRED** - we fixed data entry errors!

---

## 📁 Documentation Structure

```
icds-2025-mini-hackathon/
├── data/
│   ├── data_minihackathon_train_clean.csv ✅ USE THIS
│   ├── data_minihackathon_test_clean.csv ✅ USE THIS
│   ├── data_minihackathon_train.csv (original)
│   └── data_minihackathon_test.csv (original)
│
├── reports/
│   ├── exploration/ (initial data understanding)
│   ├── eda/ (exploratory data analysis)
│   │   ├── EXPLORATORY_DATA_ANALYSIS_REPORT.md ⭐ Main EDA
│   │   ├── EDA_SUMMARY.md
│   │   ├── FEATURE_ENGINEERING_GUIDE.md ⭐ Feature ideas
│   │   └── eda_*.csv (statistical tests, effect sizes)
│   │
│   └── dqa/ (data quality & cleaning) ⭐ NEW
│       ├── DQA_COMPLETE_SUMMARY.md ⭐ START HERE
│       ├── DATA_QUALITY_ASSESSMENT_REPORT.md ⭐ Full DQA
│       ├── DATA_QUALITY_ASSESSMENT_GUIDE.md (cleaning guide)
│       └── data/ (CSV reports)
│           ├── dqa_missing_values.csv
│           ├── dqa_anomalies.csv
│           ├── dqa_outliers_iqr.csv
│           ├── dqa_outliers_zscore.csv
│           ├── dqa_scorecard.csv
│           ├── dqa_class_balance.csv
│           ├── data_cleaning_comparison.csv
│           └── data_cleaning_summary.csv
│
├── scripts/
│   ├── eda_analysis.py (reproducible EDA)
│   ├── data_quality_assessment.py ⭐ DQA pipeline
│   └── data_cleaning.py ⭐ Cleaning pipeline
│
└── visualizations/
    ├── eda_*.png (15 EDA visualizations)
    ├── dqa_*.png (4 DQA visualizations)
    └── data_cleaning_*.png (2 cleaning comparisons)
```

---

## 🚀 Next Steps: Feature Engineering

### Priority 1: Domain-Specific Features

**Create these psychological risk indicators:**

```python
# High-priority engineered features
risk_taking = Impulsive × SS  # Combined impulsivity + novelty seeking
exploratory_tendency = Oscore × SS  # Strongest predictor combination
self_control = Cscore / (Impulsive + 2)  # Conscientiousness vs impulsivity
anxious_impulsivity = Nscore × Impulsive  # Emotional + behavioral risk
emotional_instability = Nscore - Cscore  # Dysregulation index
exploratory_profile = Oscore + SS - Cscore  # Net exploration tendency

# Substance risk score (evidence-based weights from EDA)
substance_risk = 0.35×SS + 0.30×Oscore + 0.20×Impulsive + 0.15×(-Cscore)
```

### Priority 2: Age Interactions (Age shows strong patterns!)

```python
# Age-based interactions
Age_x_Oscore  # Young people's openness → hallucinogens
Age_x_SS  # Sensation seeking varies by age
Age_x_Nscore  # Emotional patterns across lifespan
Age_x_Cscore  # Conscientiousness by age group
```

### Priority 3: Polynomial Features

```python
# Non-linear effects for top predictors
Oscore²  # Extreme openness patterns
SS²  # Sensation seeking thresholds
Cscore²  # Conscientiousness extremes
Impulsive²  # Extreme impulsivity
```

### Priority 4: Categorical Encoding

| Feature | Strategy | Reason |
|---------|----------|--------|
| Age | Ordinal (1-6) | Natural ordering |
| Education | Ordinal (1-9) | Natural ordering |
| Gender | Binary (0/1) | 2 categories |
| Country | One-Hot or Target | 7 categories |
| Ethnicity | One-Hot or Target | 7 categories |

**⚠️ Consider dropping Escore** (p=0.57, not significant)

---

## 🎯 Model Development Strategy

### Recommended Approach

**1. Baseline Models First**
```python
# Establish baseline
DummyClassifier (stratified)  # Baseline: ~46% accuracy
LogisticRegression (class_weight='balanced')
DecisionTree (max_depth=5, class_weight='balanced')
```

**2. Tree-Based Ensemble (Primary)**
```python
# Best expected performance
XGBoost (PRIMARY) - handles imbalance, interactions, missing values
LightGBM - fast, categorical support
CatBoost - native categorical handling
RandomForest - robust baseline
```

**3. Ensemble Stacking**
```python
# Final model
Voting/Stacking of top 3 models
```

### Critical Settings

**MUST USE for all models:**
```python
# 1. Stratified splits
train_test_split(..., stratify=y)
StratifiedKFold(n_splits=5)

# 2. Class weights
class_weight='balanced'  # sklearn
scale_pos_weight=...  # XGBoost
auto_class_weights='Balanced'  # CatBoost

# 3. Evaluation metric
scoring='f1_macro'  # PRIMARY METRIC (not accuracy!)
```

### Expected Performance

| Scenario | Accuracy | Macro F1 | Notes |
|----------|----------|----------|-------|
| **Baseline** | 46% | 0.30 | Majority class predictor |
| **Realistic Target** | 70-80% | 0.65-0.75 | Achievable with tuning |
| **Optimistic Target** | 80-85% | 0.75-0.80 | Extensive feature engineering + tuning |

**Per-Class Expectations:**
- Hallucinogens: F1 = 0.75-0.85 (easiest - majority class, strong patterns)
- Stimulants: F1 = 0.70-0.80 (moderate - clear patterns)
- Depressants: F1 = 0.50-0.65 (hardest - minority class, weak patterns)

---

## ⚠️ Critical Warnings

### DO NOT:
1. ❌ Use overall accuracy as primary metric (misleading with imbalance)
2. ❌ Forget stratification in splits/CV (will bias results)
3. ❌ Ignore class weights (will under-predict minority class)
4. ❌ Drop outliers (only 0.67% - keep for tree models)
5. ❌ Use original data files (use *_clean.csv files)
6. ❌ Include Escore without validation (p=0.57, likely not useful)

### DO:
1. ✅ Use **Macro F1-score** as primary evaluation metric
2. ✅ Apply **stratified sampling** in all splits
3. ✅ Use **class weights** in all models
4. ✅ Monitor **per-class performance** (confusion matrix)
5. ✅ Start with **top 5 features** (SS, Oscore, Impulsive, Cscore, Age)
6. ✅ Create **interaction features** (especially Age × Personality)
7. ✅ Use **tree-based models** (handle outliers + interactions)

---

## 📊 Data Statistics Reference

### Continuous Features (After Cleaning)

| Feature | Mean | Std | Min | Max | Skew | Kurtosis |
|---------|------|-----|-----|-----|------|----------|
| Nscore | 0.01 | 1.00 | -3.46 | 3.27 | -0.02 | -0.01 |
| Escore | -0.01 | 0.98 | -3.27 | 3.27 | -0.00 | -0.03 |
| Oscore | 0.00 | 0.99 | -2.86 | 2.90 | 0.04 | -0.14 |
| Ascore | -0.02 | 0.99 | -3.46 | 3.46 | 0.02 | 0.07 |
| Cscore | 0.00 | 1.00 | -3.16 | 3.46 | 0.00 | -0.17 |
| Impulsive | -0.00 | 0.95 | -2.56 | 2.90 | 0.10 | -0.29 |
| SS | 0.00 | 0.97 | -2.08 | 1.92 | -0.05 | -0.45 |

✅ All features now have normal distributions (skewness ≈ 0, kurtosis ≈ 0)

### Categorical Features

| Feature | Unique Values | Most Common | Distribution |
|---------|--------------|-------------|--------------|
| Age | 6 | 18-24 (34%) | Young skew |
| Gender | 2 | Male (50%) | Balanced |
| Education | 9 | Some college (25%) | Moderate spread |
| Country | 7 | UK (55%) | High geographic bias |
| Ethnicity | 7 | White (91%) | Severe ethnic bias |

⚠️ **Model may not generalize to non-UK/USA or non-White populations**

---

## 🔍 Feature Selection Guidance

### Must-Keep Features (Strong Evidence)
1. **SS** (Sensation Seeking) - η² = 0.228 ⭐⭐⭐
2. **Oscore** (Openness) - η² = 0.176 ⭐⭐⭐
3. **Age** - χ² = 285.59 ⭐⭐⭐
4. **Impulsive** - η² = 0.110 ⭐⭐
5. **Cscore** (Conscientiousness) - η² = 0.070 ⭐⭐

### Likely Useful
6. **Ascore** (Agreeableness) - η² = 0.036 ⭐
7. **Nscore** (Neuroticism) - η² = 0.015 ⭐
8. **Education** - χ² = 231.88 ⭐
9. **Country** - χ² = 430.35 ⭐
10. **Gender** - χ² = 173.69 ⭐

### Consider Dropping
11. ❌ **Escore** (Extraversion) - p = 0.57 (NOT significant)

### Feature Selection Methods to Use

```python
# 1. Mutual Information (non-linear relationships)
from sklearn.feature_selection import mutual_info_classif

# 2. Random Forest Importance
rf.feature_importances_

# 3. Recursive Feature Elimination
from sklearn.feature_selection import RFE

# 4. L1 Regularization (Lasso)
LogisticRegressionCV(penalty='l1')
```

**Target:** 20-25 total features (original + engineered)

---

## 📈 Evaluation Framework

### Primary Metric: Macro F1-Score

```python
from sklearn.metrics import f1_score, classification_report

# Macro F1 (equal weight to all classes)
macro_f1 = f1_score(y_val, y_pred, average='macro')

# Full classification report
print(classification_report(y_val, y_pred, 
      target_names=['Depressants', 'Hallucinogens', 'Stimulants']))
```

### Cross-Validation Setup

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified CV (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=skf,
    scoring='f1_macro'  # PRIMARY METRIC
)

print(f"CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Monitor These Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **Macro F1** | Overall balanced performance | >0.65 |
| Weighted F1 | Performance weighted by class size | >0.70 |
| Per-class F1 | Individual class performance | Depressants >0.50 |
| Confusion Matrix | Misclassification patterns | Visual analysis |
| ROC-AUC (OvR) | Discriminative ability | >0.80 |

---

## 💡 Quick Start Code Template

```python
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# 1. Load clean data
train_df = pd.read_csv('data/data_minihackathon_train_clean.csv')
test_df = pd.read_csv('data/data_minihackathon_test_clean.csv')

# 2. Separate features and target
X = train_df.drop(['ID', 'drug_category'], axis=1)
y = train_df['drug_category']

# 3. Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Verify class distribution
print("Train:", y_train.value_counts(normalize=True))
print("Val:", y_val.value_counts(normalize=True))

# 5. Feature engineering (add your features here)
# ... create interaction terms, polynomials, etc.

# 6. Train model with class weights
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    class_weight='balanced'  # CRITICAL for imbalanced data
)

model.fit(X_train, y_train)

# 7. Evaluate with Macro F1
y_pred = model.predict(X_val)
macro_f1 = f1_score(y_val, y_pred, average='macro')
print(f"\nMacro F1-Score: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# 8. Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, 
                            cv=skf, scoring='f1_macro')
print(f"\nCV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 📚 Reference Documents

**Read These in Order:**

1. **DQA_COMPLETE_SUMMARY.md** (this file) - Overall summary
2. **DATA_QUALITY_ASSESSMENT_REPORT.md** - Detailed DQA findings
3. **reports/eda/EXPLORATORY_DATA_ANALYSIS_REPORT.md** - Full EDA analysis
4. **reports/eda/FEATURE_ENGINEERING_GUIDE.md** - Feature engineering ideas

**Data Files:**

- **reports/dqa/data/** - All CSV reports (anomalies, outliers, stats)
- **reports/eda/** - EDA statistical tests and effect sizes

**Scripts to Run:**

- `scripts/data_quality_assessment.py` - Reproduce DQA
- `scripts/data_cleaning.py` - Reproduce cleaning
- `scripts/eda_analysis.py` - Reproduce EDA

---

## ✅ Final Checklist Before Modeling

- [x] Data cleaned (100% complete)
- [x] Anomalies fixed (3 extreme values)
- [x] Class imbalance understood (2.86:1 ratio)
- [x] Top features identified (SS, Oscore, Age, Impulsive, Cscore)
- [ ] Feature engineering completed
- [ ] Categorical encoding applied
- [ ] Feature selection performed
- [ ] Stratified train/val split created
- [ ] Class weights configured
- [ ] Baseline model trained
- [ ] Tree-based models trained
- [ ] Macro F1-score evaluated
- [ ] Cross-validation performed
- [ ] Hyperparameter tuning done
- [ ] Final predictions generated

---

## 🎓 Key Insights for High Accuracy

1. **Feature Engineering is Critical**: The interaction between Oscore × SS is likely your strongest predictor for hallucinogens

2. **Age Patterns are Gold**: 72% of 18-24 year-olds use hallucinogens, while 60-80% of 35+ use stimulants - create age interactions!

3. **Don't Fight the Imbalance**: Use stratification + class weights rather than SMOTE (works better with tree models)

4. **Depressants Will Be Hard**: Only 16% of data, weakest patterns - focus on getting hallucinogens and stimulants right first

5. **Tree Models > Linear**: Non-linear relationships detected in pair plots - XGBoost/LightGBM will outperform logistic regression

6. **Monitor Per-Class Performance**: A model with 75% accuracy might have 0% recall on depressants - always check confusion matrix!

---

**Status:** ✅ **READY FOR FEATURE ENGINEERING & MODELING**

**Your Goal:** Macro F1-Score > 0.70 (realistic), > 0.75 (optimistic)

**Good luck! 🚀**

---

_Last Updated: 2025-11-16_  
_Data Quality Score: 93.4/100_  
_Phase: Feature Engineering Ready_
