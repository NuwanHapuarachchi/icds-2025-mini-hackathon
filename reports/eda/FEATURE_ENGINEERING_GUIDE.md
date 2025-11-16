# Feature Engineering and Selection Guide
## Drug Category Prediction - ICDS 2025 Mini-Hackathon

---

## Executive Summary

This document provides comprehensive feature engineering strategies and feature selection methodologies for the drug category prediction task. Based on EDA insights and domain knowledge, this guide outlines specific features to create, encoding strategies, and selection criteria to maximize model performance.

---

## 1. Feature Engineering Strategy

### 1.1 Interaction Features

**High-Priority Interactions** (based on domain knowledge and effect sizes):

#### 1.1.1 Risk-Taking Behavior
```python
# Impulsive × SS: Combined measure of impulsivity and novelty-seeking
df['risk_taking'] = df['Impulsive'] * df['SS']
```
- **Rationale**: Strong predictors with large effect sizes (SS: η²=0.228, Impulsive: η²=0.110)
- **Expected Impact**: Predict hallucinogen use (high risk-taking)

#### 1.1.2 Exploratory Tendency
```python
# Oscore × SS: Openness combined with sensation seeking
df['exploratory_tendency'] = df['Oscore'] * df['SS']
```
- **Rationale**: Both features highly discriminative (Oscore: η²=0.176, SS: η²=0.228)
- **Expected Impact**: Strong predictor for hallucinogen use

#### 1.1.3 Self-Control Index
```python
# Cscore × (inverse of Impulsive): Conscientiousness moderated by impulsivity
df['self_control'] = df['Cscore'] * (1 / (df['Impulsive'] + 2))  # +2 to avoid division issues
```
- **Rationale**: Combines conscientiousness (η²=0.070) with impulsivity
- **Expected Impact**: Predict stimulant use (high self-control)

#### 1.1.4 Anxious Impulsivity
```python
# Nscore × Impulsive: Emotional instability combined with poor impulse control
df['anxious_impulsivity'] = df['Nscore'] * df['Impulsive']
```
- **Rationale**: Emotional regulation through substance use
- **Expected Impact**: Predict depressant use

#### 1.1.5 Age-Personality Interactions
```python
# Age × Oscore: Openness effects across lifespan
df['age_openness'] = df['Age'] * df['Oscore']

# Age × Nscore: Emotional stability across lifespan
df['age_neuroticism'] = df['Age'] * df['Nscore']

# Age × SS: Sensation seeking changes with age
df['age_sensation'] = df['Age'] * df['SS']
```
- **Rationale**: Age is strong predictor (χ²=285.59), personality effects may vary by age
- **Expected Impact**: Capture age-specific personality-drug relationships

### 1.2 Polynomial Features

**Non-Linear Transformations**:

```python
# Capture non-linear effects for top predictors
df['oscore_squared'] = df['Oscore'] ** 2
df['cscore_squared'] = df['Cscore'] ** 2
df['ss_squared'] = df['SS'] ** 2
df['impulsive_squared'] = df['Impulsive'] ** 2

# Cubic terms for strongest predictors
df['oscore_cubed'] = df['Oscore'] ** 3
df['ss_cubed'] = df['SS'] ** 3
```

- **Rationale**: Pair plot analysis showed non-linear class separations
- **Expected Impact**: Capture threshold effects (e.g., extreme openness → hallucinogens)

### 1.3 Aggregation Features

#### 1.3.1 Big Five Composite Scores

```python
# Emotional Instability: High neuroticism + low conscientiousness
df['emotional_instability'] = df['Nscore'] - df['Cscore']

# Social Behavior: Extraversion + agreeableness
df['social_behavior'] = df['Escore'] + df['Ascore']

# Exploratory Profile: Openness + SS - conscientiousness
df['exploratory_profile'] = df['Oscore'] + df['SS'] - df['Cscore']

# Personality Stability: Standard deviation across Big Five
df['personality_variance'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].std(axis=1)

# Mean personality score
df['personality_mean'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].mean(axis=1)
```

#### 1.3.2 Domain-Specific Risk Profiles

```python
# Substance Use Risk Score (weighted by effect sizes)
# Weights: SS (0.40), Oscore (0.30), Impulsive (0.20), Nscore (0.10)
df['substance_risk_score'] = (
    0.40 * df['SS'] + 
    0.30 * df['Oscore'] + 
    0.20 * df['Impulsive'] + 
    0.10 * df['Nscore']
)

# Stability Index: Conscientiousness minus neuroticism
df['stability_index'] = df['Cscore'] - df['Nscore']

# Novelty Seeking Index
df['novelty_seeking'] = df['Oscore'] + df['SS'] + df['Impulsive']
```

### 1.4 Binning and Discretization

**Age Groups** (Already Categorical):
- Maintain 6 age groups

**Education Levels** (Collapse for Simplicity):
```python
# Collapse 9 education levels into 3
education_mapping = {
    # Low education: 1-3
    -2.43591: 'Low', -1.73790: 'Low', -1.43719: 'Low',
    # Medium education: 4-6
    -1.22751: 'Medium', -0.61113: 'Medium', -0.05921: 'Medium',
    # High education: 7-9
    0.45468: 'High', 1.16365: 'High', 1.98437: 'High'
}
df['education_level'] = df['Education'].map(education_mapping)
```

**Personality Quintiles** (For Tree Models):
```python
# Create quintile bins for top predictors
for feature in ['Oscore', 'Cscore', 'SS', 'Impulsive']:
    df[f'{feature}_quintile'] = pd.qcut(df[feature], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
```

### 1.5 Feature Scaling and Transformation

**For Linear Models**:
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# StandardScaler for well-behaved features
standard_features = ['Nscore', 'Oscore', 'Ascore', 'SS']
scaler_standard = StandardScaler()
df[standard_features] = scaler_standard.fit_transform(df[standard_features])

# RobustScaler for features with outliers
robust_features = ['Escore', 'Cscore', 'Impulsive']
scaler_robust = RobustScaler()
df[robust_features] = scaler_robust.fit_transform(df[robust_features])
```

**For Tree-Based Models**:
- No scaling needed (tree models are scale-invariant)

**Skewness Correction** (If Needed):
```python
from scipy.stats import boxcox

# Log transformation for positive skew (if data permits)
df['impulsive_log'] = np.log1p(df['Impulsive'] - df['Impulsive'].min() + 1)

# Box-Cox transformation (requires positive values)
df['impulsive_boxcox'], lambda_param = boxcox(df['Impulsive'] - df['Impulsive'].min() + 1)
```

---

## 2. Categorical Encoding Strategies

### 2.1 Binary Features

**Gender** (Already Binary):
```python
# Binary encoding: Male=1, Female=0
df['gender_binary'] = df['Gender'].map({0.48246: 1, -0.48246: 0})
```

### 2.2 Ordinal Features

**Age** (Ordered Categories):
```python
# Maintain ordinal encoding (already encoded)
# Values: -0.95197 (18-24) to 2.59171 (65+)
# Already reflects age order
```

**Education** (Ordered Categories):
```python
# Use original encoding (already ordered) or label encoding
from sklearn.preprocessing import LabelEncoder

# Option 1: Keep original encoding (already ordered)
# No action needed

# Option 2: Label encoding (0-8)
le_education = LabelEncoder()
df['education_encoded'] = le_education.fit_transform(df['Education'])
```

### 2.3 Nominal Features (No Inherent Order)

**Country** (7 Categories):

**Option 1: One-Hot Encoding**
```python
# One-hot encoding for country
country_dummies = pd.get_dummies(df['Country'], prefix='country', drop_first=True)
df = pd.concat([df, country_dummies], axis=1)
```
- **Pros**: No ordinal assumptions, standard approach
- **Cons**: Adds 6 features, may increase dimensionality

**Option 2: Target Encoding** (Recommended)
```python
from category_encoders import TargetEncoder

# Target encoding (mean of target per category)
te_country = TargetEncoder(cols=['Country'])
df['country_encoded'] = te_country.fit_transform(df['Country'], df['drug_category'])
```
- **Pros**: Single feature, captures category-target relationship
- **Cons**: Risk of overfitting (use cross-validation)

**Ethnicity** (7 Categories):

**Option 1: One-Hot Encoding**
```python
ethnicity_dummies = pd.get_dummies(df['Ethnicity'], prefix='ethnicity', drop_first=True)
df = pd.concat([df, ethnicity_dummies], axis=1)
```

**Option 2: Target Encoding**
```python
te_ethnicity = TargetEncoder(cols=['Ethnicity'])
df['ethnicity_encoded'] = te_ethnicity.fit_transform(df['Ethnicity'], df['drug_category'])
```

**Recommendation**:
- Use **target encoding** for Country and Ethnicity (reduces dimensionality)
- Use **one-hot encoding** if interpretability is critical
- Apply cross-validation when using target encoding to prevent overfitting

### 2.4 Target Encoding with Cross-Validation

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, feature, target, n_folds=5):
    """Target encoding with cross-validation to prevent overfitting"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    encoded_feature = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        # Calculate mean target per category in training fold
        target_mean = df.iloc[train_idx].groupby(feature)[target].mean()
        
        # Map to validation fold
        encoded_feature[val_idx] = df.iloc[val_idx][feature].map(target_mean)
        
        # Handle unseen categories with global mean
        global_mean = df.iloc[train_idx][target].mean()
        encoded_feature[val_idx] = np.where(
            np.isnan(encoded_feature[val_idx]), 
            global_mean, 
            encoded_feature[val_idx]
        )
    
    return encoded_feature

# Apply target encoding with CV
df['country_target_encoded'] = target_encode_cv(df, 'Country', 'drug_category')
df['ethnicity_target_encoded'] = target_encode_cv(df, 'Ethnicity', 'drug_category')
```

---

## 3. Complete Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def engineer_features(df, training=True):
    """
    Complete feature engineering pipeline
    
    Parameters:
    - df: Input DataFrame (cleaned)
    - training: If True, fit encoders; if False, use pre-fitted encoders
    
    Returns:
    - DataFrame with engineered features
    """
    
    print("Feature Engineering Pipeline Started...")
    
    # 1. Interaction Features
    print("1. Creating interaction features...")
    df['risk_taking'] = df['Impulsive'] * df['SS']
    df['exploratory_tendency'] = df['Oscore'] * df['SS']
    df['self_control'] = df['Cscore'] * (1 / (df['Impulsive'] + 2))
    df['anxious_impulsivity'] = df['Nscore'] * df['Impulsive']
    df['age_openness'] = df['Age'] * df['Oscore']
    df['age_neuroticism'] = df['Age'] * df['Nscore']
    df['age_sensation'] = df['Age'] * df['SS']
    
    # 2. Polynomial Features
    print("2. Creating polynomial features...")
    df['oscore_squared'] = df['Oscore'] ** 2
    df['cscore_squared'] = df['Cscore'] ** 2
    df['ss_squared'] = df['SS'] ** 2
    df['impulsive_squared'] = df['Impulsive'] ** 2
    
    # 3. Aggregation Features
    print("3. Creating aggregation features...")
    df['emotional_instability'] = df['Nscore'] - df['Cscore']
    df['social_behavior'] = df['Escore'] + df['Ascore']
    df['exploratory_profile'] = df['Oscore'] + df['SS'] - df['Cscore']
    df['personality_variance'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].std(axis=1)
    df['personality_mean'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].mean(axis=1)
    df['substance_risk_score'] = 0.40*df['SS'] + 0.30*df['Oscore'] + 0.20*df['Impulsive'] + 0.10*df['Nscore']
    df['stability_index'] = df['Cscore'] - df['Nscore']
    df['novelty_seeking'] = df['Oscore'] + df['SS'] + df['Impulsive']
    
    # 4. Categorical Encoding (if training, fit encoders)
    print("4. Encoding categorical features...")
    # One-hot encode Gender (binary)
    df['gender_binary'] = (df['Gender'] > 0).astype(int)
    
    # Target encoding for Country and Ethnicity (requires target variable)
    # This will be done separately in model training pipeline
    
    print("Feature engineering complete!")
    print(f"Total features: {len(df.columns)}")
    
    return df

# Apply feature engineering
train_df_clean = pd.read_csv('data/data_minihackathon_train_clean.csv')
train_df_features = engineer_features(train_df_clean, training=True)

# Save feature-engineered data
train_df_features.to_csv('data/data_minihackathon_train_features.csv', index=False)
```

---

## 4. Feature Selection Strategies

### 4.1 Low-Variance Feature Removal

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < threshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")
```

### 4.2 Correlation-Based Feature Selection

```python
# Remove highly correlated features (threshold = 0.95)
def remove_correlated_features(df, threshold=0.95):
    """Remove features with correlation > threshold"""
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping highly correlated features: {to_drop}")
    return df.drop(columns=to_drop)

X_reduced = remove_correlated_features(X, threshold=0.95)
```

### 4.3 Univariate Statistical Tests

**Chi-Square Test** (Categorical Features):
```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top K features based on chi-square test
selector_chi2 = SelectKBest(chi2, k=10)
X_selected = selector_chi2.fit_transform(X_categorical, y)

# Get selected feature names
selected_features = X_categorical.columns[selector_chi2.get_support()]
```

**ANOVA F-Test** (Continuous Features):
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top K features based on ANOVA F-test
selector_anova = SelectKBest(f_classif, k=15)
X_selected = selector_anova.fit_transform(X_continuous, y)

# Get feature scores
feature_scores = pd.DataFrame({
    'Feature': X_continuous.columns,
    'F-Score': selector_anova.scores_,
    'P-Value': selector_anova.pvalues_
}).sort_values('F-Score', ascending=False)

print(feature_scores.head(15))
```

**Mutual Information**:
```python
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y, random_state=42)

# Create DataFrame with scores
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print(mi_df.head(20))

# Select top features
top_features = mi_df.head(20)['Feature'].tolist()
X_selected = X[top_features]
```

### 4.4 Model-Based Feature Selection

**Tree-Based Feature Importance**:
```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(20))

# Select top N features
top_n = 20
top_features = feature_importance.head(top_n)['Feature'].tolist()
X_selected = X[top_features]
```

**L1 Regularization (Lasso)**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Train Lasso model
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso.fit(X_train, y_train)

# Select features with non-zero coefficients
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X)

# Get selected features
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")
```

### 4.5 Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# RFE with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=15, step=1)
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print(f"RFE selected features: {list(selected_features)}")

# Get feature rankings
feature_ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print(feature_ranking)
```

### 4.6 Domain-Guided Feature Selection

**Keep Psychologically Meaningful Features**:

**Must-Keep Features** (based on EDA):
1. **SS** (Sensation Seeking): η² = 0.228 (largest effect)
2. **Oscore** (Openness): η² = 0.176 (second largest)
3. **Impulsive**: η² = 0.110 (medium effect)
4. **Cscore** (Conscientiousness): η² = 0.070 (medium effect)
5. **Age**: χ² = 285.59 (very strong association)

**Consider Dropping**:
1. **Escore** (Extraversion): p = 0.57 (not significant)

**Hybrid Approach**:
```python
# Combine domain knowledge with statistical selection
must_keep = ['SS', 'Oscore', 'Impulsive', 'Cscore', 'Age']
can_drop = ['Escore']

# Run feature selection on remaining features
other_features = [f for f in X.columns if f not in must_keep and f not in can_drop]

# Use mutual information on other features
mi_scores = mutual_info_classif(X[other_features], y, random_state=42)
mi_df = pd.DataFrame({'Feature': other_features, 'MI_Score': mi_scores})
top_other = mi_df.nlargest(10, 'MI_Score')['Feature'].tolist()

# Combine must-keep with top others
final_features = must_keep + top_other
X_selected = X[final_features]
```

---

## 5. Feature Selection Workflow

### 5.1 Complete Pipeline

```python
def select_features(X, y, method='hybrid', n_features=20):
    """
    Complete feature selection pipeline
    
    Parameters:
    - X: Feature DataFrame
    - y: Target variable
    - method: 'variance', 'correlation', 'mutual_info', 'tree', 'rfe', 'hybrid'
    - n_features: Number of features to select
    
    Returns:
    - Selected features
    """
    
    if method == 'variance':
        # Low-variance removal
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        selected = X.columns[selector.get_support()]
    
    elif method == 'mutual_info':
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X.columns, 'Score': mi_scores})
        selected = mi_df.nlargest(n_features, 'Score')['Feature'].tolist()
    
    elif method == 'tree':
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        })
        selected = importance_df.nlargest(n_features, 'Importance')['Feature'].tolist()
    
    elif method == 'rfe':
        # Recursive Feature Elimination
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=n_features)
        rfe.fit(X, y)
        selected = X.columns[rfe.support_]
    
    elif method == 'hybrid':
        # Domain-guided + statistical
        must_keep = ['SS', 'Oscore', 'Impulsive', 'Cscore', 'Age']
        can_drop = ['Escore']
        
        other_features = [f for f in X.columns if f not in must_keep and f not in can_drop]
        
        # Mutual information on others
        mi_scores = mutual_info_classif(X[other_features], y, random_state=42)
        mi_df = pd.DataFrame({'Feature': other_features, 'Score': mi_scores})
        top_other = mi_df.nlargest(n_features - len(must_keep), 'Score')['Feature'].tolist()
        
        selected = must_keep + top_other
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Selected {len(selected)} features using {method} method")
    return selected

# Apply feature selection
selected_features = select_features(X_train, y_train, method='hybrid', n_features=25)
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
```

---

## 6. Feature Engineering Validation

### 6.1 Impact Assessment

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def evaluate_feature_impact(X_before, X_after, y):
    """Compare model performance before and after feature engineering"""
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Before feature engineering
    scores_before = cross_val_score(rf, X_before, y, cv=5, scoring='f1_macro')
    mean_before = scores_before.mean()
    std_before = scores_before.std()
    
    # After feature engineering
    scores_after = cross_val_score(rf, X_after, y, cv=5, scoring='f1_macro')
    mean_after = scores_after.mean()
    std_after = scores_after.std()
    
    print("="*60)
    print("FEATURE ENGINEERING IMPACT ASSESSMENT")
    print("="*60)
    print(f"Before: F1-Macro = {mean_before:.4f} ± {std_before:.4f}")
    print(f"After:  F1-Macro = {mean_after:.4f} ± {std_after:.4f}")
    print(f"Improvement: {((mean_after - mean_before) / mean_before) * 100:.2f}%")
    print("="*60)

# Run evaluation
evaluate_feature_impact(X_original, X_engineered, y)
```

---

## 7. Summary and Recommendations

### 7.1 Recommended Feature Set

**Core Features** (Must Include):
1. SS (Sensation Seeking)
2. Oscore (Openness)
3. Impulsive
4. Cscore (Conscientiousness)
5. Age

**Engineered Features** (High Priority):
6. risk_taking (Impulsive × SS)
7. exploratory_tendency (Oscore × SS)
8. self_control (Cscore / Impulsive)
9. substance_risk_score (weighted combo)
10. oscore_squared
11. ss_squared

**Additional Features** (Based on Feature Selection):
12-25. Top features from mutual information/tree importance

### 7.2 Expected Impact

- **Baseline (original features)**: F1-Macro ~0.60-0.65
- **With feature engineering**: F1-Macro ~0.70-0.75
- **With feature engineering + selection**: F1-Macro ~0.75-0.80

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Related Documents**: EXPLORATORY_DATA_ANALYSIS_REPORT.md, DATA_QUALITY_ASSESSMENT_GUIDE.md

