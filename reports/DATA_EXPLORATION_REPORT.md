# Data Exploration and Understanding Report
## Drug Category Prediction - ICDS 2025 Mini-Hackathon

---

## 1. Problem Understanding

### 1.1 Objective
Predict the **drug_category** (target variable) for patients based on their mental health conditions and demographic characteristics. This is a **multi-class classification problem** with 3 distinct drug categories:
- Depressants
- Hallucinogens
- Stimulants

### 1.2 Business Context
The dataset focuses on the relationship between mental health conditions and drug usage patterns. The goal is to understand how personality traits, demographic factors, and psychological characteristics correlate with different types of drug usage, which can inform targeted intervention and treatment strategies.

### 1.3 Data Source
The dataset is derived from a drug consumption classification study that collected data from participants regarding their personality traits (Big Five personality model), impulsiveness, sensation-seeking behavior, and demographic information.

---

## 2. Dataset Overview

### 2.1 Dataset Dimensions

| Dataset | Records | Features | Target Variable |
|---------|---------|----------|-----------------|
| Training | 1,500 | 13 + 1 target | drug_category |
| Test | 377 | 13 | Not provided |

### 2.2 Feature Categories

The dataset contains **13 predictor variables** grouped into three main categories:

**Demographic Features (6 features):**
- ID: Record identifier
- Age: Age group (6 categories)
- Gender: Male/Female
- Education: Education level (9 categories)
- Country: Country of residence (7 categories)
- Ethnicity: Ethnicity group (7 categories)

**Personality Traits - Big Five Model (5 features):**
- Nscore: Neuroticism
- Escore: Extraversion
- Oscore: Openness to experience
- Ascore: Agreeableness
- Cscore: Conscientiousness

**Behavioral Measures (2 features):**
- Impulsive: Impulsiveness score (BIS-11)
- SS: Sensation seeking score (ImpSS)

---

## 3. Target Variable Analysis

### 3.1 Class Distribution

| Drug Category | Count | Percentage |
|---------------|-------|------------|
| Depressants | 242 | 16.1% |
| Hallucinogens | 691 | 46.1% |
| Stimulants | 567 | 37.8% |

### 3.2 Class Balance Assessment

**Key Findings:**
- **Imbalanced dataset** with significant class disparity
- Hallucinogens is the **majority class** (46.1%)
- Depressants is the **minority class** (16.1%)
- Imbalance ratio: approximately 2.9:1 (majority to minority)

**Implications:**
- Risk of model bias toward majority class (Hallucinogens)
- May require class balancing techniques (SMOTE, class weights, oversampling/undersampling)
- Evaluation metrics should include precision, recall, F1-score per class, not just accuracy
- Consider stratified sampling for train/validation splits

![Target Distribution](../visualizations/target_distribution.png)

---

## 4. Data Quality Assessment

### 4.1 Missing Values Analysis

| Feature | Missing Count | Missing Percentage |
|---------|--------------|-------------------|
| Education | 20 | 1.33% |
| Age | 2 | 0.13% |
| Country | 2 | 0.13% |
| Gender | 1 | 0.07% |
| Nscore | 1 | 0.07% |
| Escore | 1 | 0.07% |
| Oscore | 1 | 0.07% |
| Ascore | 1 | 0.07% |
| Cscore | 1 | 0.07% |
| Impulsive | 1 | 0.07% |

**Key Findings:**
- **Minimal missing data** overall (< 2% for all features)
- Education has the highest missing rate (1.33%)
- Personality traits and behavioral measures have negligible missing values
- Total missing data is manageable

**Recommended Strategies:**
- Simple imputation with median for continuous features
- Mode imputation for categorical features
- KNN imputation for personality traits to preserve relationships
- Consider creating "missing" indicator features if patterns exist

![Missing Values](../visualizations/missing_values.png)

---

## 5. Feature Analysis

### 5.1 Demographic Features

#### Age Distribution

| Age Group | Count | Percentage | Depressants | Hallucinogens | Stimulants |
|-----------|-------|------------|-------------|---------------|------------|
| 18-24 | 504 | 33.6% | 13.7% | 72.4% | 13.9% |
| 25-34 | 381 | 25.4% | 14.7% | 44.9% | 40.4% |
| 35-44 | 282 | 18.8% | 20.6% | 31.2% | 48.2% |
| 45-54 | 238 | 15.9% | 16.8% | 20.6% | 62.6% |
| 55-64 | 79 | 5.3% | 20.3% | 21.5% | 58.2% |
| 65+ | 14 | 0.9% | 21.4% | 0.0% | 78.6% |

**Key Insights:**
- Younger participants (18-24) predominantly use **Hallucinogens** (72.4%)
- Older participants (45+) predominantly use **Stimulants** (60-80%)
- Age appears to be a **strong predictor** of drug category
- Clear age-based usage patterns exist

#### Gender Distribution

| Gender | Count | Percentage |
|--------|-------|------------|
| Male | 753 | 50.2% |
| Female | 746 | 49.7% |

**Key Insights:**
- Nearly **perfect gender balance** in the dataset
- Minimal gender bias concerns

#### Country Distribution

| Country | Count | Percentage |
|---------|-------|------------|
| UK | 824 | 54.9% |
| USA | 456 | 30.4% |
| Other | 89 | 5.9% |
| Canada | 74 | 4.9% |
| Australia | 37 | 2.5% |
| Ireland | 15 | 1.0% |
| New Zealand | 3 | 0.2% |

**Key Insights:**
- Strong **geographic bias** toward UK (54.9%) and USA (30.4%)
- 85% of data from two countries
- May limit generalizability to other regions

#### Ethnicity Distribution

| Ethnicity | Count | Percentage |
|-----------|-------|------------|
| White | 1,366 | 91.1% |
| Other | 51 | 3.4% |
| Black | 28 | 1.9% |
| Asian | 21 | 1.4% |
| Mixed-White/Black | 17 | 1.1% |
| Mixed-White/Asian | 14 | 0.9% |
| Mixed-Black/Asian | 3 | 0.2% |

**Key Insights:**
- **Extreme ethnic imbalance** with 91.1% White participants
- Limited representation of other ethnic groups
- May affect model fairness and generalizability

### 5.2 Personality Traits (Big Five Model)

#### Statistical Summary by Drug Category

| Trait | Depressants (Mean) | Hallucinogens (Mean) | Stimulants (Mean) |
|-------|-------------------|---------------------|------------------|
| Neuroticism | 0.082 | 0.111 | -0.148 |
| Extraversion | -0.028 | -0.008 | 0.077 |
| Openness | -0.027 | **0.419** | **-0.491** |
| Agreeableness | -0.114 | -0.182 | **0.219** |
| Conscientiousness | -0.051 | **-0.264** | **0.335** |

**Key Insights:**

**Neuroticism (Nscore):**
- Hallucinogen and Depressant users show higher neuroticism
- Stimulant users have lower neuroticism scores
- Suggests emotional instability correlates with certain drug types

**Extraversion (Escore):**
- Relatively similar across all categories
- Stimulant users slightly more extraverted
- Weaker discriminative power

**Openness (Oscore):**
- **Strongest differentiator** among personality traits
- Hallucinogen users: High openness (0.419)
- Stimulant users: Low openness (-0.491)
- Clear separation suggests strong predictive value

**Agreeableness (Ascore):**
- Stimulant users: More agreeable (0.219)
- Hallucinogen users: Less agreeable (-0.182)
- Moderate discriminative power

**Conscientiousness (Cscore):**
- **Strong differentiator**
- Stimulant users: High conscientiousness (0.335)
- Hallucinogen users: Low conscientiousness (-0.264)
- Suggests organized individuals prefer stimulants

![Personality by Category](../visualizations/personality_by_category.png)

### 5.3 Behavioral Measures

**Impulsiveness:**
- Measured using BIS-11 scale
- Range: -2.555 to 10.000 (contains outliers)
- Mean: 0.006, Std: 0.988
- 7 outliers detected (0.5%)

**Sensation Seeking (SS):**
- Measured using ImpSS scale
- Range: -2.078 to 1.922
- Mean: 0.0002, Std: 0.966
- No significant outliers

---

## 6. Data Visualization Insights

### 6.1 Feature Distributions

![Feature Distributions](../visualizations/feature_distributions.png)

**Key Observations:**
- Age: Left-skewed (younger population dominant)
- Gender: Bimodal distribution (two distinct values)
- Education: Multimodal with concentration around higher education
- Personality traits: Generally normal distributions with slight variations
- Impulsive and SS: Approximately normal distributions

### 6.2 Correlation Analysis

![Correlation Matrix](../visualizations/correlation_matrix.png)

**Strong Correlations Identified:**
- Minimal multicollinearity among features
- Personality traits show independence (as expected from Big Five model)
- No concerning feature redundancy

**Weak/No Correlations:**
- Demographics and personality traits are largely independent
- Behavioral measures (Impulsive, SS) show weak correlation with personality

### 6.3 Demographic Patterns

![Demographics Analysis](../visualizations/demographics_analysis.png)

**Key Patterns:**
- Clear age-based drug usage trends
- Education level shows variation across drug categories
- Gender distribution relatively balanced across categories
- Impulsiveness varies by drug category

---

## 7. Outlier Detection

| Feature | Outliers | Percentage |
|---------|----------|------------|
| Ascore | 24 | 1.6% |
| Nscore | 16 | 1.1% |
| Escore | 12 | 0.8% |
| Impulsive | 7 | 0.5% |
| Oscore | 6 | 0.4% |
| Cscore | 5 | 0.3% |

**Outlier Strategy:**
- Low outlier percentage (< 2% for all features)
- Consider capping extreme values rather than removal
- Use robust scalers (RobustScaler) to minimize outlier impact
- Tree-based models naturally handle outliers well

---

## 8. Data Type Classification

### 8.1 Continuous Features (Standardized)
All features are **encoded as continuous variables** with standardized values:
- Personality traits: NEO-FFI-R scores (standardized)
- Behavioral measures: BIS-11 and ImpSS scores (standardized)
- Demographics: Categorical variables encoded as continuous values

### 8.2 Original Categorical Features
Despite continuous encoding, the following are **inherently categorical**:
- Age: 6 ordered categories
- Gender: 2 categories
- Education: 9 ordered categories
- Country: 7 nominal categories
- Ethnicity: 7 nominal categories

**Modeling Implications:**
- Current encoding suitable for tree-based models and distance-based algorithms
- Consider one-hot encoding for linear models
- Ordinal features (Age, Education) maintain meaningful order
- Nominal features (Country, Ethnicity) may benefit from target encoding

---

## 9. Key Findings Summary

### 9.1 Critical Insights

**Target Variable:**
- Significant class imbalance (46% Hallucinogens, 16% Depressants)
- Requires balanced evaluation metrics and potential resampling

**Data Quality:**
- High-quality dataset with minimal missing values (< 2%)
- Low outlier presence (< 2% per feature)
- No significant data integrity issues

**Predictive Features:**
- **Age**: Strongest demographic predictor (clear usage patterns)
- **Openness**: Strongest personality predictor (0.42 vs -0.49)
- **Conscientiousness**: Strong discriminator (0.34 vs -0.26)
- **Impulsiveness**: Moderate behavioral predictor

**Data Biases:**
- Geographic bias (85% UK/USA)
- Ethnic bias (91% White)
- Age bias (59% under 35)

### 9.2 Feature Importance Hypotheses

**High Importance (Expected):**
1. Age
2. Openness to experience (Oscore)
3. Conscientiousness (Cscore)
4. Education

**Moderate Importance:**
1. Neuroticism (Nscore)
2. Agreeableness (Ascore)
3. Impulsiveness
4. Sensation Seeking

**Low Importance (Expected):**
1. Gender (balanced across categories)
2. Extraversion (minimal variation)
3. Country (dominated by UK)
4. Ethnicity (dominated by White)

---

## 10. Recommendations for Model Development

### 10.1 Data Preprocessing

**Missing Value Handling:**
- KNN imputation for personality traits (k=5)
- Median imputation for continuous demographics
- Mode imputation for categorical features

**Feature Scaling:**
- StandardScaler for distance-based models (KNN, SVM)
- RobustScaler if outliers impact performance
- No scaling needed for tree-based models

**Feature Engineering:**
- Create age groups instead of continuous values
- Interaction features: Age × Personality traits
- Polynomial features for personality combinations
- Target encoding for Country and Ethnicity

**Class Imbalance:**
- Use stratified cross-validation
- Apply class weights in models
- Consider SMOTE or hybrid sampling approaches
- Evaluate with F1-score, not just accuracy

### 10.2 Model Selection Strategy

**Recommended Models:**

**Tree-Based Models (Preferred):**
- Random Forest: Handles categorical features, imbalanced data
- XGBoost: Superior performance, handles missing values
- LightGBM: Fast training, categorical feature support
- CatBoost: Native categorical handling, robust to outliers

**Distance-Based Models:**
- KNN with proper scaling
- SVM with RBF kernel

**Ensemble Approaches:**
- Stacking: Combine multiple model predictions
- Voting classifier: Aggregate tree-based and distance-based models

**Not Recommended:**
- Linear models (non-linear relationships observed)
- Naive Bayes (features not independent)

### 10.3 Evaluation Strategy

**Metrics:**
- **Primary**: Macro F1-score (accounts for class imbalance)
- **Secondary**: Weighted F1-score, Per-class precision/recall
- **Monitoring**: Confusion matrix, ROC-AUC per class

**Cross-Validation:**
- Stratified K-Fold (k=5 or k=10)
- Ensures balanced class representation
- Reduces overfitting risk

**Validation Approach:**
- 80-20 train-validation split (stratified)
- Hold-out test set for final evaluation
- Monitor for overfitting with learning curves

### 10.4 Feature Selection

**Methods:**
- Recursive Feature Elimination (RFE)
- Feature importance from tree models
- Mutual information scores
- Correlation-based feature selection

**Expected Outcome:**
- Reduce dimensionality if needed
- Improve model interpretability
- Potentially enhance generalization

---

## 11. Data Collection and Methodology Notes

### 11.1 Data Collection Method
- Participants completed validated psychological assessments
- NEO-FFI-R for personality traits (Big Five model)
- BIS-11 for impulsiveness measurement
- ImpSS for sensation-seeking behavior
- Self-reported demographic information

### 11.2 Feature Encoding
All features use **standardized continuous encoding**:
- Categorical variables mapped to specific numeric values
- Personality traits standardized using NEO-FFI-R norms
- Maintains statistical properties while enabling numeric computation

### 11.3 Ethical Considerations
- Data anonymized (ID cannot trace to participants)
- Mental health and drug usage data requires careful handling
- Model predictions should inform, not replace, clinical judgment
- Consider fairness metrics given demographic biases

---

## 12. Next Steps

### Phase 1: Data Preparation
1. Handle missing values using recommended strategies
2. Perform feature engineering (interactions, polynomials)
3. Apply appropriate scaling based on model choice
4. Create stratified train/validation splits

### Phase 2: Baseline Model
1. Train simple Random Forest model
2. Establish baseline performance metrics
3. Identify initial feature importance
4. Validate with cross-validation

### Phase 3: Advanced Modeling
1. Experiment with XGBoost, LightGBM, CatBoost
2. Hyperparameter tuning using GridSearch/RandomSearch
3. Address class imbalance with SMOTE/class weights
4. Ensemble multiple top-performing models

### Phase 4: Model Evaluation
1. Final validation on hold-out set
2. Analyze confusion matrix for error patterns
3. Calculate per-class performance metrics
4. Assess model fairness across demographic groups

### Phase 5: Deployment Preparation
1. Generate predictions for test set
2. Format submission file correctly
3. Document model methodology and assumptions
4. Prepare model interpretation and insights

---

## 13. Conclusion

The dataset presents a well-structured **multi-class classification problem** with clear patterns between psychological characteristics and drug usage categories. Key success factors include:

**Strengths:**
- High-quality data with minimal missing values
- Clear discriminative features (Age, Openness, Conscientiousness)
- Validated psychological assessment tools
- Sufficient sample size (1,500 training records)

**Challenges:**
- Class imbalance requiring careful handling
- Geographic and ethnic biases limiting generalizability
- Need for sophisticated feature engineering
- Requires balanced evaluation metrics

**Expected Model Performance:**
- Baseline accuracy: 60-65%
- Optimized accuracy: 75-85%
- Focus on macro F1-score for fair evaluation across classes

With proper preprocessing, feature engineering, and model selection, achieving strong predictive performance while maintaining model interpretability is feasible. The combination of demographic and psychological features provides a rich foundation for accurate drug category prediction.

---

**Report Generated:** 2025-11-16  
**Dataset Version:** ICDS 2025 Mini-Hackathon Training Data  
**Total Training Records:** 1,500  
**Total Test Records:** 377  
**Target Classes:** 3 (Depressants, Hallucinogens, Stimulants)
