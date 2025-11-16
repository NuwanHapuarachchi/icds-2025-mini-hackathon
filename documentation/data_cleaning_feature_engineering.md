# Data Cleaning and Feature Engineering Documentation

**ICDS 2025 Mini-Hackathon - Drug Category Prediction**

---

## Executive Summary

This document details the comprehensive data cleaning and feature engineering pipeline implemented for the drug category prediction task. The pipeline successfully transformed the raw dataset into a high-quality, feature-rich dataset ready for machine learning model development.

### Key Achievements
- ✅ **100% data completeness** - All missing values and anomalies handled
- ✅ **72 engineered features** - Expanded from 13 original features (454% increase)
- ✅ **Domain-specific features** - 10 psychological risk indicators created
- ✅ **No data leakage** - All features are measurement-based, no temporal issues
- ✅ **Standardized scaling** - All numeric features normalized using StandardScaler

---

## 1. Data Cleaning Pipeline

### 1.1 Anomaly Detection and Treatment

**Extreme Anomalies Identified:**
- **Escore**: 1 value > 10 (expected range: -3.27 to 3.27)
- **Cscore**: 1 value < -5 (expected range: -3.46 to 3.46)
- **Impulsive**: 1 value > 5 (expected range: -2.56 to 2.90)

**Treatment Strategy:**
- Flagged extreme values (|Z| > 5) as NaN
- Total anomalies fixed: **3**

**Rationale:** Extreme values beyond 5 standard deviations likely represent data entry errors or measurement artifacts. Replacing with NaN allows KNN imputation to estimate reasonable values based on similar observations.

### 1.2 Missing Value Treatment

**Missing Value Pattern Analysis:**

| Feature    | Missing Count | Missing % | Missingness Type |
|------------|---------------|-----------|------------------|
| Education  | 20            | 1.33%     | MCAR             |
| Impulsive  | 2             | 0.13%     | MCAR             |
| Escore     | 2             | 0.13%     | MCAR             |
| Cscore     | 2             | 0.13%     | MCAR             |
| Age        | 2             | 0.13%     | MCAR             |
| Country    | 2             | 0.13%     | MCAR             |
| Nscore     | 1             | 0.07%     | MCAR             |
| Gender     | 1             | 0.07%     | MCAR             |
| Oscore     | 1             | 0.07%     | MCAR             |
| Ascore     | 1             | 0.07%     | MCAR             |

**Imputation Strategy:**

1. **Categorical Features** (Age, Gender, Education, Country, Ethnicity)
   - Method: Mode imputation using `SimpleImputer(strategy='most_frequent')`
   - Rationale: Categorical features with low missingness; mode preserves class distribution

2. **Continuous Features** (Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, SS)
   - Method: KNN imputation with k=5, distance-weighted
   - Rationale: Preserves multivariate relationships between personality traits
   - Total values imputed: 9 (including 3 anomalies)

**Missingness Assessment:**
- Pattern: Missing Completely At Random (MCAR)
- Evidence: Low percentage (<2%) and random distribution across features
- No systematic patterns suggesting MAR or MNAR

### 1.3 Outlier Treatment

**IQR Method Results (1.5 × IQR):**

| Feature    | Outlier Count | Outlier % | Treatment     |
|------------|---------------|-----------|---------------|
| Nscore     | 87            | 5.80%     | Keep          |
| Escore     | 77            | 5.13%     | Keep          |
| Oscore     | 82            | 5.47%     | Keep          |
| Ascore     | 83            | 5.53%     | Keep          |
| Cscore     | 87            | 5.80%     | Keep          |
| Impulsive  | 95            | 6.33%     | Keep          |
| SS         | 103           | 6.87%     | Keep          |

**Decision: Keep All Outliers**

**Rationale:**
1. Outlier percentages (5-7%) are within normal range for standardized psychological scales
2. Values represent valid extreme personality traits, not errors
3. Tree-based models (planned for use) are robust to outliers
4. RobustScaler will be available for linear models if needed

### 1.4 Data Quality Validation

**Post-Cleaning Validation:**
- ✅ Missing values: 0 (100% complete)
- ✅ Duplicates: 0
- ✅ Range validation: All features within ±5 SD
- ✅ Distribution preservation: Minimal changes to mean/std

**Data Quality Scorecard:**

| Dimension        | Score  | Status    |
|------------------|--------|-----------|
| Missing Values   | 100.0  | Excellent |
| Duplicates       | 100.0  | Excellent |
| Outliers         | 94.2   | Good      |
| Anomalies        | 99.8   | Good      |
| Consistency      | 100.0  | Good      |
| **OVERALL**      | **98.8** | **Excellent** |

---

## 2. Feature Engineering Pipeline

### 2.1 Domain-Specific Psychological Features (10 features)

**Risk Indicators Based on Psychological Theory:**

1. **HighRisk_Score** = Nscore + Impulsive - Cscore
   - Combines neuroticism, impulsivity, and low conscientiousness
   - Strong theoretical link to substance use risk

2. **SensationRisk_Score** = SS + Impulsive
   - Combines sensation-seeking and impulsivity
   - Behavioral disinhibition indicator

3. **EmotionalInstability** = Nscore - Cscore
   - Emotional regulation capacity
   - Negative coping predictor

4. **Conscientiousness_Deficit** = -Cscore
   - Inverted conscientiousness
   - Lack of self-control indicator

5. **Social_Risk** = Escore + Oscore
   - Social exposure and novelty-seeking
   - Environmental risk factor

6. **Behavioral_Disinhibition** = Impulsive + SS - Cscore
   - Composite disinhibition score
   - Key predictor of risky behaviors

7. **NE_Balance** = Nscore - Escore
   - Neuroticism-Extraversion balance
   - Social anxiety vs. social seeking

8. **BigFive_Mean** = mean(Nscore, Escore, Oscore, Ascore, Cscore)
   - Overall personality trait level
   - Baseline personality indicator

9. **BigFive_Std** = std(Nscore, Escore, Oscore, Ascore, Cscore)
   - Personality trait variability
   - Profile differentiation measure

10. **RiskFactor_Mean** = mean(Nscore, Impulsive, SS)
    - Average risk factor level
    - Composite risk indicator

### 2.2 Interaction Terms (11 features)

**Theory-Driven Interactions:**

**Risk Behavior Interactions:**
- Impulsive × SS
- Nscore × Impulsive
- Nscore × SS

**Personality Trait Interactions:**
- Nscore × Cscore
- Escore × Oscore
- Ascore × Cscore

**Protective vs Risk Factors:**
- Cscore × Impulsive
- Cscore × SS
- Ascore × Impulsive

**Complex Interactions:**
- Nscore × Escore
- Oscore × Impulsive

### 2.3 Polynomial Features (10 features)

**Quadratic Terms (7 features):**
- Nscore², Escore², Oscore², Ascore², Cscore², Impulsive², SS²
- Captures non-linear relationships
- U-shaped or inverted-U relationships

**Cubic Terms (3 features):**
- Nscore³, Impulsive³, SS³
- Captures complex non-linear patterns for key risk factors

### 2.4 Ratio Features (6 features)

**Risk/Protective Ratios:**
1. **Impulsive_to_Cscore_ratio** = Impulsive / (Cscore + 3)
2. **SS_to_Cscore_ratio** = SS / (Cscore + 3)
3. **Nscore_to_Ascore_ratio** = Nscore / (Ascore + 3)
4. **Risk_to_Protective_ratio** = (Nscore + Impulsive) / (Cscore + Ascore + 6)

**Personality Balance Ratios:**
5. **NE_ratio** = Nscore / (Escore + 3)
6. **OC_ratio** = Oscore / (Cscore + 3)

Note: Added constant (+3) to denominators to avoid division issues and maintain stability.

### 2.5 Binned Features (4 features)

**Discretization Strategy:**
- **Impulsive_Level**: Low (-∞, -0.5], Medium (-0.5, 0.5], High (0.5, ∞)
- **SS_Level**: Low (-∞, -0.5], Medium (-0.5, 0.5], High (0.5, ∞)
- **Nscore_Level**: Low (-∞, -0.5], Medium (-0.5, 0.5], High (0.5, ∞)
- **Cscore_Level**: Low (-∞, -0.5], Medium (-0.5, 0.5], High (0.5, ∞)

**Purpose:**
- Capture threshold effects
- Enable tree-based models to learn level-specific patterns
- Reduce sensitivity to small variations

### 2.6 Transformed Features (9 features)

**Log Transformations (7 features):**
- Applied to all continuous features: Nscore_log, Escore_log, Oscore_log, Ascore_log, Cscore_log, Impulsive_log, SS_log
- Handles skewness and reduces impact of extreme values
- Shifted negative values before transformation

**Square Root Transformations (2 features):**
- Impulsive_sqrt, SS_sqrt
- Alternative transformation for highly skewed risk factors

### 2.7 Categorical Encoding

**One-Hot Encoding (13 features):**
- Applied to: Gender, Country, Ethnicity
- Method: `OneHotEncoder(drop='first')` to avoid multicollinearity
- Handles nominal categorical variables

**Ordinal Encoding (2 features):**
- Applied to: Age, Education
- Retained original numeric encoding (represents ordered categories)

### 2.8 Feature Scaling

**Method:** StandardScaler (z-score normalization)
- Applied to all 72 numeric features
- Mean = 0, Std = 1 for each feature
- Ensures equal contribution in distance-based algorithms

---

## 3. Feature Engineering Summary

### 3.1 Feature Count Breakdown

| Category                    | Count |
|-----------------------------|-------|
| Original Features           | 13    |
| Domain-Specific             | 10    |
| Interaction Terms           | 11    |
| Polynomial Features         | 10    |
| Ratio Features              | 6     |
| Binned Features             | 4     |
| Transformed Features        | 9     |
| One-Hot Encoded             | 13    |
| **TOTAL ENGINEERED**        | **72** |

**Feature Expansion:** 13 → 72 features (454% increase)

### 3.2 Dataset Dimensions

**Training Set:**
- Original: (1500, 14) → Cleaned: (1500, 14) → Engineered: (1500, 74)
- 1500 samples, 72 features + ID + target

**Test Set:**
- Original: (377, 13) → Cleaned: (377, 13) → Engineered: (377, 73)
- 377 samples, 72 features + ID

---

## 4. Data Quality Assurance

### 4.1 Validation Checks Performed

✅ **Missing Values:** 0% - Complete dataset  
✅ **Duplicates:** None detected  
✅ **Range Validation:** All features within expected ranges  
✅ **Distribution Preservation:** Statistical properties maintained  
✅ **Data Leakage:** No target information in features  
✅ **Consistency:** All categorical values within valid ranges  
✅ **Scaling:** All numeric features standardized  

### 4.2 Class Balance (Target Variable)

| Drug Category | Count | Percentage |
|---------------|-------|------------|
| 0             | 590   | 39.3%      |
| 1             | 504   | 33.6%      |
| 2             | 406   | 27.1%      |

**Imbalance Ratio:** 1.45 (590/406)

**Assessment:** Reasonably balanced, but recommend:
- Stratified cross-validation
- Consider class weights in models
- Monitor performance across all classes

---

## 5. Files Generated

### 5.1 Data Files

```
data/
├── data_minihackathon_train_clean.csv       (1500 × 14)
├── data_minihackathon_test_clean.csv        (377 × 13)
├── data_minihackathon_train_engineered.csv  (1500 × 74)
└── data_minihackathon_test_engineered.csv   (377 × 73)
```

### 5.2 Reports

```
reports/
├── dqa/
│   ├── dqa_missing_values.csv
│   ├── dqa_outliers_iqr.csv
│   ├── dqa_outliers_zscore.csv
│   ├── dqa_anomalies.csv
│   ├── dqa_consistency.csv
│   ├── dqa_class_balance.csv
│   ├── dqa_summary_statistics.csv
│   ├── dqa_scorecard.csv
│   ├── data_cleaning_comparison.csv
│   └── data_cleaning_summary.csv
├── feature_names.csv
├── feature_engineering_summary.csv
└── feature_correlations.csv
```

### 5.3 Visualizations

```
visualizations/
├── dqa_missing_values.png
├── dqa_outliers_boxplots.png
├── dqa_class_balance.png
├── dqa_scorecard.png
├── data_cleaning_comparison.png
├── data_cleaning_boxplots.png
├── feature_engineering_summary.png
└── top_feature_correlations.png
```

---

## 6. Key Decisions and Rationale

### 6.1 Why KNN Imputation for Continuous Features?

**Decision:** KNN (k=5, distance-weighted)

**Rationale:**
1. Preserves multivariate relationships between personality traits
2. More sophisticated than mean/median imputation
3. Distance-weighted ensures closer neighbors have more influence
4. k=5 provides good balance between bias and variance
5. Suitable for low missingness rate (<2%)

### 6.2 Why Keep Outliers?

**Decision:** Retain all outliers detected by IQR method

**Rationale:**
1. Outliers (5-7%) are within normal range for standardized scales
2. Represent valid extreme personality traits, not errors
3. Tree-based models are robust to outliers
4. Domain knowledge: Extreme traits are clinically meaningful
5. Could indicate high-risk individuals (valuable signal)

### 6.3 Why Create Domain-Specific Features?

**Decision:** 10 psychological risk indicators

**Rationale:**
1. Leverage domain knowledge from substance use research
2. Well-established risk factors in psychology literature
3. Composite scores often more predictive than individual traits
4. Capture theoretical constructs (e.g., behavioral disinhibition)
5. Improve model interpretability

### 6.4 Why Use StandardScaler?

**Decision:** StandardScaler over MinMaxScaler or RobustScaler

**Rationale:**
1. Data already standardized (z-scores from psychological tests)
2. Assumes approximately normal distributions (valid for Big Five)
3. Preserves outlier information (unlike RobustScaler)
4. Suitable for most ML algorithms
5. Maintains interpretability (units = standard deviations)

---

## 7. Next Steps for Model Development

### 7.1 Recommended Modeling Approach

**Priority 1: Ensemble Methods**
- XGBoost, LightGBM, CatBoost
- Random Forest, Extra Trees
- Robust to outliers, handles non-linear relationships

**Priority 2: Linear Models (for interpretability)**
- Logistic Regression with L1/L2 regularization
- Use RobustScaler if needed
- Feature selection via regularization

**Priority 3: Neural Networks**
- Multi-layer perceptron for complex patterns
- Already scaled and ready for deep learning

### 7.2 Cross-Validation Strategy

- **Method:** Stratified K-Fold (k=5 or 10)
- **Rationale:** Preserves class distribution across folds
- **Metrics:** Accuracy, F1-score (macro/weighted), AUC-ROC

### 7.3 Feature Selection Considerations

**Current Feature Count:** 72

**Options:**
1. **Use all features** - Tree-based models handle high dimensionality
2. **Regularization-based** - L1 (Lasso) for automatic selection
3. **Importance-based** - Use feature_importances_ from tree models
4. **Correlation-based** - Remove highly correlated features (|r| > 0.95)

**Recommendation:** Start with all features for tree-based models, apply selection for linear models if needed.

### 7.4 Hyperparameter Tuning

**Suggested Tools:**
- Optuna for Bayesian optimization
- GridSearchCV/RandomizedSearchCV for systematic search
- Cross-validation for unbiased evaluation

### 7.5 Model Evaluation

**Metrics to Track:**
- **Accuracy:** Overall correctness
- **F1-Score (Macro):** Equal weight to all classes
- **F1-Score (Weighted):** Account for class imbalance
- **Confusion Matrix:** Class-specific performance
- **AUC-ROC (Multiclass):** Discrimination ability

---

## 8. Potential Issues and Mitigation

### 8.1 Curse of Dimensionality

**Issue:** 72 features for 1500 samples (ratio: 1:21)

**Mitigation:**
- Feature selection if needed
- Regularization (L1/L2)
- Tree-based models (robust to high dimensions)
- Cross-validation to detect overfitting

### 8.2 Feature Correlation

**Issue:** Engineered features may be highly correlated

**Mitigation:**
- Tree-based models handle multicollinearity
- For linear models: VIF analysis and feature selection
- Regularization (Ridge/Elastic Net)

### 8.3 Class Imbalance (Mild)

**Issue:** Imbalance ratio 1.45

**Mitigation:**
- Stratified sampling ✅ (recommended)
- Class weights in models
- Monitor per-class performance
- SMOTE if needed (likely unnecessary)

---

## 9. Reproducibility

### 9.1 Random Seeds

**Not Set in Current Pipeline**

**Recommendation for Model Development:**
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

### 9.2 Dependencies

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
```

### 9.3 Execution Order

1. `data_quality_assessment.py` (analysis only)
2. `data_cleaning.py` (produces cleaned data)
3. `feature_engineering.py` (produces engineered features)
4. Model development scripts (next step)

---

## 10. Conclusion

The data cleaning and feature engineering pipeline has successfully prepared a high-quality dataset for machine learning model development. Key achievements include:

✅ **Complete data quality** with 98.8% overall score  
✅ **Rich feature set** with 72 engineered features (454% expansion)  
✅ **Domain expertise integration** through psychological risk indicators  
✅ **No data leakage** and proper train-test separation  
✅ **Comprehensive documentation** of all decisions and rationale  

The dataset is now ready for model development, with clear recommendations for algorithms, validation strategies, and evaluation metrics. The extensive feature engineering should provide strong predictive signal for drug category classification.

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-16  
**Author:** Data Science Pipeline  
**Status:** Ready for Model Development
