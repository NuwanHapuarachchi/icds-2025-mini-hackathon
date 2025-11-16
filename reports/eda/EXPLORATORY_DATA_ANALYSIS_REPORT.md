# Exploratory Data Analysis Report
## Drug Category Prediction - ICDS 2025 Mini-Hackathon

---

## Executive Summary

This report presents a comprehensive exploratory data analysis of the drug category prediction dataset. The analysis includes univariate distributions, bivariate relationships, multivariate correlations, statistical significance tests, and data quality assessment. Key findings reveal strong discriminative features (Openness, Sensation Seeking, Conscientiousness) with large effect sizes, class imbalance requiring careful handling, and minimal data quality issues. The analysis provides actionable insights for data cleaning, feature engineering, and model development.

---

## Table of Contents

1. [Univariate Analysis](#1-univariate-analysis)
2. [Bivariate Analysis](#2-bivariate-analysis)
3. [Multivariate Analysis](#3-multivariate-analysis)
4. [Statistical Significance Tests](#4-statistical-significance-tests)
5. [Data Quality Assessment](#5-data-quality-assessment)
6. [Domain Research Insights](#6-domain-research-insights)
7. [Feature Engineering Recommendations](#7-feature-engineering-recommendations)
8. [Next Steps](#8-next-steps)

---

## 1. Univariate Analysis

### 1.1 Continuous Features Distribution

**Personality Traits (Big Five Model) and Behavioral Measures:**

| Feature | Mean | Std Dev | Min | Max | Skewness | Kurtosis | Interpretation |
|---------|------|---------|-----|-----|----------|----------|----------------|
| **Nscore** | 0.009 | 1.003 | -3.46 | 3.27 | -0.017 | -0.011 | Symmetric, normal distribution |
| **Escore** | 0.021 | 1.622 | -3.27 | **50.00** | 19.54 | 601.72 | **Severe positive skew, extreme outlier** |
| **Oscore** | 0.003 | 0.988 | -2.86 | 2.90 | 0.042 | -0.143 | Symmetric, slightly platykurtic |
| **Ascore** | -0.019 | 0.991 | -3.46 | 3.46 | 0.024 | 0.065 | Symmetric, normal distribution |
| **Cscore** | -0.003 | 1.032 | **-10.00** | 3.46 | -0.585 | 5.399 | **Negative skew, extreme outlier** |
| **Impulsive** | 0.006 | 0.988 | -2.56 | **10.00** | 0.766 | 6.358 | **Positive skew, extreme outlier** |
| **SS** | 0.000 | 0.966 | -2.08 | 1.92 | -0.048 | -0.450 | Symmetric, normal distribution |

**Key Findings:**

1. **Three features contain extreme outliers** that violate standardization assumptions:
   - Escore: max = 50.00 (30+ standard deviations above mean)
   - Cscore: min = -10.00 (10 standard deviations below mean)
   - Impulsive: max = 10.00 (10 standard deviations above mean)

2. **Most features follow approximately normal distributions** (Nscore, Oscore, Ascore, SS)

3. **Standardization is generally effective**, with values within ±3 SD range for most features

**Visualizations:**
- Histograms with mean/median lines for all continuous features
- Box plots showing quartiles and outliers
- See: `eda_univariate_continuous.png`

### 1.2 Categorical Features Distribution

**Demographic Features:**

| Feature | Unique Values | Most Common Category | Percentage | Imbalance |
|---------|---------------|----------------------|------------|-----------|
| **Age** | 6 | 18-24 years | ~34% | Moderate (young skew) |
| **Gender** | 2 | Male | ~50% | Balanced |
| **Education** | 9 | Some college | ~25% | Moderate |
| **Country** | 7 | UK | ~55% | High (geographic bias) |
| **Ethnicity** | 7 | White | ~91% | Severe (ethnic bias) |

**Key Findings:**

1. **Severe geographic concentration**: 85% of data from UK + USA
2. **Severe ethnic imbalance**: 91% White participants
3. **Age skew**: 59% of participants under 35 years old
4. **Gender balance**: Nearly 50/50 split (no bias concern)

**Implications:**
- Model may not generalize well to non-Western populations
- Ethnic diversity is insufficient for fair cross-ethnic predictions
- Age distribution skews toward younger adults

**Visualizations:**
- Bar charts with count labels for all categorical features
- See: `eda_univariate_categorical.png`

---

## 2. Bivariate Analysis

### 2.1 Continuous Features vs Drug Category

**Grouped Statistics by Drug Category:**

| Feature | Depressants (Mean) | Hallucinogens (Mean) | Stimulants (Mean) | Range |
|---------|-------------------|---------------------|------------------|-------|
| **Nscore** | 0.082 | 0.111 | **-0.148** | 0.259 |
| **Escore** | -0.028 | -0.008 | 0.077 | 0.105 |
| **Oscore** | -0.027 | **0.419** | **-0.491** | **0.910** |
| **Ascore** | -0.114 | -0.182 | **0.219** | 0.401 |
| **Cscore** | -0.051 | **-0.264** | **0.335** | **0.599** |
| **Impulsive** | 0.024 | **0.326** | **-0.393** | **0.719** |
| **SS** | -0.061 | **0.467** | **-0.543** | **1.010** |

**Key Patterns:**

1. **Openness (Oscore)**: 
   - Hallucinogen users: High openness (0.419)
   - Stimulant users: Low openness (-0.491)
   - **Largest mean difference** across categories

2. **Sensation Seeking (SS)**:
   - Hallucinogen users: Very high SS (0.467)
   - Stimulant users: Very low SS (-0.543)
   - **Strongest discriminator** (1.01 point difference)

3. **Conscientiousness (Cscore)**:
   - Stimulant users: High conscientiousness (0.335)
   - Hallucinogen users: Low conscientiousness (-0.264)
   - Suggests organized individuals prefer stimulants

4. **Impulsiveness**:
   - Hallucinogen users: More impulsive (0.326)
   - Stimulant users: Less impulsive (-0.393)
   - Moderate discriminative power

5. **Neuroticism (Nscore)**:
   - Stimulant users: Lower neuroticism (-0.148)
   - Depressant/Hallucinogen users: Higher neuroticism (~0.10)
   - Weaker discriminator

6. **Agreeableness (Ascore)**:
   - Stimulant users: More agreeable (0.219)
   - Hallucinogen users: Less agreeable (-0.182)

7. **Extraversion (Escore)**:
   - Minimal variation across categories
   - Weakest discriminative feature

**Visualizations:**
- Violin plots showing distributions by drug category
- Box plots highlighting median and quartiles
- See: `eda_bivariate_violin.png`, `eda_bivariate_boxplot.png`

### 2.2 Categorical Features vs Drug Category

**Age Group Patterns:**

| Age Group | Depressants | Hallucinogens | Stimulants | Dominant Category |
|-----------|-------------|---------------|------------|-------------------|
| 18-24 | 13.7% | **72.4%** | 13.9% | Hallucinogens |
| 25-34 | 14.7% | 44.9% | 40.4% | Hallucinogens |
| 35-44 | 20.6% | 31.2% | **48.2%** | Stimulants |
| 45-54 | 16.8% | 20.6% | **62.6%** | Stimulants |
| 55-64 | 20.3% | 21.5% | **58.2%** | Stimulants |
| 65+ | 21.4% | 0.0% | **78.6%** | Stimulants |

**Key Insights:**

1. **Strong age-based drug usage patterns**:
   - Young adults (18-24): Overwhelmingly use hallucinogens (72%)
   - Middle-aged and older (35+): Predominantly use stimulants (50-80%)
   - Age is a **strong predictor** of drug category

2. **Gender patterns**:
   - Minimal gender differences across categories
   - Nearly balanced distribution

3. **Education patterns**:
   - Higher education correlates with specific drug categories
   - Requires detailed cross-tabulation analysis

4. **Country/Ethnicity patterns**:
   - Limited diversity restricts pattern analysis
   - UK/USA dominance may introduce geographic bias

**Visualizations:**
- Stacked bar charts showing percentage distribution
- Cross-tabulation heatmaps
- See: `eda_bivariate_demographics.png`

---

## 3. Multivariate Analysis

### 3.1 Correlation Matrix (Continuous Features)

**Inter-Feature Correlations:**

| Feature Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| All Big Five pairs | -0.15 to +0.15 | **Minimal multicollinearity** (expected from Big Five model) |
| Impulsive × SS | ~0.10-0.15 | Weak positive correlation |
| Nscore × Cscore | ~-0.10 | Weak negative correlation |
| All others | < 0.10 | Negligible correlation |

**Key Findings:**

1. **No significant multicollinearity** among features
2. **Personality traits are independent** as designed by Big Five model
3. **Behavioral measures** (Impulsive, SS) show weak correlation with personality
4. **No redundant features** requiring removal

**Implications:**
- All features provide unique information
- No need for multicollinearity-based feature removal
- Can use all features without VIF concerns

**Visualizations:**
- Heatmap with upper triangle masked
- Color-coded correlation strength
- See: `eda_correlation_matrix.png`

### 3.2 Feature Correlation with Target

**Correlation with Drug Categories (One-Hot Encoded):**

| Feature | Depressants | Hallucinogens | Stimulants |
|---------|-------------|---------------|------------|
| **SS** | +0.018 | **+0.375** | **-0.417** |
| **Oscore** | +0.011 | **+0.344** | **-0.374** |
| **Cscore** | +0.026 | **-0.230** | **+0.217** |
| **Impulsive** | -0.016 | **+0.226** | **-0.225** |
| **Ascore** | +0.008 | **-0.125** | **+0.125** |
| **Nscore** | -0.018 | +0.068 | -0.062 |
| **Escore** | +0.004 | -0.016 | +0.013 |

**Key Insights:**

1. **Strongest positive predictors for Hallucinogens**:
   - Sensation Seeking (SS): +0.375
   - Openness (Oscore): +0.344
   - Impulsiveness: +0.226

2. **Strongest negative predictors for Stimulants** (positive for Stimulants):
   - Sensation Seeking: -0.417 (low SS → stimulants)
   - Openness: -0.374 (low openness → stimulants)
   - Conscientiousness: +0.217 (high C → stimulants)

3. **Weak predictors for Depressants**:
   - All correlations < 0.05
   - Depressants may be harder to predict

**Visualizations:**
- Heatmap showing correlation strength with target
- See: `eda_target_correlation.png`

### 3.3 Pair Plot Analysis

**Key Features Examined**: Oscore, Cscore, Nscore, Impulsive, SS

**Observations:**

1. **Clear class separation** visible in:
   - Oscore vs SS scatter plot
   - Cscore vs Impulsive scatter plot

2. **Overlapping distributions** for:
   - Nscore-based combinations
   - Escore-based combinations

3. **Non-linear relationships** detected between:
   - Oscore and SS for different drug categories
   - Cscore and Impulsive

**Implications**:
- Tree-based models may capture non-linear patterns better than linear models
- Feature interactions (Oscore × SS, Cscore × Impulsive) are promising

**Visualizations:**
- Pair plot with color-coded drug categories
- KDE plots on diagonal
- See: `eda_pairplot.png`

---

## 4. Statistical Significance Tests

### 4.1 ANOVA / Kruskal-Wallis Tests

**Testing whether feature distributions differ significantly across drug categories:**

| Feature | Test Used | Statistic | P-Value | Significance | Effect Size (η²) | Interpretation |
|---------|-----------|-----------|---------|--------------|------------------|----------------|
| **SS** | Kruskal-Wallis | 355.27 | 7.15e-78 | *** | **0.228** | **Large effect** |
| **Oscore** | ANOVA | 159.83 | 1.23e-63 | *** | **0.176** | **Large effect** |
| **Impulsive** | Kruskal-Wallis | 172.25 | 3.94e-38 | *** | **0.110** | Medium effect |
| **Cscore** | ANOVA | 56.61 | 1.99e-24 | *** | **0.070** | Medium effect |
| **Ascore** | ANOVA | 27.70 | 1.53e-12 | *** | 0.036 | Small effect |
| **Nscore** | ANOVA | 11.32 | 1.32e-05 | *** | 0.015 | Small effect |
| **Escore** | ANOVA | 0.56 | 5.73e-01 | No | 0.001 | Negligible |

**Key Findings:**

1. **Highly Discriminative Features** (Large effect size):
   - **Sensation Seeking (SS)**: Strongest discriminator (η² = 0.228)
   - **Openness (Oscore)**: Second strongest (η² = 0.176)
   - Both have p < 10⁻⁶⁰ (extremely significant)

2. **Moderately Discriminative Features** (Medium effect size):
   - **Impulsiveness**: η² = 0.110
   - **Conscientiousness (Cscore)**: η² = 0.070

3. **Weakly Discriminative Features** (Small effect size):
   - Agreeableness (Ascore): η² = 0.036
   - Neuroticism (Nscore): η² = 0.015

4. **Non-Discriminative Features**:
   - **Extraversion (Escore)**: p = 0.57 (not significant)
   - May be candidate for removal

**Effect Size Interpretation**:
- Small: η² ≈ 0.01
- Medium: η² ≈ 0.06
- Large: η² ≥ 0.14

### 4.2 Chi-Square Tests (Categorical Features)

**Testing association between demographic features and drug category:**

| Feature | Chi-Square | P-Value | DOF | Significance | Interpretation |
|---------|-----------|---------|-----|--------------|----------------|
| **Country** | 430.35 | 1.40e-84 | 12 | *** | Very strong association |
| **Age** | 285.59 | 1.72e-55 | 10 | *** | Very strong association |
| **Education** | 231.88 | 2.65e-40 | 16 | *** | Very strong association |
| **Gender** | 173.69 | 1.92e-38 | 2 | *** | Strong association |
| **Ethnicity** | 43.23 | 2.06e-05 | 12 | *** | Moderate association |

**Key Findings:**

1. **All demographic features are statistically significant** predictors (p < 0.001)

2. **Strongest associations**:
   - Country: χ² = 430.35 (but biased by UK/USA dominance)
   - Age: χ² = 285.59 (confirms bivariate findings)
   - Education: χ² = 231.88

3. **Gender shows strong association** despite balanced distribution
   - May indicate gender-specific drug usage patterns

4. **Ethnicity shows weakest (but still significant) association**
   - Limited by severe ethnic imbalance (91% White)

### 4.3 Post-Hoc Pairwise Comparisons

**Top 3 features with largest effect sizes:**

**Sensation Seeking (SS):**
- Depressants vs Hallucinogens: -0.528
- Depressants vs Stimulants: +0.483
- **Hallucinogens vs Stimulants: +1.011** (largest difference)

**Openness (Oscore):**
- Depressants vs Hallucinogens: -0.446
- Depressants vs Stimulants: +0.464
- **Hallucinogens vs Stimulants: +0.910** (second largest)

**Conscientiousness (Cscore):**
- Depressants vs Hallucinogens: +0.213
- Depressants vs Stimulants: -0.386
- **Hallucinogens vs Stimulants: -0.599**

**Interpretation**:
- **Hallucinogens vs Stimulants** show the largest separation across all features
- **Depressants are hardest to distinguish** (smallest mean differences)
- SS and Oscore provide complementary discriminative power

---

## 5. Data Quality Assessment

### 5.1 Missing Values Analysis

| Feature | Missing Count | Missing % | Recommended Action |
|---------|--------------|-----------|-------------------|
| **Education** | 20 | 1.33% | Mode imputation or predictive imputation |
| **Age** | 2 | 0.13% | Mode imputation |
| **Country** | 2 | 0.13% | Mode imputation |
| **Gender** | 1 | 0.07% | Mode imputation |
| **Nscore** | 1 | 0.07% | KNN imputation (k=5) |
| **Escore** | 1 | 0.07% | KNN imputation (k=5) |
| **Oscore** | 1 | 0.07% | KNN imputation (k=5) |
| **Ascore** | 1 | 0.07% | KNN imputation (k=5) |
| **Cscore** | 1 | 0.07% | KNN imputation (k=5) |
| **Impulsive** | 1 | 0.07% | KNN imputation (k=5) |

**Key Findings:**

1. **Minimal missing data** (all features < 2%)
2. **Education has highest missingness** but still manageable at 1.33%
3. **Likely Missing Completely at Random (MCAR)** based on low percentages
4. **No systematic missingness patterns** detected

**Recommended Strategy:**
- **Simple imputation acceptable** due to low missing rates
- Mode imputation for categorical features
- KNN imputation for personality traits to preserve inter-feature relationships
- Consider creating "missing" indicator features if patterns emerge

**Visualizations:**
- Horizontal bar chart of missing value counts
- See: `eda_missing_values_detailed.png`

### 5.2 Outlier Detection

**IQR Method (1.5 × IQR rule):**

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % | Min Value | Max Value |
|---------|-------------|-------------|---------------|-----------|-----------|-----------|
| **Ascore** | -2.40 | 2.39 | 24 | 1.60% | -3.46 | 3.46 |
| **Nscore** | -2.88 | 2.88 | 6 | 0.40% | -3.46 | 3.27 |
| **Impulsive** | -2.57 | 2.39 | 7 | 0.47% | -2.56 | **10.00** |
| **Cscore** | -2.77 | 2.87 | 5 | 0.33% | **-10.00** | 3.46 |
| **Oscore** | -2.88 | 2.88 | 6 | 0.40% | -2.86 | 2.90 |
| **SS** | -2.46 | 2.70 | 0 | 0.00% | -2.08 | 1.92 |

**Z-Score Method (|Z| > 3):**

| Feature | Z-Outlier Count | Z-Outlier % |
|---------|----------------|-------------|
| **Nscore** | 4 | 0.27% |
| **Ascore** | 4 | 0.27% |
| **Cscore** | 3 | 0.20% |
| **Escore** | 1 | 0.07% |
| **Impulsive** | 1 | 0.07% |
| **Oscore** | 0 | 0.00% |
| **SS** | 0 | 0.00% |

**Key Findings:**

1. **Low outlier prevalence** (< 2% for all features)
2. **Ascore has most IQR outliers** (1.6%)
3. **SS has no outliers** (well-behaved distribution)

**Recommended Strategy**:
- **Do not remove outliers** (< 2% threshold)
- Use **RobustScaler** instead of StandardScaler if needed
- **Tree-based models handle outliers naturally**
- Cap extreme values for linear models

### 5.3 Anomaly Detection

**Extreme Values Flagged:**

| Feature | Anomaly Type | Count | Value | Action |
|---------|-------------|-------|-------|--------|
| **Impulsive** | Extreme Value | 1 | 10.00 | Cap at 99th percentile or treat as missing |
| **Escore** | Extreme High | 1 | 50.00 | Investigate validity, likely data entry error |
| **Cscore** | Extreme Low | 1 | -10.00 | Investigate validity, likely data entry error |

**Key Findings:**

1. **Three extreme anomalies detected**:
   - Escore = 50.00 (30+ SD above mean)
   - Cscore = -10.00 (10 SD below mean)
   - Impulsive = 10.00 (10 SD above mean)

2. **Likely data entry errors** or measurement issues

**Recommended Actions**:
- **Escore = 50**: Replace with missing value or cap at 3 SD
- **Cscore = -10**: Replace with missing value or cap at -3 SD
- **Impulsive = 10**: Replace with missing value or cap at 3 SD
- **Document all capping/replacements** for transparency

### 5.4 Duplicate Records

**Findings:**
- **Exact duplicate rows**: 0
- **Duplicate rows (excluding ID)**: 0
- **No duplicate issues detected**

### 5.5 Data Consistency Checks

**Categorical Encoding Consistency:**
- Age: 6 unique values ✓
- Gender: 2 unique values ✓
- Education: 9 unique values ✓
- Country: 7 unique values ✓
- Ethnicity: 7 unique values ✓

**Standardization Range Violations:**

| Feature | Expected Range | Actual Range | Issue |
|---------|----------------|--------------|-------|
| **Escore** | [-3, 3] | [-3.27, **50.00**] | Extreme high outlier |
| **Cscore** | [-3, 3] | [**-10.00**, 3.46] | Extreme low outlier |
| **Impulsive** | [-3, 3] | [-2.56, **10.00**] | Extreme high outlier |

**Other Features**: Within expected range (±3 SD)

**Recommendation**: Address outliers during data cleaning phase

---

## 6. Domain Research Insights

### 6.1 Psychology Literature on Personality and Substance Use

**Big Five Personality Traits and Drug Usage:**

**Openness to Experience (Oscore):**
- **Literature Finding**: High openness associated with experimentation and hallucinogen use
- **Our Data**: Hallucinogen users have significantly higher Oscore (0.419 vs -0.491 for stimulants)
- **Interpretation**: **Consistent with literature** - open individuals seek novel, consciousness-altering experiences

**Conscientiousness (Cscore):**
- **Literature Finding**: Low conscientiousness associated with substance abuse and impulsivity
- **Our Data**: Hallucinogen users have low Cscore (-0.264), stimulant users have high Cscore (0.335)
- **Interpretation**: **Partially consistent** - Hallucinogen use aligns with low conscientiousness, but stimulant users show HIGH conscientiousness, possibly due to functional use (e.g., ADHD medication, productivity enhancement)

**Neuroticism (Nscore):**
- **Literature Finding**: High neuroticism associated with substance use for emotional regulation
- **Our Data**: Depressant and hallucinogen users show slightly higher Nscore (0.08-0.11), stimulant users lower (-0.15)
- **Interpretation**: **Weak alignment** - Effect size is small, suggesting neuroticism is a minor predictor

**Extraversion (Escore):**
- **Literature Finding**: Mixed evidence; some studies link extraversion to social drug use
- **Our Data**: No significant differences across categories (p = 0.57)
- **Interpretation**: **Not a discriminator in this dataset**

**Agreeableness (Ascore):**
- **Literature Finding**: Low agreeableness may correlate with rule-breaking behavior
- **Our Data**: Hallucinogen users less agreeable (-0.182), stimulant users more agreeable (0.219)
- **Interpretation**: **Moderate alignment** - Legal status of hallucinogens may explain lower agreeableness

### 6.2 Sensation Seeking and Impulsivity

**Sensation Seeking (SS):**
- **Literature Finding**: Strong predictor of substance use, especially hallucinogens and stimulants
- **Our Data**: **Strongest discriminator** (η² = 0.228) - Hallucinogen users very high (0.467), stimulant users very low (-0.543)
- **Interpretation**: **Strongly consistent with literature** - Hallucinogen users seek intense, novel experiences

**Impulsiveness:**
- **Literature Finding**: Impulsivity predicts substance use initiation and abuse
- **Our Data**: Hallucinogen users more impulsive (0.326), stimulant users less impulsive (-0.393)
- **Interpretation**: **Consistent** - Impulsive individuals drawn to immediate, intense experiences (hallucinogens)

### 6.3 Age and Drug Usage Patterns

**Literature Finding**: Drug preferences change with age - younger users prefer hallucinogens/party drugs, older users prefer functional stimulants

**Our Data**: 
- 18-24 age group: 72% hallucinogens
- 45+ age groups: 60-80% stimulants

**Interpretation**: **Strongly consistent** - Likely reflects:
- Recreational vs functional use patterns
- Social context (younger adults in party settings)
- Prescription stimulant use increases with age (ADHD, cognitive enhancement)

### 6.4 Model Implications

**Expected Model Behavior Based on Domain Knowledge:**

1. **Age will be top predictor**: Clear age-based usage patterns
2. **Openness + SS will strongly predict hallucinogens**: Both large effect sizes, consistent with literature
3. **Conscientiousness + low Impulsiveness will predict stimulants**: Functional, organized use
4. **Depressants will be hardest to predict**: Smallest mean differences, weak correlations
5. **Extraversion can be dropped**: No discriminative power, consistent with mixed literature

**Feature Interactions to Explore:**
- **Oscore × SS**: Exploratory behavior profile
- **Cscore × Impulsive**: Self-control index
- **Nscore × Age**: Emotional regulation patterns across lifespan

---

## 7. Feature Engineering Recommendations

### 7.1 Interaction Features

**High-Priority Interactions** (based on domain knowledge and EDA):

1. **Risk-Taking Behavior**: `Impulsive × SS`
   - Rationale: Combined measure of impulsivity and novelty-seeking
   - Expected to predict hallucinogen use

2. **Anxious Impulsivity**: `Nscore × Impulsive`
   - Rationale: Emotional instability combined with poor impulse control
   - May predict substance use for emotional regulation

3. **Exploratory Tendency**: `Oscore × SS`
   - Rationale: Openness to new experiences combined with sensation seeking
   - Strong predictor for hallucinogen use

4. **Self-Control Index**: `Cscore × (1 / (Impulsive + 1))`
   - Rationale: Conscientiousness moderated by impulsivity
   - Expected to predict stimulant use (high self-control)

5. **Age × Personality Interactions**:
   - `Age × Oscore`: Openness may matter more for younger users
   - `Age × Nscore`: Emotional instability effects across lifespan

### 7.2 Polynomial Features

**Recommended Non-Linear Transformations**:

1. **Oscore²**: Capture non-linear effect of extreme openness
2. **Cscore²**: Capture non-linear effect of extreme conscientiousness
3. **SS²**: Capture threshold effects in sensation seeking

**Rationale**: Pair plot analysis showed non-linear class separations

### 7.3 Aggregation Features

**Big Five Composite Scores**:

1. **Emotional Instability**: `Nscore - Cscore`
   - High neuroticism + low conscientiousness = emotional dysregulation

2. **Social Behavior**: `Escore + Ascore`
   - Combined measure of social engagement

3. **Personality Variance**: `std(Nscore, Escore, Oscore, Ascore, Cscore)`
   - Overall personality profile consistency

4. **Exploratory Profile**: `Oscore + SS - Cscore`
   - Combined novelty-seeking minus constraint

**Risk Profile Score**:

5. **Substance Use Risk**: `0.3×Nscore + 0.4×Impulsive + 0.3×SS`
   - Weighted combination of risk factors (weights based on effect sizes)

### 7.4 Binning and Discretization

**Age Groups** (already categorical): Maintain 6 groups

**Education Levels**: 
- Collapse 9 levels into 3: Low (1-3), Medium (4-6), High (7-9)
- Reduces dimensionality while preserving order

**Personality Quintiles**:
- Create quintile bins for Oscore, Cscore, SS
- Allows tree models to find threshold effects more easily

### 7.5 Encoding Strategies

**Nominal Features** (no inherent order):
- **Country**: One-hot encoding (7 categories) or target encoding
- **Ethnicity**: One-hot encoding (7 categories) or target encoding

**Ordinal Features** (meaningful order):
- **Age**: Ordinal encoding (1-6)
- **Education**: Ordinal encoding (1-9 or collapsed 1-3)

**Binary Features**:
- **Gender**: Binary encoding (0/1) - already suitable

**Target Encoding** (for high-cardinality):
- Use for Country/Ethnicity if one-hot encoding creates too many features
- Apply with cross-validation to avoid overfitting

### 7.6 Transformations

**Skewed Features** (if needed for linear models):
- **Impulsive**: Log transformation or Box-Cox (positive skew)
- **Escore**: Cap extreme outliers, then standardize

**Standardization/Scaling**:
- **RobustScaler**: Use for features with outliers (Escore, Cscore, Impulsive)
- **StandardScaler**: Use for well-behaved features (Nscore, Oscore, Ascore, SS)
- **No scaling needed**: For tree-based models (Random Forest, XGBoost)

---

## 8. Next Steps

### 8.1 Immediate Actions (Data Cleaning)

**Priority 1: Handle Extreme Anomalies**
- [ ] Cap Escore at 99th percentile or replace Escore=50 with missing
- [ ] Cap Cscore at 1st percentile or replace Cscore=-10 with missing
- [ ] Cap Impulsive at 99th percentile or replace Impulsive=10 with missing
- [ ] Document all capping decisions

**Priority 2: Missing Value Imputation**
- [ ] Apply mode imputation for categorical features (Age, Gender, Education, Country, Ethnicity)
- [ ] Apply KNN imputation (k=5) for personality traits
- [ ] Validate imputation quality (compare distributions before/after)

**Priority 3: Outlier Treatment**
- [ ] Apply RobustScaler for features with remaining outliers
- [ ] Consider Winsorization (capping at 1st/99th percentile)
- [ ] Keep outliers for tree-based models (they handle them naturally)

**Priority 4: Data Consistency**
- [ ] Validate all categorical encodings are consistent
- [ ] Ensure no data leakage (no future information in features)
- [ ] Create train/validation splits with stratification

### 8.2 Feature Engineering Pipeline

**Phase 1: Create Interaction Features**
- [ ] Implement interaction terms (Impulsive×SS, Oscore×SS, Cscore×Impulsive, Nscore×Impulsive)
- [ ] Create age-personality interactions (Age×Oscore, Age×Nscore)

**Phase 2: Polynomial and Aggregation Features**
- [ ] Create polynomial features (Oscore², Cscore², SS²)
- [ ] Create aggregation features (Nscore-Cscore, Oscore+SS-Cscore)
- [ ] Calculate personality variance and risk profile scores

**Phase 3: Encoding**
- [ ] One-hot encode Country and Ethnicity (or use target encoding)
- [ ] Ordinal encode Age and Education
- [ ] Binary encode Gender (if not already)

**Phase 4: Feature Selection**
- [ ] Calculate feature importance using Random Forest
- [ ] Apply mutual information scores
- [ ] Remove low-variance features if any
- [ ] Consider dropping Escore (no discriminative power)

### 8.3 Model Development Strategy

**Baseline Models**:
1. **Dummy Classifier** (stratified): Establish baseline accuracy (~46% - majority class)
2. **Logistic Regression** (with one-hot encoding): Linear baseline
3. **Decision Tree** (depth=5): Simple tree baseline

**Advanced Models** (Expected Best Performance):
1. **Random Forest**: Handle non-linearity, feature interactions, outliers
2. **XGBoost**: Superior gradient boosting, handle missing values
3. **LightGBM**: Fast training, categorical feature support
4. **CatBoost**: Native categorical handling, robust to outliers

**Ensemble Approach**:
- Stacking: Combine RF, XGBoost, LightGBM predictions
- Voting Classifier: Majority voting across top models

**Hyperparameter Tuning**:
- Use GridSearchCV or RandomizedSearchCV
- Focus on class_weight, max_depth, n_estimators, learning_rate
- Apply stratified cross-validation (k=5 or k=10)

### 8.4 Evaluation Strategy

**Metrics** (due to class imbalance):
- **Primary**: Macro F1-score (equal weight to all classes)
- **Secondary**: Weighted F1-score, per-class precision/recall
- **Monitor**: Confusion matrix, ROC-AUC (one-vs-rest)

**Cross-Validation**:
- Stratified K-Fold (k=5 or k=10)
- Ensures balanced class representation
- Report mean ± std across folds

**Class Imbalance Handling**:
- Apply class weights (inversely proportional to class frequency)
- Consider SMOTE (Synthetic Minority Over-sampling) for minority class
- Evaluate impact on per-class performance

**Model Interpretation**:
- SHAP values for feature importance
- Partial dependence plots for key features
- Confusion matrix analysis to identify error patterns

### 8.5 Expected Performance

**Baseline Accuracy**: 46.1% (majority class - Hallucinogens)

**Realistic Target**:
- **Accuracy**: 70-80%
- **Macro F1-Score**: 0.65-0.75
- **Per-Class F1-Score**:
  - Hallucinogens: 0.75-0.85 (majority class, high SS/Oscore)
  - Stimulants: 0.70-0.80 (clear patterns, high Cscore)
  - Depressants: 0.50-0.65 (minority class, weak patterns)

**Optimistic Target** (with extensive tuning):
- **Accuracy**: 80-85%
- **Macro F1-Score**: 0.75-0.80

### 8.6 Documentation and Reporting

**Create Documentation**:
- [ ] Data cleaning log (all transformations and rationale)
- [ ] Feature engineering documentation
- [ ] Model performance comparison table
- [ ] Final model selection justification

**Deliverables**:
- [ ] Cleaned dataset (CSV)
- [ ] Feature-engineered dataset (CSV)
- [ ] Model training script (Python)
- [ ] Prediction script for test set (Python)
- [ ] Final report with visualizations (Markdown)

---

## Summary and Key Takeaways

### Critical Findings

**Strongest Predictors**:
1. **Sensation Seeking (SS)**: η² = 0.228 (large effect)
2. **Openness (Oscore)**: η² = 0.176 (large effect)
3. **Impulsiveness**: η² = 0.110 (medium effect)
4. **Conscientiousness (Cscore)**: η² = 0.070 (medium effect)
5. **Age** (categorical): χ² = 285.59 (very strong association)

**Weakest Predictors**:
1. **Extraversion (Escore)**: p = 0.57 (not significant) - **candidate for removal**
2. **Neuroticism (Nscore)**: η² = 0.015 (small effect)

**Data Quality**:
- Minimal missing values (< 2%)
- Low outlier prevalence (< 2% per feature)
- **Three extreme anomalies requiring treatment** (Escore=50, Cscore=-10, Impulsive=10)
- No duplicate records

**Class Imbalance**:
- Hallucinogens: 46.1% (majority)
- Stimulants: 37.8%
- Depressants: 16.1% (minority)
- **Requires balanced evaluation metrics and class weighting**

**Domain Alignment**:
- **High openness → Hallucinogens**: Consistent with literature
- **High sensation seeking → Hallucinogens**: Strongly consistent
- **High conscientiousness → Stimulants**: Partially consistent (functional use)
- **Age patterns**: Strongly consistent (young → hallucinogens, old → stimulants)

### Recommendations Summary

1. **Clean extreme anomalies** (cap or treat as missing)
2. **Use KNN imputation** for personality traits
3. **Engineer interaction features** (Oscore×SS, Cscore×Impulsive)
4. **Apply one-hot encoding** for Country/Ethnicity
5. **Use tree-based models** (Random Forest, XGBoost, LightGBM)
6. **Apply class weights** or SMOTE for imbalance
7. **Evaluate with macro F1-score** (not just accuracy)
8. **Focus on Depressants performance** (minority class)

### Confidence Level

Based on:
- Large effect sizes for top features (SS, Oscore)
- Clear class separation in pair plots
- Statistical significance of all predictors (except Escore)
- Sufficient sample size (1,500 training records)
- Domain knowledge alignment

**Expected Model Performance**: **70-85% accuracy**, **0.65-0.75 macro F1-score**

---

**Report Generated**: 2025-11-16  
**Analysis Script**: `scripts/eda_analysis.py`  
**Visualizations**: 9 PNG files in `visualizations/`  
**Data Reports**: 7 CSV files in `reports/`  

**Next Phase**: Data Cleaning and Feature Engineering

---
