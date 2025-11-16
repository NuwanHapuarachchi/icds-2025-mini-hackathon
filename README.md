# ICDS 2025 Mini-Hackathon
## Drug Category Prediction - Preliminary Round

---

## Project Overview

This project aims to predict drug categories (Depressants, Hallucinogens, Stimulants) based on mental health conditions and demographic characteristics of patients. The competition is part of the ICDS2025 Mini-Hackathon preliminary screening round.

---

## Directory Structure

```
icds-2025-mini-hackathon/
│
├── data/
│   ├── data_minihackathon_train.csv    # Training dataset (1,500 records)
│   ├── data_minihackathon_test.csv     # Test dataset (377 records)
│   └── sampleSubmission.csv            # Sample submission format
│
├── documentation/
│   ├── attributes.md                    # Detailed attribute descriptions
│   ├── data_overview.md                 # Dataset field descriptions
│   ├── overview.md                      # Competition overview
│   └── Attribute_Information.pdf        # Official attribute documentation
│
├── visualizations/
│   ├── target_distribution.png          # Drug category distribution
│   ├── missing_values.png               # Missing values analysis
│   ├── feature_distributions.png        # All feature distributions
│   ├── correlation_matrix.png           # Feature correlation heatmap
│   ├── personality_by_category.png      # Personality traits by drug category
│   └── demographics_analysis.png        # Demographic patterns
│
├── reports/
│   └── DATA_EXPLORATION_REPORT.md       # Comprehensive data exploration report
│
└── README.md                             # This file
```

---

## Dataset Information

### Training Data
- **Records:** 1,500
- **Features:** 13 predictor variables + 1 target variable
- **Target:** drug_category (3 classes)

### Test Data
- **Records:** 377
- **Features:** 13 predictor variables (no target)

### Feature Categories

**Demographics (6 features):**
- ID, Age, Gender, Education, Country, Ethnicity

**Personality Traits - Big Five Model (5 features):**
- Nscore (Neuroticism)
- Escore (Extraversion)
- Oscore (Openness)
- Ascore (Agreeableness)
- Cscore (Conscientiousness)

**Behavioral Measures (2 features):**
- Impulsive (BIS-11 scale)
- SS (Sensation Seeking - ImpSS scale)

---

## Target Variable

| Drug Category | Count | Percentage |
|---------------|-------|------------|
| Depressants   | 242   | 16.1%      |
| Hallucinogens | 691   | 46.1%      |
| Stimulants    | 567   | 37.8%      |

**Note:** The dataset is imbalanced with Hallucinogens as the majority class.

---

## Key Findings

### Strong Predictors Identified
1. **Age** - Clear usage patterns across age groups
2. **Openness (Oscore)** - Strongest personality discriminator
3. **Conscientiousness (Cscore)** - Strong differentiator between categories
4. **Education** - Shows variation across drug categories

### Data Quality
- Minimal missing values (< 2% for all features)
- Low outlier presence (< 2% per feature)
- High-quality standardized measurements

### Notable Patterns
- Younger users (18-24): 72% use Hallucinogens
- Older users (45+): 60-80% use Stimulants
- Hallucinogen users: High Openness (0.42), Low Conscientiousness (-0.26)
- Stimulant users: Low Openness (-0.49), High Conscientiousness (0.34)

---

## Recommended Approach

### Data Preprocessing
1. Handle missing values (KNN imputation for personality traits)
2. Feature engineering (interaction terms, polynomial features)
3. Address class imbalance (SMOTE, class weights)
4. Stratified train/validation split

### Model Selection
**Preferred Models:**
- XGBoost
- LightGBM
- CatBoost
- Random Forest

**Evaluation Metrics:**
- Primary: Macro F1-score
- Secondary: Per-class precision/recall
- Monitor: Confusion matrix

### Expected Performance
- Baseline accuracy: 60-65%
- Optimized accuracy: 75-85%

---

## Documentation

For detailed analysis and insights, refer to:
- **Comprehensive Report:** `reports/DATA_EXPLORATION_REPORT.md`
- **Attribute Details:** `documentation/attributes.md`
- **Dataset Overview:** `documentation/data_overview.md`

---

## Visualizations

All generated visualizations are available in the `visualizations/` directory:
- Target distribution charts
- Feature distributions
- Correlation analysis
- Personality trait comparisons
- Demographic patterns

---

## Submission Format

Predictions should be submitted as a CSV file with the following format:

```csv
ID,drug_category
1,Hallucinogens
2,Stimulants
3,Depressants
...
```

Reference: `data/sampleSubmission.csv`

---

## Competition Timeline

- **Start:** 18 hours ago
- **Close:** 1 day remaining

---

## Notes

- All features are standardized using validated psychological assessment tools
- Data is anonymized (ID cannot trace to participants)
- Geographic bias present (85% UK/USA)
- Ethnic bias present (91% White participants)
- Consider fairness metrics in model evaluation

---

**Last Updated:** 2025-11-16  
**Competition:** ICDS2025 Mini-Hackathon - Preliminary Round
