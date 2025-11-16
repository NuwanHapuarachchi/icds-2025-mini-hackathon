# Proper Machine Learning Model Training Procedure

1. **Problem Understanding & Data Exploration**
    - Define the objective: Clearly understand what you're predicting (drug_category: 3 classes)
    - Understand the business context: Mental health + drug usage patterns
    - Initial data inspection: Load data, check dimensions, view first/last rows
    - Read documentation: Study variable meanings, data collection methods
    - Identify data types: Categorical vs continuous features
    - Check target distribution: Understand class balance issues

2. **Exploratory Data Analysis (EDA)**
    - Univariate analysis: Distribution of each feature (histograms, box plots)
    - Bivariate analysis: Relationships between features and target (violin plots, grouped statistics)
    - Multivariate analysis: Correlation matrices, pair plots for key features
    - Statistical tests: Check for significant differences across classes
    - Visualization: Create meaningful plots to understand patterns
    - Domain research: Study psychology literature on personality traits and substance use

3. **Data Quality Assessment**
    - Missing values: Count and visualize missing data patterns
    - Outliers: Identify using IQR, Z-scores, or domain knowledge
    - Anomalies: Flag impossible/suspicious values (like Impulsive=10)
    - Data consistency: Check for logical inconsistencies
    - Duplicates: Identify and handle duplicate records
    - Data leakage: Ensure no future information in features

4. **Data Cleaning** ✅ COMPLETED
    - ✅ Handle anomalies: 3 extreme values fixed (Escore, Cscore, Impulsive)
    - ✅ Missing value treatment:
      - ✅ Analyzed missingness pattern (MCAR confirmed)
      - ✅ Categorical: Mode imputation (25 values)
      - ✅ Continuous: KNN imputation k=5 (9 values)
      - ✅ Documented all decisions and rationale
    - ✅ Outlier treatment: Kept all outliers (valid extreme traits)
    - ✅ Data type corrections: All formats validated
    - **Quality Score: 98.8/100 (Excellent)**

5. **Feature Engineering** ✅ COMPLETED
    - ✅ Domain-specific features: 10 psychological risk indicators created
    - ✅ Interaction terms: 11 features (Impulsive × SS, Nscore × Cscore, etc.)
    - ✅ Polynomial features: 10 features (7 squared + 3 cubed)
    - ✅ Aggregations: 6 ratio features (risk/protective ratios)
    - ✅ Binning/discretization: 4 features binned into 3 levels
    - ✅ Encoding categorical variables: One-hot (13) + Ordinal (2)
    - ✅ Feature transformations: 9 features (7 log + 2 sqrt)
    - **Total Features: 72 (454% expansion from 13 originals)**

6. **Feature Selection**
    - Remove low-variance features: Eliminate constants or near-constants
    - Correlation analysis: Remove highly correlated redundant features
    - Statistical tests: Chi-square, ANOVA, mutual information
    - Model-based selection: Feature importance from tree models, L1 regularization
    - Recursive elimination: Iteratively remove least important features
    - Domain expertise: Keep psychologically meaningful features

7. **Data Splitting Strategy**
    - Train-validation-test split: Typically 70-15-15 or 80-20 (then CV on train)
    - Stratified splitting: Preserve target class distribution in each split
    - Time-based splits: If temporal ordering matters (not applicable here)
    - Cross-validation setup: Choose K-fold (5 or 10) with stratification
    - Holdout set: Keep completely untouched for final evaluation

8. **Handle Class Imbalance**
    - Understand impact: How imbalance affects your problem
    - Resampling techniques:
      - Oversampling minority: SMOTE, ADASYN, BorderlineSMOTE
      - Undersampling majority: Random, Tomek links, NearMiss
      - Combination: SMOTE-ENN, SMOTE-Tomek
    - Algorithmic approaches: Class weights, cost-sensitive learning
    - Evaluation metrics: Choose appropriate metrics (F1, precision-recall)
    - Apply only to training: Never resample validation/test sets

9. **Feature Scaling/Normalization**
    - Understand when needed: Distance-based models, neural networks, regularization
    - Choose scaling method:
      - StandardScaler: Mean=0, SD=1 (assumes normal distribution)
      - MinMaxScaler: Scale to [0,1] range
      - RobustScaler: Uses median/IQR (resistant to outliers)
    - Fit on training only: Transform validation/test using training parameters
    - Handle categorical encodings: Decide whether to scale encoded categories

10. **Baseline Model Creation**
     - Simple models first: Logistic regression, decision tree, naive Bayes
     - Establish benchmark: Performance floor to beat
     - Majority class baseline: Predict most frequent class
     - Random baseline: Random predictions within class distribution
     - Document baseline metrics: Accuracy, F1, precision, recall per class

11. **Model Selection & Training**
     - Choose candidate algorithms:
        - Tree-based: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
        - Linear: Logistic Regression, SVM
        - Neural Networks: MLP, deep learning
        - Instance-based: KNN
     - Start simple, increase complexity: Add models progressively
     - Train multiple model types: Diversity helps ensemble performance
     - Use cross-validation: Get robust performance estimates
     - Track experiments: Log parameters, metrics, models systematically

12. **Hyperparameter Optimization**
     - Understand hyperparameter impact: Learn what each parameter controls
     - Choose optimization strategy:
        - Grid Search: Exhaustive but slow
        - Random Search: Faster, good for exploration
        - Bayesian Optimization: Smart, efficient (Optuna, Hyperopt)
        - Evolutionary algorithms: Genetic algorithms
     - Define search space: Based on experience and documentation
     - Use cross-validation: Optimize on CV score, not single validation
     - Budget management: Balance time vs performance gains
     - Early stopping: Prevent overfitting in iterative models
     - Parallel processing: Utilize multiple cores/GPUs

13. **Model Evaluation**
     - Multiple metrics: Don't rely on single metric
     - Classification: Accuracy, precision, recall, F1 (macro/weighted), ROC-AUC
     - Per-class metrics: Understand performance on each class
     - Confusion matrix: Visualize misclassification patterns
     - Cross-validation scores: Mean and standard deviation across folds
     - Learning curves: Plot train vs validation performance
     - Validation curves: Visualize hyperparameter effects
     - Statistical significance: Compare models with statistical tests

14. **Model Interpretation & Analysis**
     - Feature importance: Which features drive predictions
     - SHAP values: Model-agnostic explanations
     - Partial dependence plots: Feature effect on predictions
     - Individual predictions: Examine correct and incorrect cases
     - Error analysis: Study misclassified examples
     - Bias detection: Check for demographic biases

15. **Ensemble Methods**
     - Understand ensemble types:
        - Bagging: Bootstrap aggregating (Random Forest)
        - Boosting: Sequential learning (XGBoost, AdaBoost)
        - Stacking: Meta-learning from multiple models
        - Blending: Weighted averaging
     - Create diverse base models: Different algorithms, features, parameters
     - Optimize ensemble weights: Based on validation performance
     - Meta-learner selection: Logistic regression, neural network, or another model
     - Out-of-fold predictions: Prevent overfitting in stacking

16. **Model Validation & Diagnostics**
     - Overfitting check: Compare train vs validation performance
     - Underfitting check: Both train and validation poor
     - Bias-variance tradeoff: Understand model complexity
     - Residual analysis: For regression-like interpretations
     - Cross-validation consistency: Check variance across folds
     - Robustness testing: Small data perturbations

17. **Model Optimization & Refinement**
     - Iterative improvement: Based on error analysis findings
     - Feature engineering refinement: Add/remove features based on importance
     - Hyperparameter fine-tuning: Narrow search around best parameters
     - Threshold optimization: Adjust classification thresholds per class
     - Calibration: Ensure predicted probabilities are accurate
     - Regularization tuning: Prevent overfitting

18. **Final Model Selection**
     - Compare all models: Use consistent metrics on validation set
     - Consider tradeoffs: Accuracy vs interpretability vs speed
     - Business requirements: Align with project constraints
     - Stakeholder input: If applicable
     - Ensemble decision: Single model vs ensemble
     - Document rationale: Why this model was chosen

19. **Final Evaluation on Test Set**
     - One-time evaluation: Use held-out test set only once
     - Report all metrics: Comprehensive performance report
     - Confidence intervals: Bootstrap or statistical methods
     - Per-class performance: Detailed breakdown
     - Compare to baseline: Quantify improvement
     - Real-world validation: If possible, validate on new data

20. **Prediction Generation**
     - Load test data: Same preprocessing pipeline
     - Apply transformations: Use fitted scalers/encoders from training
     - Handle missing values: Same imputation strategy
     - Feature engineering: Apply same transformations
     - Generate predictions: Use final model
     - Post-processing: Format according to submission requirements
     - Sanity checks: Distribution, ranges, format compliance

21. **Documentation & Reproducibility**
     - Code documentation: Clear comments and docstrings
     - Methodology document: Explain all decisions
     - Experiment tracking: MLflow, Weights & Biases, or spreadsheets
     - Version control: Git for code and data versioning
     - Environment specification: Requirements.txt, conda environment
     - Random seeds: Set for reproducibility
     - Data provenance: Track data sources and versions

22. **Model Deployment Preparation (if applicable)**
     - Model serialization: Save trained model (pickle, joblib, ONNX)
     - Pipeline creation: End-to-end preprocessing + prediction
     - Performance benchmarking: Inference time, memory usage
     - API creation: REST API for predictions
     - Monitoring setup: Track model performance over time
     - Retraining strategy: When and how to update model

## Key Principles Throughout
- ✅ Stratification: Always maintain class distribution in splits
- ✅ No data leakage: Fit only on training, transform on validation/test
- ✅ Reproducibility: Set random seeds, document everything
- ✅ Iterative process: EDA → Model → Analysis → Refinement → Repeat
- ✅ Validation-driven: All decisions based on validation performance
- ✅ Documentation: Track every experiment, decision, and result
- ✅ Domain knowledge: Incorporate psychological/medical understanding
- ✅ Simplicity first: Start simple, add complexity when justified

This procedure ensures a systematic, scientifically rigorous approach to building high-accuracy predictive models while avoiding common pitfalls like overfitting, data leakage, and inadequate validation.