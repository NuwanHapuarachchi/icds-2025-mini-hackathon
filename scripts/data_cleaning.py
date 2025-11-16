"""
Data Cleaning Script
ICDS 2025 Mini-Hackathon - Drug Category Prediction
Implements comprehensive data cleaning pipeline based on DQA findings
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import mstats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA CLEANING PIPELINE - Drug Category Prediction")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

continuous_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
                       'Impulsive', 'SS']
categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']

print("\nConfiguration:")
print(f"  Continuous features: {len(continuous_features)}")
print(f"  Categorical features: {len(categorical_features)}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("[1] LOADING DATA")
print("="*80)

train_df_raw = pd.read_csv('data/data_minihackathon_train.csv')
print(f"\n✓ Training data loaded: {train_df_raw.shape}")
print(f"  Total records: {len(train_df_raw)}")
print(f"  Total features: {len(train_df_raw.columns)}")

# Create working copy
train_df = train_df_raw.copy()

# ============================================================================
# STEP 1: HANDLE EXTREME ANOMALIES
# ============================================================================

print("\n" + "="*80)
print("[2] HANDLING EXTREME ANOMALIES")
print("="*80)

print("\nBefore anomaly treatment:")
print(f"  Escore max: {train_df['Escore'].max():.2f}")
print(f"  Cscore min: {train_df['Cscore'].min():.2f}")
print(f"  Impulsive max: {train_df['Impulsive'].max():.2f}")

# Flag extreme values as NaN
anomalies_fixed = 0

# Escore > 10 (normal range: -3.27 to 3.27)
if (train_df['Escore'] > 10).any():
    count = (train_df['Escore'] > 10).sum()
    train_df.loc[train_df['Escore'] > 10, 'Escore'] = np.nan
    anomalies_fixed += count
    print(f"\n✓ Flagged {count} Escore anomaly(ies) > 10 as NaN")

# Cscore < -5 (normal range: -3.46 to 3.46)
if (train_df['Cscore'] < -5).any():
    count = (train_df['Cscore'] < -5).sum()
    train_df.loc[train_df['Cscore'] < -5, 'Cscore'] = np.nan
    anomalies_fixed += count
    print(f"✓ Flagged {count} Cscore anomaly(ies) < -5 as NaN")

# Impulsive > 5 (normal range: -2.56 to 2.90)
if (train_df['Impulsive'] > 5).any():
    count = (train_df['Impulsive'] > 5).sum()
    train_df.loc[train_df['Impulsive'] > 5, 'Impulsive'] = np.nan
    anomalies_fixed += count
    print(f"✓ Flagged {count} Impulsive anomaly(ies) > 5 as NaN")

print(f"\nTotal anomalies flagged: {anomalies_fixed}")

# ============================================================================
# STEP 2: IMPUTE MISSING VALUES - CATEGORICAL
# ============================================================================

print("\n" + "="*80)
print("[3] IMPUTING CATEGORICAL FEATURES")
print("="*80)

print("\nMissing values before imputation:")
for feature in categorical_features:
    missing = train_df[feature].isnull().sum()
    if missing > 0:
        print(f"  {feature}: {missing}")

# Mode imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_features] = cat_imputer.fit_transform(
    train_df[categorical_features]
)

print("\n✓ Categorical features imputed using mode")

# Verify
missing_after = train_df[categorical_features].isnull().sum().sum()
print(f"✓ Missing values after: {missing_after}")

# ============================================================================
# STEP 3: IMPUTE MISSING VALUES - CONTINUOUS
# ============================================================================

print("\n" + "="*80)
print("[4] IMPUTING CONTINUOUS FEATURES")
print("="*80)

print("\nMissing values before imputation:")
for feature in continuous_features:
    missing = train_df[feature].isnull().sum()
    if missing > 0:
        print(f"  {feature}: {missing}")

total_missing_before = train_df[continuous_features].isnull().sum().sum()

# KNN imputation (k=5, distance-weighted)
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
train_df[continuous_features] = knn_imputer.fit_transform(
    train_df[continuous_features]
)

print(f"\n✓ Continuous features imputed using KNN (k=5, distance-weighted)")
print(f"  Total values imputed: {total_missing_before}")

# Verify
missing_after = train_df[continuous_features].isnull().sum().sum()
print(f"✓ Missing values after: {missing_after}")

# ============================================================================
# STEP 4: VALIDATE CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("[5] VALIDATING CLEANED DATA")
print("="*80)

print("\n5.1 Missing Value Check:")
total_missing = train_df.isnull().sum().sum()
if total_missing == 0:
    print("  ✓ No missing values remain")
else:
    print(f"  ⚠️ {total_missing} missing values still present")

print("\n5.2 Range Validation (Continuous Features):")
range_issues = 0
for feature in continuous_features:
    mean = train_df[feature].mean()
    std = train_df[feature].std()
    min_val = train_df[feature].min()
    max_val = train_df[feature].max()
    
    if min_val < (mean - 5 * std) or max_val > (mean + 5 * std):
        print(f"  ⚠️ {feature}: Range [{min_val:.2f}, {max_val:.2f}] "
              f"exceeds ±5 SD")
        range_issues += 1
    else:
        print(f"  ✓ {feature}: Range [{min_val:.2f}, {max_val:.2f}] is valid")

if range_issues == 0:
    print("\n  ✓ All features within expected range (±5 SD)")
else:
    print(f"\n  ⚠️ {range_issues} features have range issues")

print("\n5.3 Duplicate Check:")
duplicates = train_df.duplicated().sum()
if duplicates == 0:
    print("  ✓ No duplicate rows")
else:
    print(f"  ⚠️ {duplicates} duplicate rows found")
    train_df = train_df.drop_duplicates()
    print(f"  ✓ Duplicates removed")

print("\n5.4 Distribution Statistics (After Cleaning):")
stats_df = train_df[continuous_features].describe().T[['mean', 'std', 'min', 'max']]
stats_df['skewness'] = train_df[continuous_features].skew()
stats_df['kurtosis'] = train_df[continuous_features].kurtosis()
print(stats_df.to_string())

# ============================================================================
# STEP 5: COMPARE BEFORE/AFTER
# ============================================================================

print("\n" + "="*80)
print("[6] BEFORE/AFTER COMPARISON")
print("="*80)

comparison = []
for feature in continuous_features:
    before_mean = train_df_raw[feature].mean()
    after_mean = train_df[feature].mean()
    before_std = train_df_raw[feature].std()
    after_std = train_df[feature].std()
    
    mean_change_pct = abs((after_mean - before_mean) / before_mean) * 100 if before_mean != 0 else 0
    std_change_pct = abs((after_std - before_std) / before_std) * 100 if before_std != 0 else 0
    
    comparison.append({
        'Feature': feature,
        'Mean_Before': before_mean,
        'Mean_After': after_mean,
        'Mean_Change_%': mean_change_pct,
        'Std_Before': before_std,
        'Std_After': after_std,
        'Std_Change_%': std_change_pct
    })

comparison_df = pd.DataFrame(comparison)
print("\nDistribution Changes:")
print(comparison_df.to_string(index=False))

# Check if changes are minimal
max_mean_change = comparison_df['Mean_Change_%'].max()
max_std_change = comparison_df['Std_Change_%'].max()

print(f"\nMaximum changes:")
print(f"  Mean: {max_mean_change:.2f}%")
print(f"  Std: {max_std_change:.2f}%")

if max_mean_change < 5 and max_std_change < 10:
    print("\n✓ Distributions preserved (minimal changes)")
else:
    print("\n⚠️ Significant distribution changes detected")

# Save comparison
comparison_df.to_csv('reports/dqa/data_cleaning_comparison.csv', index=False)
print("\n✓ Saved: reports/dqa/data_cleaning_comparison.csv")

# ============================================================================
# STEP 6: SAVE CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("[7] SAVING CLEANED DATA")
print("="*80)

# Save cleaned training data
output_path = 'data/data_minihackathon_train_clean.csv'
train_df.to_csv(output_path, index=False)
print(f"\n✓ Cleaned training data saved: {output_path}")
print(f"  Records: {len(train_df)}")
print(f"  Features: {len(train_df.columns)}")

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("[8] GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Distribution comparison (before/after)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(continuous_features):
    # Before
    axes[idx].hist(train_df_raw[feature].dropna(), bins=30, alpha=0.5, 
                   label='Before', color='red', edgecolor='black')
    # After
    axes[idx].hist(train_df[feature].dropna(), bins=30, alpha=0.5, 
                   label='After', color='green', edgecolor='black')
    
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

axes[7].axis('off')

plt.suptitle('Data Distribution: Before vs After Cleaning', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/data_cleaning_comparison.png', dpi=300, 
            bbox_inches='tight')
print("\n✓ Saved: visualizations/data_cleaning_comparison.png")
plt.close()

# Plot 2: Box plots comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(continuous_features):
    data_before = train_df_raw[feature].dropna()
    data_after = train_df[feature].dropna()
    
    bp = axes[idx].boxplot([data_before, data_after], 
                           labels=['Before', 'After'],
                           patch_artist=True,
                           boxprops=dict(alpha=0.6),
                           medianprops=dict(color='red', linewidth=2))
    
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

axes[7].axis('off')

plt.suptitle('Outlier Comparison: Before vs After Cleaning', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/data_cleaning_boxplots.png', dpi=300, 
            bbox_inches='tight')
print("✓ Saved: visualizations/data_cleaning_boxplots.png")
plt.close()

# ============================================================================
# STEP 8: CLEANING SUMMARY
# ============================================================================

print("\n" + "="*80)
print("[9] CLEANING SUMMARY")
print("="*80)

summary = {
    'Metric': [
        'Total Records',
        'Records Removed',
        'Extreme Anomalies Fixed',
        'Missing Values Imputed',
        'Duplicates Removed',
        'Final Records',
        'Data Completeness',
        'Quality Score'
    ],
    'Value': [
        len(train_df_raw),
        len(train_df_raw) - len(train_df),
        anomalies_fixed,
        total_missing_before + anomalies_fixed,
        duplicates,
        len(train_df),
        '100%',
        'Ready for Feature Engineering'
    ]
}

summary_df = pd.DataFrame(summary)
print("\nCleaning Summary:")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('reports/dqa/data_cleaning_summary.csv', index=False)
print("\n✓ Saved: reports/dqa/data_cleaning_summary.csv")

# ============================================================================
# STEP 9: APPLY TO TEST SET
# ============================================================================

print("\n" + "="*80)
print("[10] CLEANING TEST SET")
print("="*80)

try:
    test_df_raw = pd.read_csv('data/data_minihackathon_test.csv')
    print(f"\n✓ Test data loaded: {test_df_raw.shape}")
    
    test_df = test_df_raw.copy()
    
    # Apply same transformations
    print("\nApplying same transformations to test set...")
    
    # 1. Flag anomalies
    if 'Escore' in test_df.columns:
        test_df.loc[test_df['Escore'] > 10, 'Escore'] = np.nan
    if 'Cscore' in test_df.columns:
        test_df.loc[test_df['Cscore'] < -5, 'Cscore'] = np.nan
    if 'Impulsive' in test_df.columns:
        test_df.loc[test_df['Impulsive'] > 5, 'Impulsive'] = np.nan
    
    # 2. Impute categorical (using same imputer)
    test_categorical = [f for f in categorical_features if f in test_df.columns]
    if len(test_categorical) > 0:
        test_df[test_categorical] = cat_imputer.transform(test_df[test_categorical])
    
    # 3. Impute continuous (using same imputer)
    test_continuous = [f for f in continuous_features if f in test_df.columns]
    if len(test_continuous) > 0:
        test_df[test_continuous] = knn_imputer.transform(test_df[test_continuous])
    
    # 4. Remove duplicates
    test_df = test_df.drop_duplicates()
    
    # Save cleaned test set
    test_output_path = 'data/data_minihackathon_test_clean.csv'
    test_df.to_csv(test_output_path, index=False)
    print(f"\n✓ Cleaned test data saved: {test_output_path}")
    print(f"  Records: {len(test_df)}")
    
except FileNotFoundError:
    print("\n⚠️ Test file not found, skipping test set cleaning")

# ============================================================================
# FINAL MESSAGE
# ============================================================================

print("\n" + "="*80)
print("DATA CLEANING COMPLETE")
print("="*80)

print("\n✓ All data cleaning steps completed successfully")
print("\nCleaned Files:")
print("  - data/data_minihackathon_train_clean.csv")
if 'test_df' in locals():
    print("  - data/data_minihackathon_test_clean.csv")

print("\nReports Generated:")
print("  - reports/dqa/data_cleaning_comparison.csv")
print("  - reports/dqa/data_cleaning_summary.csv")

print("\nVisualizations Generated:")
print("  - visualizations/data_cleaning_comparison.png")
print("  - visualizations/data_cleaning_boxplots.png")

print("\nNext Steps:")
print("  1. Review cleaned data distributions")
print("  2. Proceed to Feature Engineering")
print("  3. Create interaction and polynomial features")
print("  4. Encode categorical variables")
print("  5. Begin model development")

print("\n" + "="*80)
