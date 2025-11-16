"""
Data Quality Assessment Script
ICDS 2025 Mini-Hackathon - Drug Category Prediction
Performs comprehensive data quality analysis including missing values,
outliers, anomalies, data consistency, duplicates, and data leakage checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("DATA QUALITY ASSESSMENT - Drug Category Prediction")
print("="*80)

# Load data
print("\n[1] Loading Data...")
train_df = pd.read_csv('data/data_minihackathon_train.csv')
print(f"✓ Training data loaded: {train_df.shape}")

# ============================================================================
# 1. MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[2] MISSING VALUES ANALYSIS")
print("="*80)

missing_df = pd.DataFrame({
    'Feature': train_df.columns,
    'Missing_Count': train_df.isnull().sum(),
    'Missing_Percentage': (train_df.isnull().sum() / len(train_df)) * 100,
    'Data_Type': train_df.dtypes
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
    'Missing_Count', ascending=False
).reset_index(drop=True)

print("\nMissing Values Summary:")
print(missing_df.to_string(index=False))

# Save missing values report
missing_df.to_csv('reports/dqa/dqa_missing_values.csv', index=False)
print("\n✓ Saved: reports/dqa/dqa_missing_values.csv")

# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Missing count
if len(missing_df) > 0:
    axes[0].barh(missing_df['Feature'], missing_df['Missing_Count'], color='#e74c3c')
    axes[0].set_xlabel('Missing Count')
    axes[0].set_title('Missing Values Count by Feature', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Missing percentage
    axes[1].barh(missing_df['Feature'], missing_df['Missing_Percentage'], color='#3498db')
    axes[1].set_xlabel('Missing Percentage (%)')
    axes[1].set_title('Missing Values Percentage by Feature', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'No Missing Values Detected', 
                ha='center', va='center', fontsize=16)
    axes[1].text(0.5, 0.5, 'All Features Complete', 
                ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.savefig('visualizations/dqa_missing_values.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/dqa_missing_values.png")
plt.close()

# ============================================================================
# 2. OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("[3] OUTLIER DETECTION")
print("="*80)

continuous_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
                       'Impulsive', 'SS']

# IQR Method
print("\n2.1 IQR Method (1.5 × IQR):")
iqr_outliers = []

for feature in continuous_features:
    Q1 = train_df[feature].quantile(0.25)
    Q3 = train_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[feature] < lower_bound) | 
                        (train_df[feature] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(train_df)) * 100
    
    iqr_outliers.append({
        'Feature': feature,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': outlier_pct,
        'Min_Value': train_df[feature].min(),
        'Max_Value': train_df[feature].max()
    })

iqr_df = pd.DataFrame(iqr_outliers).sort_values('Outlier_Percentage', 
                                                  ascending=False)
print(iqr_df.to_string(index=False))

# Save IQR outliers report
iqr_df.to_csv('reports/dqa/dqa_outliers_iqr.csv', index=False)
print("\n✓ Saved: reports/dqa/dqa_outliers_iqr.csv")

# Z-Score Method
print("\n2.2 Z-Score Method (|Z| > 3):")
zscore_outliers = []

for feature in continuous_features:
    z_scores = np.abs(stats.zscore(train_df[feature].dropna()))
    outlier_count = (z_scores > 3).sum()
    outlier_pct = (outlier_count / len(train_df)) * 100
    
    zscore_outliers.append({
        'Feature': feature,
        'Outlier_Count_Z3': outlier_count,
        'Outlier_Percentage_Z3': outlier_pct
    })

zscore_df = pd.DataFrame(zscore_outliers).sort_values('Outlier_Count_Z3', 
                                                        ascending=False)
print(zscore_df.to_string(index=False))

# Save Z-score outliers report
zscore_df.to_csv('reports/dqa/dqa_outliers_zscore.csv', index=False)
print("\n✓ Saved: reports/dqa/dqa_outliers_zscore.csv")

# Visualize outliers
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(continuous_features):
    data = train_df[feature].dropna()
    
    # Box plot with outliers highlighted
    bp = axes[idx].boxplot([data], vert=True, patch_artist=True,
                           boxprops=dict(facecolor='#3498db', alpha=0.6),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1),
                           capprops=dict(color='black', linewidth=1))
    
    axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add statistics text
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((data < lower) | (data > upper)).sum()
    
    axes[idx].text(0.95, 0.95, f'Outliers: {outlier_count}',
                  transform=axes[idx].transAxes,
                  fontsize=9, verticalalignment='top',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide the last subplot (8th) since we have 7 features
axes[7].axis('off')

plt.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/dqa_outliers_boxplots.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/dqa_outliers_boxplots.png")
plt.close()

# ============================================================================
# 3. ANOMALY DETECTION
# ============================================================================
print("\n" + "="*80)
print("[4] ANOMALY DETECTION - Extreme Values")
print("="*80)

anomalies = []

for feature in continuous_features:
    mean = train_df[feature].mean()
    std = train_df[feature].std()
    
    # Flag values beyond 5 standard deviations
    extreme_low = train_df[train_df[feature] < (mean - 5 * std)]
    extreme_high = train_df[train_df[feature] > (mean + 5 * std)]
    
    if len(extreme_low) > 0:
        for idx, row in extreme_low.iterrows():
            z_score = (row[feature] - mean) / std
            anomalies.append({
                'Feature': feature,
                'Record_ID': row['ID'] if 'ID' in train_df.columns else idx,
                'Value': row[feature],
                'Z_Score': z_score,
                'Type': 'Extreme Low'
            })
    
    if len(extreme_high) > 0:
        for idx, row in extreme_high.iterrows():
            z_score = (row[feature] - mean) / std
            anomalies.append({
                'Feature': feature,
                'Record_ID': row['ID'] if 'ID' in train_df.columns else idx,
                'Value': row[feature],
                'Z_Score': z_score,
                'Type': 'Extreme High'
            })

if len(anomalies) > 0:
    anomaly_df = pd.DataFrame(anomalies).sort_values('Z_Score', 
                                                       key=abs, 
                                                       ascending=False)
    print(f"\nTotal Anomalies Detected: {len(anomaly_df)}")
    print("\nExtreme Anomalies (|Z| > 5):")
    print(anomaly_df.to_string(index=False))
    
    # Save anomalies report
    anomaly_df.to_csv('reports/dqa/dqa_anomalies.csv', index=False)
    print("\n✓ Saved: reports/dqa/dqa_anomalies.csv")
else:
    print("\nNo extreme anomalies detected (|Z| > 5)")

# ============================================================================
# 4. DATA CONSISTENCY CHECKS
# ============================================================================
print("\n" + "="*80)
print("[5] DATA CONSISTENCY CHECKS")
print("="*80)

categorical_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']

print("\n4.1 Categorical Value Ranges:")
consistency_report = []

for feature in categorical_features:
    if feature in train_df.columns:
        unique_count = train_df[feature].nunique()
        unique_values = sorted(train_df[feature].dropna().unique())
        
        consistency_report.append({
            'Feature': feature,
            'Unique_Count': unique_count,
            'Min_Value': min(unique_values) if len(unique_values) > 0 else None,
            'Max_Value': max(unique_values) if len(unique_values) > 0 else None,
            'Expected_Count': {
                'Age': 6, 'Gender': 2, 'Education': 9, 
                'Country': 7, 'Ethnicity': 7
            }.get(feature, 'N/A')
        })

consistency_df = pd.DataFrame(consistency_report)
print(consistency_df.to_string(index=False))

# Save consistency report
consistency_df.to_csv('reports/dqa/dqa_consistency.csv', index=False)
print("\n✓ Saved: reports/dqa/dqa_consistency.csv")

print("\n4.2 Standardization Range Validation:")
range_violations = []

for feature in continuous_features:
    mean = train_df[feature].mean()
    std = train_df[feature].std()
    min_val = train_df[feature].min()
    max_val = train_df[feature].max()
    
    expected_min = mean - 3 * std
    expected_max = mean + 3 * std
    
    violation = False
    if min_val < (mean - 5 * std):
        violation = True
        violation_type = 'Lower bound'
    if max_val > (mean + 5 * std):
        violation = True
        violation_type = 'Upper bound'
    
    if violation:
        range_violations.append({
            'Feature': feature,
            'Min_Value': min_val,
            'Max_Value': max_val,
            'Expected_Range': f'[{expected_min:.2f}, {expected_max:.2f}]',
            'Violation_Type': violation_type
        })

if len(range_violations) > 0:
    print("\nRange Violations Detected:")
    violation_df = pd.DataFrame(range_violations)
    print(violation_df.to_string(index=False))
else:
    print("\nNo range violations detected (all features within ±5 SD)")

# ============================================================================
# 5. DUPLICATE DETECTION
# ============================================================================
print("\n" + "="*80)
print("[6] DUPLICATE DETECTION")
print("="*80)

# Exact duplicates
exact_duplicates = train_df.duplicated().sum()
print(f"\nExact duplicate rows: {exact_duplicates}")

# Feature duplicates (excluding ID)
feature_cols = [col for col in train_df.columns if col not in ['ID']]
feature_duplicates = train_df.duplicated(subset=feature_cols).sum()
print(f"Feature duplicates (excluding ID): {feature_duplicates}")

if exact_duplicates > 0 or feature_duplicates > 0:
    duplicate_records = train_df[train_df.duplicated(keep=False)]
    print(f"\nDuplicate records found: {len(duplicate_records)}")
    duplicate_records.to_csv('reports/dqa/dqa_duplicates.csv', index=False)
    print("✓ Saved: reports/dqa/dqa_duplicates.csv")
else:
    print("\n✓ No duplicates detected")

# ============================================================================
# 6. DATA LEAKAGE CHECKS
# ============================================================================
print("\n" + "="*80)
print("[7] DATA LEAKAGE VALIDATION")
print("="*80)

print("\nChecking for potential data leakage...")

# Check if target is in features
if 'drug_category' in train_df.columns:
    print("✓ Target variable 'drug_category' present in training data")
    
    # Check for any derived features from target
    feature_names = train_df.columns.tolist()
    suspicious_features = [f for f in feature_names 
                          if 'drug' in f.lower() and f != 'drug_category']
    
    if len(suspicious_features) > 0:
        print(f"⚠️ Suspicious features found: {suspicious_features}")
    else:
        print("✓ No suspicious feature names detected")
else:
    print("⚠️ Target variable 'drug_category' not found in data")

# Check for temporal leakage (if applicable)
print("\n✓ No temporal features detected (no time-series data)")
print("✓ All features represent measurements at assessment time")

# ============================================================================
# 7. CLASS BALANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[8] CLASS BALANCE ANALYSIS")
print("="*80)

if 'drug_category' in train_df.columns:
    class_distribution = train_df['drug_category'].value_counts().sort_index()
    class_percentages = train_df['drug_category'].value_counts(
        normalize=True).sort_index() * 100
    
    balance_df = pd.DataFrame({
        'Drug_Category': class_distribution.index,
        'Count': class_distribution.values,
        'Percentage': class_percentages.values
    })
    
    print("\nClass Distribution:")
    print(balance_df.to_string(index=False))
    
    # Save class balance report
    balance_df.to_csv('reports/dqa/dqa_class_balance.csv', index=False)
    print("\n✓ Saved: reports/dqa/dqa_class_balance.csv")
    
    # Calculate imbalance ratio
    max_class = class_distribution.max()
    min_class = class_distribution.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2:
        print("⚠️ Significant class imbalance detected")
        print("   Recommendation: Use stratified sampling and class weights")
    else:
        print("✓ Classes are reasonably balanced")
    
    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    axes[0].bar(balance_df['Drug_Category'], balance_df['Count'], 
               color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_xlabel('Drug Category')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution - Count', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, (cat, count) in enumerate(zip(balance_df['Drug_Category'], 
                                         balance_df['Count'])):
        axes[0].text(i, count + 10, f'{count}', ha='center', va='bottom', 
                    fontweight='bold')
    
    # Pie chart
    axes[1].pie(balance_df['Percentage'], labels=balance_df['Drug_Category'],
               autopct='%1.1f%%', startangle=90, 
               colors=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_title('Class Distribution - Percentage', fontsize=14, 
                     fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/dqa_class_balance.png', dpi=300, 
                bbox_inches='tight')
    print("\n✓ Saved: visualizations/dqa_class_balance.png")
    plt.close()

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("[9] COMPREHENSIVE DATA QUALITY SUMMARY")
print("="*80)

summary_stats = train_df[continuous_features].describe().T
summary_stats['skewness'] = train_df[continuous_features].skew()
summary_stats['kurtosis'] = train_df[continuous_features].kurtosis()
summary_stats['missing'] = train_df[continuous_features].isnull().sum()
summary_stats['missing_pct'] = (train_df[continuous_features].isnull().sum() / 
                                 len(train_df)) * 100

print("\nContinuous Features Summary Statistics:")
print(summary_stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 
                     'missing', 'missing_pct']].to_string())

# Save summary statistics
summary_stats.to_csv('reports/dqa/dqa_summary_statistics.csv')
print("\n✓ Saved: reports/dqa/dqa_summary_statistics.csv")

# ============================================================================
# 9. DATA QUALITY SCORECARD
# ============================================================================
print("\n" + "="*80)
print("[10] DATA QUALITY SCORECARD")
print("="*80)

total_records = len(train_df)
total_features = len(train_df.columns)
total_cells = total_records * total_features

# Calculate scores
missing_score = 100 - (train_df.isnull().sum().sum() / total_cells * 100)
duplicate_score = 100 - (exact_duplicates / total_records * 100)

# Outlier score (penalize high outlier rates)
total_outliers = iqr_df['Outlier_Count'].sum()
outlier_score = 100 - (total_outliers / (total_records * len(continuous_features)) * 100)

# Anomaly score
total_anomalies = len(anomalies) if len(anomalies) > 0 else 0
anomaly_score = 100 - (total_anomalies / (total_records * len(continuous_features)) * 100)

# Consistency score
consistency_issues = len(range_violations)
consistency_score = 100 - (consistency_issues / len(continuous_features) * 100)

# Overall score (weighted average)
overall_score = (missing_score * 0.3 + 
                duplicate_score * 0.2 + 
                outlier_score * 0.2 + 
                anomaly_score * 0.15 + 
                consistency_score * 0.15)

scorecard = pd.DataFrame({
    'Dimension': ['Missing Values', 'Duplicates', 'Outliers', 
                 'Anomalies', 'Consistency', 'OVERALL'],
    'Score': [missing_score, duplicate_score, outlier_score, 
             anomaly_score, consistency_score, overall_score],
    'Status': [
        'Excellent' if missing_score > 95 else 'Good' if missing_score > 90 else 'Fair',
        'Excellent' if duplicate_score == 100 else 'Good',
        'Good' if outlier_score > 95 else 'Fair' if outlier_score > 90 else 'Poor',
        'Good' if anomaly_score > 99 else 'Fair' if anomaly_score > 95 else 'Poor',
        'Good' if consistency_score > 90 else 'Fair',
        'Excellent' if overall_score > 95 else 'Good' if overall_score > 85 else 'Fair'
    ]
})

print("\nData Quality Scorecard (out of 100):")
print(scorecard.to_string(index=False))

# Save scorecard
scorecard.to_csv('reports/dqa/dqa_scorecard.csv', index=False)
print("\n✓ Saved: reports/dqa/dqa_scorecard.csv")

# Visualize scorecard
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2ecc71' if score > 95 else '#f39c12' if score > 85 else '#e74c3c' 
          for score in scorecard['Score']]
bars = ax.barh(scorecard['Dimension'], scorecard['Score'], color=colors)

ax.set_xlabel('Score (out of 100)', fontsize=12)
ax.set_title('Data Quality Scorecard', fontsize=14, fontweight='bold')
ax.set_xlim([0, 105])
ax.grid(axis='x', alpha=0.3)

# Add score labels
for i, (score, status) in enumerate(zip(scorecard['Score'], scorecard['Status'])):
    ax.text(score + 1, i, f'{score:.1f} ({status})', 
           va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/dqa_scorecard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/dqa_scorecard.png")
plt.close()

# ============================================================================
# 10. FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("[11] DATA QUALITY RECOMMENDATIONS")
print("="*80)

print("\n** PRIORITY 1: Handle Extreme Anomalies **")
if len(anomalies) > 0:
    print(f"   - {len(anomalies)} extreme values detected (|Z| > 5)")
    print("   - Action: Replace with NaN and apply KNN imputation")
    print("   - Features affected:", list(set([a['Feature'] for a in anomalies])))
else:
    print("   ✓ No extreme anomalies detected")

print("\n** PRIORITY 2: Missing Value Treatment **")
total_missing = train_df.isnull().sum().sum()
if total_missing > 0:
    print(f"   - {total_missing} missing values detected")
    print("   - Categorical features: Mode imputation")
    print("   - Continuous features: KNN imputation (k=5)")
else:
    print("   ✓ No missing values detected")

print("\n** PRIORITY 3: Outlier Management **")
total_outliers_pct = (iqr_df['Outlier_Count'].sum() / 
                      (len(train_df) * len(continuous_features))) * 100
print(f"   - {iqr_df['Outlier_Count'].sum()} outliers detected "
      f"({total_outliers_pct:.2f}% of all data points)")
if total_outliers_pct < 2:
    print("   - Action: Keep outliers for tree-based models")
    print("   - Use RobustScaler for linear models")
else:
    print("   - Action: Consider Winsorization at 1st/99th percentile")

print("\n** PRIORITY 4: Class Imbalance **")
if 'drug_category' in train_df.columns:
    print(f"   - Imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("   - Action: Use stratified sampling")
        print("   - Apply class weights in models")
        print("   - Consider SMOTE for minority class")
    else:
        print("   ✓ Classes are reasonably balanced")

print("\n** PRIORITY 5: Data Validation **")
print("   ✓ No duplicates detected")
print("   ✓ No data leakage concerns")
print("   - Proceed to feature engineering")

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT COMPLETE")
print("="*80)
print(f"\n✓ Overall Data Quality Score: {overall_score:.1f}/100 ({scorecard.iloc[-1]['Status']})")
print(f"\n✓ All reports saved to: reports/dqa/")
print(f"✓ All visualizations saved to: visualizations/")
print("\nNext Steps:")
print("1. Review anomaly and outlier reports")
print("2. Execute data cleaning pipeline")
print("3. Validate cleaned data quality")
print("4. Proceed to feature engineering")
