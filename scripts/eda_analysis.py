"""
Comprehensive Exploratory Data Analysis (EDA) Script
Drug Category Prediction - ICDS 2025 Mini-Hackathon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
train_df = pd.read_csv(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\data\data_minihackathon_train.csv')
print(f"Data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

# Define feature groups
demographic_features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
personality_features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']
behavioral_features = ['Impulsive', 'SS']
all_features = demographic_features + personality_features + behavioral_features

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS")
print("="*80)

# 1. UNIVARIATE ANALYSIS - CONTINUOUS FEATURES
print("\n1. Distribution of Personality Traits and Behavioral Features")

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

continuous_features = personality_features + behavioral_features

for idx, feature in enumerate(continuous_features):
    # Histogram
    ax1 = axes[idx*2]
    train_df[feature].hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_title(f'{feature} - Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')
    ax1.axvline(train_df[feature].mean(), color='red', linestyle='--', label='Mean')
    ax1.axvline(train_df[feature].median(), color='green', linestyle='--', label='Median')
    ax1.legend()
    
    # Box plot
    ax2 = axes[idx*2 + 1]
    train_df.boxplot(column=feature, ax=ax2, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='black'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))
    ax2.set_title(f'{feature} - Box Plot', fontsize=12, fontweight='bold')
    ax2.set_ylabel(feature)

# Hide unused subplots
for idx in range(len(continuous_features)*2, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_univariate_continuous.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_univariate_continuous.png")
plt.close()

# 2. UNIVARIATE ANALYSIS - CATEGORICAL FEATURES (Demographics)
print("\n2. Distribution of Demographic Features")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(demographic_features):
    ax = axes[idx]
    value_counts = train_df[feature].value_counts().sort_index()
    value_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black', alpha=0.8)
    ax.set_title(f'{feature} - Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=9)

axes[-1].axis('off')  # Hide last subplot

plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_univariate_categorical.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_univariate_categorical.png")
plt.close()

# Statistical summary
print("\n3. Statistical Summary of Continuous Features")
summary_stats = train_df[continuous_features].describe().T
summary_stats['skewness'] = train_df[continuous_features].skew()
summary_stats['kurtosis'] = train_df[continuous_features].kurtosis()
print(summary_stats.to_string())

print("\n" + "="*80)
print("BIVARIATE ANALYSIS")
print("="*80)

# 4. BIVARIATE ANALYSIS - Violin Plots
print("\n4. Violin Plots: Features vs Drug Category")

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for idx, feature in enumerate(continuous_features):
    ax = axes[idx]
    sns.violinplot(data=train_df, x='drug_category', y=feature, ax=ax, palette='Set2')
    ax.set_title(f'{feature} by Drug Category', fontsize=11, fontweight='bold')
    ax.set_xlabel('Drug Category')
    ax.set_ylabel(feature)
    ax.tick_params(axis='x', rotation=45)

# Hide unused subplots
for idx in range(len(continuous_features), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_bivariate_violin.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_bivariate_violin.png")
plt.close()

# 5. BIVARIATE ANALYSIS - Box Plots by Category
print("\n5. Box Plots: Features vs Drug Category")

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for idx, feature in enumerate(continuous_features):
    ax = axes[idx]
    sns.boxplot(data=train_df, x='drug_category', y=feature, ax=ax, palette='viridis')
    ax.set_title(f'{feature} by Drug Category', fontsize=11, fontweight='bold')
    ax.set_xlabel('Drug Category')
    ax.set_ylabel(feature)
    ax.tick_params(axis='x', rotation=45)

# Hide unused subplots
for idx in range(len(continuous_features), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_bivariate_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_bivariate_boxplot.png")
plt.close()

# 6. Grouped Statistics
print("\n6. Grouped Statistics by Drug Category")
grouped_stats = train_df.groupby('drug_category')[continuous_features].agg(['mean', 'median', 'std', 'min', 'max'])
print(grouped_stats.to_string())

# 7. Demographic features vs target
print("\n7. Demographic Features vs Drug Category")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for idx, feature in enumerate(demographic_features):
    ax = axes[idx]
    crosstab = pd.crosstab(train_df[feature], train_df['drug_category'], normalize='index') * 100
    crosstab.plot(kind='bar', ax=ax, stacked=False, colormap='tab10', edgecolor='black')
    ax.set_title(f'{feature} vs Drug Category', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Drug Category')
    ax.tick_params(axis='x', rotation=45)

axes[-1].axis('off')

plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_bivariate_demographics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_bivariate_demographics.png")
plt.close()

print("\n" + "="*80)
print("MULTIVARIATE ANALYSIS")
print("="*80)

# 8. Correlation Matrix
print("\n8. Correlation Matrix")

plt.figure(figsize=(14, 12))
corr_matrix = train_df[continuous_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Personality & Behavioral Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_correlation_matrix.png")
plt.close()

# 9. Pair Plot for Key Features
print("\n9. Pair Plot for Key Features (Openness, Conscientiousness, Age, Impulsive)")

key_features = ['Oscore', 'Cscore', 'Nscore', 'Impulsive', 'SS', 'drug_category']
pairplot_data = train_df[key_features].dropna()

g = sns.pairplot(pairplot_data, hue='drug_category', palette='Set1', 
                 diag_kind='kde', plot_kws={'alpha': 0.6}, height=2.5)
g.fig.suptitle('Pair Plot - Key Discriminative Features', y=1.01, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_pairplot.png")
plt.close()

# 10. Correlation with target (encoded)
print("\n10. Feature Correlation with Target (Encoded)")

# Encode target
target_encoded = pd.get_dummies(train_df['drug_category'], prefix='target')
combined_df = pd.concat([train_df[continuous_features], target_encoded], axis=1)

plt.figure(figsize=(8, 10))
target_corr = combined_df.corr()[['target_Depressants', 'target_Hallucinogens', 'target_Stimulants']].drop(['target_Depressants', 'target_Hallucinogens', 'target_Stimulants'])
sns.heatmap(target_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
            linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation with Drug Categories', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_target_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_target_correlation.png")
plt.close()

print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# 11. ANOVA/Kruskal-Wallis Tests
print("\n11. Statistical Significance Tests (ANOVA/Kruskal-Wallis)")
print("\nTesting if feature distributions differ significantly across drug categories:")

statistical_results = []

for feature in continuous_features:
    # Get data for each category
    groups = [train_df[train_df['drug_category'] == cat][feature].dropna() for cat in train_df['drug_category'].unique()]
    
    # Shapiro-Wilk test for normality (on first group as sample)
    if len(groups[0]) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(groups[0][:min(5000, len(groups[0]))])
        is_normal = shapiro_p > 0.05
    else:
        is_normal = False
    
    # Choose test based on normality
    if is_normal:
        # ANOVA for normal distributions
        f_stat, p_value = f_oneway(*groups)
        test_used = 'ANOVA'
    else:
        # Kruskal-Wallis for non-normal distributions
        h_stat, p_value = kruskal(*groups)
        test_used = 'Kruskal-Wallis'
    
    statistical_results.append({
        'Feature': feature,
        'Test': test_used,
        'Statistic': f_stat if is_normal else h_stat,
        'P-Value': p_value,
        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'No'
    })

stats_df = pd.DataFrame(statistical_results)
print(stats_df.to_string(index=False))

# 12. Chi-Square Tests for Categorical Features
print("\n12. Chi-Square Tests for Categorical Features vs Drug Category")

chi_square_results = []

for feature in demographic_features:
    contingency_table = pd.crosstab(train_df[feature].dropna(), train_df.loc[train_df[feature].notna(), 'drug_category'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    chi_square_results.append({
        'Feature': feature,
        'Chi-Square': chi2,
        'P-Value': p_value,
        'DOF': dof,
        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'No'
    })

chi_df = pd.DataFrame(chi_square_results)
print(chi_df.to_string(index=False))

# 13. Effect Size (Eta-squared) for significant features
print("\n13. Effect Size Analysis (Eta-Squared for ANOVA)")

effect_sizes = []

for feature in continuous_features:
    groups = [train_df[train_df['drug_category'] == cat][feature].dropna() for cat in train_df['drug_category'].unique()]
    
    # Calculate between-group and within-group variance
    grand_mean = train_df[feature].mean()
    
    # Between-group sum of squares
    ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
    
    # Total sum of squares
    ss_total = sum([(x - grand_mean)**2 for g in groups for x in g])
    
    # Eta-squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    effect_sizes.append({
        'Feature': feature,
        'Eta-Squared': eta_squared,
        'Effect Size': 'Large' if eta_squared > 0.14 else 'Medium' if eta_squared > 0.06 else 'Small'
    })

effect_df = pd.DataFrame(effect_sizes)
print(effect_df.to_string(index=False))

# 14. Post-hoc pairwise comparisons for top features
print("\n14. Post-hoc Analysis: Pairwise Mean Differences")
print("\nTop 3 features with largest effect sizes:")

top_features = effect_df.nlargest(3, 'Eta-Squared')['Feature'].tolist()

for feature in top_features:
    print(f"\n{feature}:")
    print("-" * 60)
    
    categories = train_df['drug_category'].unique()
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            group1 = train_df[train_df['drug_category'] == cat1][feature].dropna()
            group2 = train_df[train_df['drug_category'] == cat2][feature].dropna()
            
            mean_diff = group1.mean() - group2.mean()
            print(f"  {cat1} vs {cat2}: Mean Difference = {mean_diff:.4f}")

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT FOR NEXT STEPS")
print("="*80)

# 15. Missing Values Analysis
print("\n15. Missing Values Analysis")

missing_analysis = pd.DataFrame({
    'Feature': train_df.columns,
    'Missing_Count': train_df.isnull().sum(),
    'Missing_Percentage': (train_df.isnull().sum() / len(train_df)) * 100,
    'Data_Type': train_df.dtypes
})
missing_analysis = missing_analysis[missing_analysis['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_analysis.to_string(index=False))

# Visualize missing pattern
plt.figure(figsize=(10, 6))
missing_data = train_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_data.plot(kind='barh', color='coral', edgecolor='black')
plt.xlabel('Number of Missing Values')
plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Downloads\Compressed\icds-2025-mini-hackathon\visualizations\eda_missing_values_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_missing_values_detailed.png")
plt.close()

# 16. Outlier Detection
print("\n16. Outlier Detection (IQR Method)")

outlier_summary = []

for feature in continuous_features:
    Q1 = train_df[feature].quantile(0.25)
    Q3 = train_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)][feature]
    
    outlier_summary.append({
        'Feature': feature,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': (len(outliers) / len(train_df)) * 100,
        'Min_Value': train_df[feature].min(),
        'Max_Value': train_df[feature].max()
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# 17. Z-Score Outlier Detection
print("\n17. Outlier Detection (Z-Score Method, |Z| > 3)")

zscore_summary = []

for feature in continuous_features:
    z_scores = np.abs(stats.zscore(train_df[feature].dropna()))
    outliers_z = train_df[feature].dropna()[z_scores > 3]
    
    zscore_summary.append({
        'Feature': feature,
        'Z_Outlier_Count': len(outliers_z),
        'Z_Outlier_Percentage': (len(outliers_z) / len(train_df)) * 100
    })

zscore_df = pd.DataFrame(zscore_summary)
print(zscore_df.to_string(index=False))

# 18. Anomaly Detection - Extreme Values
print("\n18. Anomaly Detection - Flagging Suspicious Values")

anomalies = []

# Check for Impulsive = 10 (mentioned as suspicious)
impulsive_10 = train_df[train_df['Impulsive'] == 10.0]
if len(impulsive_10) > 0:
    anomalies.append({
        'Feature': 'Impulsive',
        'Anomaly_Type': 'Extreme Value (=10)',
        'Count': len(impulsive_10),
        'Action': 'Investigate and potentially cap or treat as missing'
    })

# Check for values beyond reasonable ranges
for feature in continuous_features:
    extreme_high = train_df[train_df[feature] > train_df[feature].mean() + 5*train_df[feature].std()]
    extreme_low = train_df[train_df[feature] < train_df[feature].mean() - 5*train_df[feature].std()]
    
    if len(extreme_high) > 0:
        anomalies.append({
            'Feature': feature,
            'Anomaly_Type': 'Extreme High (>5 SD)',
            'Count': len(extreme_high),
            'Action': 'Investigate domain validity'
        })
    
    if len(extreme_low) > 0:
        anomalies.append({
            'Feature': feature,
            'Anomaly_Type': 'Extreme Low (<-5 SD)',
            'Count': len(extreme_low),
            'Action': 'Investigate domain validity'
        })

if anomalies:
    anomaly_df = pd.DataFrame(anomalies)
    print(anomaly_df.to_string(index=False))
else:
    print("No extreme anomalies detected beyond outliers already identified.")

# 19. Duplicate Check
print("\n19. Duplicate Records Check")

duplicates = train_df.duplicated().sum()
duplicate_features = train_df.duplicated(subset=all_features).sum()

print(f"Exact duplicate rows: {duplicates}")
print(f"Duplicate rows (excluding ID): {duplicate_features}")

if duplicate_features > 0:
    print("\nSample duplicate records:")
    print(train_df[train_df.duplicated(subset=all_features, keep=False)].head(10))

# 20. Data Consistency Checks
print("\n20. Data Consistency Checks")

consistency_issues = []

# Check if categorical encoding is consistent
for feature in demographic_features:
    unique_values = train_df[feature].dropna().unique()
    print(f"{feature}: {len(unique_values)} unique values")
    if len(unique_values) > 20:
        consistency_issues.append(f"{feature} has unusually many categories: {len(unique_values)}")

# Check for logical inconsistencies
# Example: All scores should be standardized (roughly -3 to +3 range for normalized data)
for feature in personality_features + behavioral_features:
    min_val = train_df[feature].min()
    max_val = train_df[feature].max()
    if min_val < -5 or max_val > 5:
        consistency_issues.append(f"{feature} has values outside expected standardized range: [{min_val:.2f}, {max_val:.2f}]")

if consistency_issues:
    print("\nPotential consistency issues:")
    for issue in consistency_issues:
        print(f"  - {issue}")
else:
    print("No major consistency issues detected.")

# 21. Feature Engineering Opportunities
print("\n" + "="*80)
print("FEATURE ENGINEERING OPPORTUNITIES")
print("="*80)

print("""
Identified opportunities for feature engineering:

1. INTERACTION FEATURES:
   - Impulsive × SS (risk-taking behavior)
   - Nscore × Impulsive (anxious impulsivity)
   - Oscore × SS (exploratory behavior)
   - Cscore × Impulsive (self-control index)

2. POLYNOMIAL FEATURES:
   - Oscore² (non-linear openness effect)
   - Cscore² (non-linear conscientiousness effect)

3. AGGREGATIONS:
   - Big Five mean/std (overall personality profile)
   - Emotional instability: Nscore - Cscore
   - Exploratory tendency: Oscore + SS

4. BINNING:
   - Age groups (already categorical)
   - Education levels (group similar levels)
   - Personality quintiles

5. ENCODING:
   - One-hot encoding for Country, Ethnicity (nominal)
   - Ordinal encoding for Age, Education (ordered)
   - Target encoding for high-cardinality categoricals

6. DOMAIN-SPECIFIC:
   - Risk profile score: w1*Nscore + w2*Impulsive + w3*SS
   - Stability index: Cscore - Nscore
   - Social behavior: Escore + Ascore
""")

# 22. Save comprehensive summary
print("\n22. Saving Comprehensive EDA Summary...")

summary_dict = {
    'statistical_tests': stats_df,
    'chi_square_tests': chi_df,
    'effect_sizes': effect_df,
    'missing_values': missing_analysis,
    'outliers_iqr': outlier_df,
    'outliers_zscore': zscore_df,
    'grouped_statistics': grouped_stats
}

# Save to CSV
for name, df in summary_dict.items():
    filepath = f'C:\\Users\\HP\\Downloads\\Compressed\\icds-2025-mini-hackathon\\reports\\eda_{name}.csv'
    df.to_csv(filepath)
    print(f"✓ Saved: eda_{name}.csv")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"""
Generated Visualizations:
  • eda_univariate_continuous.png
  • eda_univariate_categorical.png
  • eda_bivariate_violin.png
  • eda_bivariate_boxplot.png
  • eda_bivariate_demographics.png
  • eda_correlation_matrix.png
  • eda_pairplot.png
  • eda_target_correlation.png
  • eda_missing_values_detailed.png

Generated Reports:
  • eda_statistical_tests.csv
  • eda_chi_square_tests.csv
  • eda_effect_sizes.csv
  • eda_missing_values.csv
  • eda_outliers_iqr.csv
  • eda_outliers_zscore.csv
  • eda_grouped_statistics.csv

Next Steps:
  3. Data Quality Assessment ✓ (Completed in this analysis)
  4. Data Cleaning (Ready to proceed)
  5. Feature Engineering (Opportunities identified)
  6. Feature Selection (Ready after engineering)
""")
