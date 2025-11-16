"""
Feature Engineering Script
ICDS 2025 Mini-Hackathon - Drug Category Prediction
Creates domain-specific features, interaction terms, polynomial features,
and performs feature transformations for improved model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("FEATURE ENGINEERING PIPELINE - Drug Category Prediction")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONTINUOUS_FEATURES = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
                       'Impulsive', 'SS']
CATEGORICAL_FEATURES = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
BIG_FIVE = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']
RISK_FACTORS = ['Nscore', 'Impulsive', 'SS']

print("\nConfiguration:")
print(f"  Continuous features: {len(CONTINUOUS_FEATURES)}")
print(f"  Categorical features: {len(CATEGORICAL_FEATURES)}")
print(f"  Big Five traits: {len(BIG_FIVE)}")
print(f"  Risk factors: {len(RISK_FACTORS)}")

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("[1] LOADING CLEANED DATA")
print("="*80)

train_df = pd.read_csv('data/data_minihackathon_train_clean.csv')
print(f"\n✓ Training data loaded: {train_df.shape}")

try:
    test_df = pd.read_csv('data/data_minihackathon_test_clean.csv')
    print(f"✓ Test data loaded: {test_df.shape}")
    has_test = True
except FileNotFoundError:
    print("⚠️ Test data not found, processing training data only")
    test_df = None
    has_test = False

# Store original feature names
original_features = train_df.columns.tolist()
print(f"\nOriginal features: {len(original_features)}")

# Separate target if present
if 'drug_category' in train_df.columns:
    y_train = train_df['drug_category'].copy()
    X_train = train_df.drop('drug_category', axis=1)
    has_target = True
    print(f"✓ Target variable separated (drug_category)")
else:
    X_train = train_df.copy()
    y_train = None
    has_target = False
    print("⚠️ No target variable found")

# Save ID column
train_ids = X_train['ID'].copy() if 'ID' in X_train.columns else None
test_ids = test_df['ID'].copy() if has_test and 'ID' in test_df.columns else None

# Remove ID from features
if 'ID' in X_train.columns:
    X_train = X_train.drop('ID', axis=1)
if has_test and 'ID' in test_df.columns:
    X_test = test_df.drop('ID', axis=1)
else:
    X_test = test_df

print(f"\nFeature matrix shape: {X_train.shape}")

# ============================================================================
# STEP 1: DOMAIN-SPECIFIC PSYCHOLOGICAL RISK INDICATORS
# ============================================================================

print("\n" + "="*80)
print("[2] CREATING DOMAIN-SPECIFIC FEATURES")
print("="*80)

def create_psychological_features(df):
    """Create psychological risk indicators based on domain knowledge"""
    df_new = df.copy()
    
    # 1. High-Risk Personality Score (Neuroticism + Impulsivity - Conscientiousness)
    df_new['HighRisk_Score'] = (df_new['Nscore'] + 
                                 df_new['Impulsive'] - 
                                 df_new['Cscore'])
    
    # 2. Sensation-Seeking Risk Score (SS + Impulsive)
    df_new['SensationRisk_Score'] = (df_new['SS'] + 
                                     df_new['Impulsive'])
    
    # 3. Emotional Instability (Neuroticism - Conscientiousness)
    df_new['EmotionalInstability'] = (df_new['Nscore'] - 
                                      df_new['Cscore'])
    
    # 4. Conscientiousness Deficit (inverted)
    df_new['Conscientiousness_Deficit'] = -df_new['Cscore']
    
    # 5. Social Risk (Extraversion + Openness)
    df_new['Social_Risk'] = (df_new['Escore'] + 
                             df_new['Oscore'])
    
    # 6. Behavioral Disinhibition (Impulsive + SS - Cscore)
    df_new['Behavioral_Disinhibition'] = (df_new['Impulsive'] + 
                                          df_new['SS'] - 
                                          df_new['Cscore'])
    
    # 7. Neuroticism-Extraversion Balance
    df_new['NE_Balance'] = (df_new['Nscore'] - 
                            df_new['Escore'])
    
    # 8. Big Five Mean (average personality trait)
    df_new['BigFive_Mean'] = df_new[BIG_FIVE].mean(axis=1)
    
    # 9. Big Five Std (personality trait variability)
    df_new['BigFive_Std'] = df_new[BIG_FIVE].std(axis=1)
    
    # 10. Risk Factor Mean
    df_new['RiskFactor_Mean'] = df_new[RISK_FACTORS].mean(axis=1)
    
    return df_new

print("\nCreating psychological risk indicators...")
X_train_fe = create_psychological_features(X_train)
if has_test:
    X_test_fe = create_psychological_features(X_test)

new_features_count = len(X_train_fe.columns) - len(X_train.columns)
print(f"✓ Created {new_features_count} domain-specific features")

# ============================================================================
# STEP 2: INTERACTION TERMS
# ============================================================================

print("\n" + "="*80)
print("[3] CREATING INTERACTION FEATURES")
print("="*80)

def create_interaction_features(df):
    """Create interaction terms between related features"""
    df_new = df.copy()
    
    # Key interactions based on psychological theory
    interactions = [
        # Risk behavior interactions
        ('Impulsive', 'SS', 'Impulsive_x_SS'),
        ('Nscore', 'Impulsive', 'Nscore_x_Impulsive'),
        ('Nscore', 'SS', 'Nscore_x_SS'),
        
        # Personality trait interactions
        ('Nscore', 'Cscore', 'Nscore_x_Cscore'),
        ('Escore', 'Oscore', 'Escore_x_Oscore'),
        ('Ascore', 'Cscore', 'Ascore_x_Cscore'),
        
        # Protective vs risk factors
        ('Cscore', 'Impulsive', 'Cscore_x_Impulsive'),
        ('Cscore', 'SS', 'Cscore_x_SS'),
        ('Ascore', 'Impulsive', 'Ascore_x_Impulsive'),
        
        # Complex interactions
        ('Nscore', 'Escore', 'Nscore_x_Escore'),
        ('Oscore', 'Impulsive', 'Oscore_x_Impulsive'),
    ]
    
    for feat1, feat2, new_name in interactions:
        df_new[new_name] = df_new[feat1] * df_new[feat2]
    
    return df_new

print("\nCreating interaction terms...")
X_train_fe = create_interaction_features(X_train_fe)
if has_test:
    X_test_fe = create_interaction_features(X_test_fe)

interaction_count = 11
print(f"✓ Created {interaction_count} interaction features")

# ============================================================================
# STEP 3: POLYNOMIAL FEATURES
# ============================================================================

print("\n" + "="*80)
print("[4] CREATING POLYNOMIAL FEATURES")
print("="*80)

def create_polynomial_features(df, features):
    """Create polynomial features for capturing non-linear relationships"""
    df_new = df.copy()
    
    for feature in features:
        # Squared terms (quadratic)
        df_new[f'{feature}_squared'] = df_new[feature] ** 2
        
        # Cubic terms for key risk factors
        if feature in ['Impulsive', 'SS', 'Nscore']:
            df_new[f'{feature}_cubed'] = df_new[feature] ** 3
    
    return df_new

print("\nCreating polynomial features...")
poly_features = CONTINUOUS_FEATURES
X_train_fe = create_polynomial_features(X_train_fe, poly_features)
if has_test:
    X_test_fe = create_polynomial_features(X_test_fe, poly_features)

# Count polynomial features (7 squared + 3 cubed)
poly_count = len(poly_features) + 3
print(f"✓ Created {poly_count} polynomial features")
print(f"  - Squared terms: {len(poly_features)}")
print(f"  - Cubic terms: 3 (Impulsive, SS, Nscore)")

# ============================================================================
# STEP 4: RATIO FEATURES
# ============================================================================

print("\n" + "="*80)
print("[5] CREATING RATIO FEATURES")
print("="*80)

def create_ratio_features(df):
    """Create ratio features from related variables"""
    df_new = df.copy()
    
    # Risk/Protective ratios
    df_new['Impulsive_to_Cscore_ratio'] = (df_new['Impulsive'] / 
                                           (df_new['Cscore'] + 3))  # +3 to avoid division issues
    
    df_new['SS_to_Cscore_ratio'] = (df_new['SS'] / 
                                    (df_new['Cscore'] + 3))
    
    df_new['Nscore_to_Ascore_ratio'] = (df_new['Nscore'] / 
                                        (df_new['Ascore'] + 3))
    
    df_new['Risk_to_Protective_ratio'] = ((df_new['Nscore'] + df_new['Impulsive']) / 
                                          (df_new['Cscore'] + df_new['Ascore'] + 6))
    
    # Personality balance ratios
    df_new['NE_ratio'] = (df_new['Nscore'] / 
                          (df_new['Escore'] + 3))
    
    df_new['OC_ratio'] = (df_new['Oscore'] / 
                          (df_new['Cscore'] + 3))
    
    return df_new

print("\nCreating ratio features...")
X_train_fe = create_ratio_features(X_train_fe)
if has_test:
    X_test_fe = create_ratio_features(X_test_fe)

ratio_count = 6
print(f"✓ Created {ratio_count} ratio features")

# ============================================================================
# STEP 5: BINNING/DISCRETIZATION
# ============================================================================

print("\n" + "="*80)
print("[6] CREATING BINNED FEATURES")
print("="*80)

def create_binned_features(df):
    """Discretize continuous variables into categories"""
    df_new = df.copy()
    
    # Impulsivity levels (Low, Medium, High)
    df_new['Impulsive_Level'] = pd.cut(df_new['Impulsive'], 
                                       bins=[-np.inf, -0.5, 0.5, np.inf],
                                       labels=[0, 1, 2])
    
    # Sensation-seeking levels
    df_new['SS_Level'] = pd.cut(df_new['SS'], 
                                bins=[-np.inf, -0.5, 0.5, np.inf],
                                labels=[0, 1, 2])
    
    # Neuroticism levels
    df_new['Nscore_Level'] = pd.cut(df_new['Nscore'], 
                                    bins=[-np.inf, -0.5, 0.5, np.inf],
                                    labels=[0, 1, 2])
    
    # Conscientiousness levels
    df_new['Cscore_Level'] = pd.cut(df_new['Cscore'], 
                                    bins=[-np.inf, -0.5, 0.5, np.inf],
                                    labels=[0, 1, 2])
    
    # Convert to numeric
    for col in ['Impulsive_Level', 'SS_Level', 'Nscore_Level', 'Cscore_Level']:
        df_new[col] = df_new[col].astype(int)
    
    return df_new

print("\nCreating binned features...")
X_train_fe = create_binned_features(X_train_fe)
if has_test:
    X_test_fe = create_binned_features(X_test_fe)

binned_count = 4
print(f"✓ Created {binned_count} binned features (3 levels each)")

# ============================================================================
# STEP 6: FEATURE TRANSFORMATIONS
# ============================================================================

print("\n" + "="*80)
print("[7] APPLYING FEATURE TRANSFORMATIONS")
print("="*80)

def apply_transformations(df):
    """Apply transformations to handle skewness"""
    df_new = df.copy()
    
    # Get only continuous features that exist
    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in df_new.columns]
    
    # Log transformation (add constant to handle negative values)
    for col in continuous_cols:
        min_val = df_new[col].min()
        shift = abs(min_val) + 1 if min_val < 0 else 0
        df_new[f'{col}_log'] = np.log1p(df_new[col] + shift)
    
    # Square root transformation (for highly skewed features)
    for col in ['Impulsive', 'SS']:
        if col in df_new.columns:
            min_val = df_new[col].min()
            shift = abs(min_val) + 1 if min_val < 0 else 0
            df_new[f'{col}_sqrt'] = np.sqrt(df_new[col] + shift)
    
    return df_new

print("\nApplying transformations for skewed distributions...")
X_train_fe = apply_transformations(X_train_fe)
if has_test:
    X_test_fe = apply_transformations(X_test_fe)

transform_count = len(CONTINUOUS_FEATURES) + 2
print(f"✓ Created {transform_count} transformed features")
print(f"  - Log transformations: {len(CONTINUOUS_FEATURES)}")
print(f"  - Square root transformations: 2")

# ============================================================================
# STEP 7: ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("[8] ENCODING CATEGORICAL VARIABLES")
print("="*80)

# One-hot encoding for nominal variables
print("\n8.1 One-Hot Encoding (Gender, Country, Ethnicity)...")
nominal_features = ['Gender', 'Country', 'Ethnicity']
nominal_features = [f for f in nominal_features if f in X_train_fe.columns]

if len(nominal_features) > 0:
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Fit on training data
    encoded_train = encoder.fit_transform(X_train_fe[nominal_features])
    feature_names = encoder.get_feature_names_out(nominal_features)
    
    # Create DataFrames
    encoded_train_df = pd.DataFrame(encoded_train, 
                                    columns=feature_names,
                                    index=X_train_fe.index)
    
    # Drop original and add encoded
    X_train_fe = X_train_fe.drop(nominal_features, axis=1)
    X_train_fe = pd.concat([X_train_fe, encoded_train_df], axis=1)
    
    if has_test:
        encoded_test = encoder.transform(X_test_fe[nominal_features])
        encoded_test_df = pd.DataFrame(encoded_test, 
                                       columns=feature_names,
                                       index=X_test_fe.index)
        X_test_fe = X_test_fe.drop(nominal_features, axis=1)
        X_test_fe = pd.concat([X_test_fe, encoded_test_df], axis=1)
    
    onehot_count = len(feature_names)
    print(f"✓ Created {onehot_count} one-hot encoded features")

# Ordinal encoding for ordinal variables
print("\n8.2 Ordinal Encoding (Age, Education)...")
ordinal_features = ['Age', 'Education']
ordinal_features = [f for f in ordinal_features if f in X_train_fe.columns]

if len(ordinal_features) > 0:
    # Age and Education are already numerically encoded
    # Just keep them as is (they represent ordinal relationships)
    print(f"✓ Retained {len(ordinal_features)} ordinal features (already encoded)")

# ============================================================================
# STEP 8: FEATURE SCALING
# ============================================================================

print("\n" + "="*80)
print("[9] FEATURE SCALING")
print("="*80)

# Identify numeric columns (exclude categorical encoded features)
numeric_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nScaling {len(numeric_cols)} numeric features...")

# StandardScaler for normally distributed features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe[numeric_cols])
X_train_scaled_df = pd.DataFrame(X_train_scaled, 
                                 columns=numeric_cols,
                                 index=X_train_fe.index)

if has_test:
    X_test_scaled = scaler.transform(X_test_fe[numeric_cols])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, 
                                    columns=numeric_cols,
                                    index=X_test_fe.index)

print(f"✓ Applied StandardScaler to {len(numeric_cols)} features")

# ============================================================================
# STEP 9: FEATURE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("[10] FEATURE ENGINEERING SUMMARY")
print("="*80)

feature_summary = {
    'Category': [
        'Original Features',
        'Domain-Specific (Psychological)',
        'Interaction Terms',
        'Polynomial Features',
        'Ratio Features',
        'Binned Features',
        'Transformed Features',
        'One-Hot Encoded',
        'TOTAL ENGINEERED FEATURES'
    ],
    'Count': [
        len(original_features) - 1,  # Exclude ID
        new_features_count,
        interaction_count,
        poly_count,
        ratio_count,
        binned_count,
        transform_count,
        onehot_count if len(nominal_features) > 0 else 0,
        len(X_train_scaled_df.columns)
    ]
}

summary_df = pd.DataFrame(feature_summary)
print("\nFeature Engineering Summary:")
print(summary_df.to_string(index=False))

print(f"\n✓ Total features created: {len(X_train_scaled_df.columns)}")
print(f"  Original: {len(original_features) - 1}")
print(f"  New: {len(X_train_scaled_df.columns) - (len(original_features) - 1)}")

# ============================================================================
# STEP 10: SAVE ENGINEERED FEATURES
# ============================================================================

print("\n" + "="*80)
print("[11] SAVING ENGINEERED FEATURES")
print("="*80)

# Add back ID and target
if train_ids is not None:
    X_train_scaled_df.insert(0, 'ID', train_ids)

if has_target:
    X_train_scaled_df['drug_category'] = y_train.values

# Save training features
train_output_path = 'data/data_minihackathon_train_engineered.csv'
X_train_scaled_df.to_csv(train_output_path, index=False)
print(f"\n✓ Training features saved: {train_output_path}")
print(f"  Shape: {X_train_scaled_df.shape}")

# Save test features
if has_test:
    if test_ids is not None:
        X_test_scaled_df.insert(0, 'ID', test_ids)
    
    test_output_path = 'data/data_minihackathon_test_engineered.csv'
    X_test_scaled_df.to_csv(test_output_path, index=False)
    print(f"✓ Test features saved: {test_output_path}")
    print(f"  Shape: {X_test_scaled_df.shape}")

# Save feature names
feature_names_df = pd.DataFrame({
    'Feature_Name': X_train_scaled_df.columns.tolist(),
    'Feature_Index': range(len(X_train_scaled_df.columns))
})
feature_names_df.to_csv('reports/feature_names.csv', index=False)
print(f"✓ Feature names saved: reports/feature_names.csv")

# Save feature engineering summary
summary_df.to_csv('reports/feature_engineering_summary.csv', index=False)
print(f"✓ Summary saved: reports/feature_engineering_summary.csv")

# ============================================================================
# STEP 11: FEATURE IMPORTANCE PREVIEW
# ============================================================================

print("\n" + "="*80)
print("[12] FEATURE STATISTICS PREVIEW")
print("="*80)

# Show correlation with some key engineered features
if has_target:
    print("\nTop 20 Features by Absolute Correlation with Target:")
    
    # Prepare data for correlation
    corr_df = X_train_scaled_df.copy()
    numeric_features = corr_df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlations = {}
    for feature in numeric_features:
        if feature not in ['drug_category']:
            try:
                corr = abs(corr_df[feature].corr(corr_df['drug_category']))
                correlations[feature] = corr
            except:
                pass
    
    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
    
    corr_preview = pd.DataFrame(sorted_corr, columns=['Feature', 'Abs_Correlation'])
    print(corr_preview.to_string(index=False))
    
    # Save correlation analysis
    corr_full = pd.DataFrame(sorted(correlations.items(), key=lambda x: x[1], reverse=True),
                            columns=['Feature', 'Abs_Correlation'])
    corr_full.to_csv('reports/feature_correlations.csv', index=False)
    print(f"\n✓ Full correlation analysis saved: reports/feature_correlations.csv")

# ============================================================================
# STEP 12: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("[13] GENERATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Feature count by category
fig, ax = plt.subplots(figsize=(12, 6))

categories = summary_df['Category'][:-1]  # Exclude total
counts = summary_df['Count'][:-1]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
          '#9b59b6', '#1abc9c', '#34495e', '#e67e22']

bars = ax.bar(range(len(categories)), counts, color=colors[:len(categories)])

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_ylabel('Feature Count', fontsize=12)
ax.set_title('Feature Engineering Summary by Category', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add count labels
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
           f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/feature_engineering_summary.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/feature_engineering_summary.png")
plt.close()

# Visualization 2: Top correlations (if target available)
if has_target and len(sorted_corr) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = [x[0] for x in sorted_corr[:15]]
    top_corrs = [x[1] for x in sorted_corr[:15]]
    
    bars = ax.barh(range(len(top_features)), top_corrs, color='#3498db')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Absolute Correlation with Target', fontsize=12)
    ax.set_title('Top 15 Features by Correlation with Drug Category', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add correlation values
    for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
        ax.text(corr + 0.005, bar.get_y() + bar.get_height()/2,
               f'{corr:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/top_feature_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/top_feature_correlations.png")
    plt.close()

# ============================================================================
# FINAL MESSAGE
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)

print("\n✓ All feature engineering steps completed successfully")

print("\nEngineered Data Files:")
print("  - data/data_minihackathon_train_engineered.csv")
if has_test:
    print("  - data/data_minihackathon_test_engineered.csv")

print("\nReports Generated:")
print("  - reports/feature_names.csv")
print("  - reports/feature_engineering_summary.csv")
if has_target:
    print("  - reports/feature_correlations.csv")

print("\nVisualizations Generated:")
print("  - visualizations/feature_engineering_summary.png")
if has_target:
    print("  - visualizations/top_feature_correlations.png")

print("\nFeature Engineering Highlights:")
print(f"  ✓ {new_features_count} domain-specific psychological features")
print(f"  ✓ {interaction_count} interaction terms")
print(f"  ✓ {poly_count} polynomial features")
print(f"  ✓ {ratio_count} ratio features")
print(f"  ✓ {binned_count} binned features")
print(f"  ✓ {transform_count} transformed features")
print(f"  ✓ All features scaled using StandardScaler")

print("\nNext Steps:")
print("  1. Review feature correlations")
print("  2. Perform feature selection (if needed)")
print("  3. Begin model development")
print("  4. Use cross-validation for model evaluation")
print("  5. Tune hyperparameters")

print("\n" + "="*80)
