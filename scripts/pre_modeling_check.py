"""
Comprehensive Pre-Modeling Verification Script
===============================================
Performs final checks before model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

def run_pre_modeling_checks():
    """Run comprehensive pre-modeling verification"""
    
    base_dir = Path(__file__).parent.parent
    
    print('='*80)
    print('COMPREHENSIVE PRE-MODELING VERIFICATION')
    print('='*80)
    
    # 1. DATA FILES CHECK
    print('\n[1] DATA FILES VERIFICATION')
    print('-'*80)
    train = pd.read_csv(base_dir / 'data' / 'data_minihackathon_train_engineered.csv')
    test = pd.read_csv(base_dir / 'data' / 'data_minihackathon_test_engineered.csv')
    
    print(f'Train data: {train.shape[0]} rows x {train.shape[1]} columns')
    print(f'Test data:  {test.shape[0]} rows x {test.shape[1]} columns')
    
    # 2. TARGET VARIABLE CHECK
    print('\n[2] TARGET VARIABLE ANALYSIS')
    print('-'*80)
    print(f'Target column: drug_category')
    has_target_train = 'drug_category' in train.columns
    has_target_test = 'drug_category' in test.columns
    print(f'Target in train: {has_target_train}')
    print(f'Target in test: {has_target_test}')
    
    if has_target_train:
        print(f'\nTarget distribution:')
        target_dist = train['drug_category'].value_counts().sort_index()
        for cat, count in target_dist.items():
            pct = (count / len(train)) * 100
            print(f'  {cat}: {count:4d} ({pct:5.2f}%)')
        
        balance_ratio = target_dist.max() / target_dist.min()
        print(f'\nClass balance ratio: {balance_ratio:.2f}:1')
        if balance_ratio > 3:
            print('  WARNING: Imbalanced dataset - consider stratified sampling')
        else:
            print('  OK: Reasonably balanced')
    
    # 3. FEATURE CONSISTENCY
    print('\n[3] FEATURE CONSISTENCY CHECK')
    print('-'*80)
    train_features = set(train.columns) - {'drug_category', 'ID'}
    test_features = set(test.columns) - {'ID'}
    
    print(f'Train features: {len(train_features)}')
    print(f'Test features: {len(test_features)}')
    
    features_match = train_features == test_features
    if features_match:
        print('OK: Feature sets match perfectly')
    else:
        missing_test = train_features - test_features
        missing_train = test_features - train_features
        print(f'ERROR: Mismatch detected!')
        if missing_test:
            print(f'  Missing in test: {missing_test}')
        if missing_train:
            print(f'  Missing in train: {missing_train}')
    
    # 4. DATA TYPES
    print('\n[4] DATA TYPE VERIFICATION')
    print('-'*80)
    train_types = train.drop(['ID', 'drug_category'], axis=1, errors='ignore').dtypes
    numeric_count = sum(train_types.apply(lambda x: np.issubdtype(x, np.number)))
    print(f'Numeric features: {numeric_count}/{len(train_types)}')
    print(f'Non-numeric features: {len(train_types) - numeric_count}')
    
    all_numeric = numeric_count == len(train_types)
    if all_numeric:
        print('OK: All features are numeric')
    else:
        print('ERROR: Some non-numeric features found')
    
    # 5. MISSING VALUES
    print('\n[5] MISSING VALUES CHECK')
    print('-'*80)
    train_missing = train.drop(['ID'], axis=1, errors='ignore').isnull().sum().sum()
    test_missing = test.drop(['ID'], axis=1, errors='ignore').isnull().sum().sum()
    
    print(f'Train missing values: {train_missing}')
    print(f'Test missing values: {test_missing}')
    
    no_missing = train_missing == 0 and test_missing == 0
    if no_missing:
        print('OK: No missing values')
    else:
        print('WARNING: Missing values detected')
    
    # 6. DUPLICATES CHECK
    print('\n[6] DUPLICATE ROWS CHECK')
    print('-'*80)
    train_dupes = train.drop(['ID'], axis=1, errors='ignore').duplicated().sum()
    test_dupes = test.drop(['ID'], axis=1, errors='ignore').duplicated().sum()
    
    print(f'Train duplicates: {train_dupes}')
    print(f'Test duplicates: {test_dupes}')
    
    no_dupes = train_dupes == 0 and test_dupes == 0
    if no_dupes:
        print('OK: No duplicates')
    else:
        print('INFO: Duplicates found (may be expected)')
    
    # 7. FEATURE STATISTICS
    print('\n[7] FEATURE STATISTICS')
    print('-'*80)
    numeric_features = train.select_dtypes(include=[np.number]).drop(['ID'], axis=1, errors='ignore')
    if 'drug_category' in train.columns:
        numeric_features = numeric_features.drop(['drug_category'], axis=1, errors='ignore')
    
    print(f'Total features for modeling: {len(numeric_features.columns)}')
    print(f'Feature value ranges:')
    print(f'  Min: {numeric_features.min().min():.4f}')
    print(f'  Max: {numeric_features.max().max():.4f}')
    print(f'  Mean: {numeric_features.mean().mean():.4f}')
    print(f'  Median: {numeric_features.median().median():.4f}')
    
    # Check for infinite values
    inf_count = np.isinf(numeric_features.values).sum()
    print(f'\nInfinite values: {inf_count}')
    no_inf = inf_count == 0
    if no_inf:
        print('OK: No infinite values')
    else:
        print('ERROR: Infinite values detected')
    
    # 8. TOP FEATURES REVIEW
    print('\n[8] TOP 10 FEATURES BY TARGET CORRELATION')
    print('-'*80)
    if has_target_train:
        le = LabelEncoder()
        target_encoded = le.fit_transform(train['drug_category'])
        correlations = numeric_features.corrwith(pd.Series(target_encoded)).abs().sort_values(ascending=False)
        
        for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
            print(f'{i:2d}. {feat:45s} {corr:.4f}')
    
    # 9. PROBLEM TYPE IDENTIFICATION
    print('\n[9] PROBLEM TYPE IDENTIFICATION')
    print('-'*80)
    if has_target_train:
        n_classes = train['drug_category'].nunique()
        print(f'Number of classes: {n_classes}')
        print(f'Problem type: Multi-class Classification')
        print(f'Suitable models: Random Forest, XGBoost, LightGBM, CatBoost')
        print(f'Evaluation metrics: Accuracy, F1-Score (macro/weighted), Log Loss')
    
    # 10. FINAL READINESS CHECK
    print('\n' + '='*80)
    print('READINESS ASSESSMENT')
    print('='*80)
    
    checks = {
        'Data loaded': True,
        'Features match': features_match,
        'All numeric': all_numeric,
        'No missing values': no_missing,
        'No duplicates': no_dupes,
        'No infinite values': no_inf,
        'Target available': has_target_train
    }
    
    all_pass = all(checks.values())
    
    for check, status in checks.items():
        symbol = 'OK' if status else 'FAIL'
        print(f'[{symbol}] {check}')
    
    print('\n' + '='*80)
    if all_pass:
        print('STATUS: READY FOR MODEL TRAINING')
        print('='*80)
        return True
    else:
        print('STATUS: ISSUES NEED RESOLUTION')
        print('='*80)
        return False


if __name__ == '__main__':
    success = run_pre_modeling_checks()
    sys.exit(0 if success else 1)
