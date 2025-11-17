"""
Adjust the Depressants prediction threshold to match expected distribution
"""
import pandas as pd
import numpy as np

# Load binary predictions
df = pd.read_csv('binary_predictions/depressants_predictions_20251116_212045.csv')

# Expected number of Depressants (16% of test set)
expected_depressants = int(377 * 0.16)  # 60 samples

print("="*80)
print("DEPRESSANTS THRESHOLD ADJUSTMENT")
print("="*80)

print(f"\nExpected Depressants (16% of 377): {expected_depressants}")
print(f"Current predictions at 0.5 threshold: {(df['depressant_probability'] > 0.5).sum()}")

# Strategy: Select top N samples with highest probabilities
# where N = expected number of Depressants
df_sorted = df.sort_values('depressant_probability', ascending=False)
top_n_indices = df_sorted.head(expected_depressants).index

# Create adjusted predictions
df['is_depressant_adjusted'] = 0
df.loc[top_n_indices, 'is_depressant_adjusted'] = 1

# Find the threshold used
threshold = df_sorted.iloc[expected_depressants-1]['depressant_probability']

print(f"\nOptimal threshold (to get {expected_depressants} predictions): {threshold:.4f}")
print(f"Predictions with adjusted threshold: {df['is_depressant_adjusted'].sum()}")

# Fix IDs: 501-877
df['id'] = range(501, 501 + len(df))

# Create submission with adjusted threshold
submission_adjusted = pd.DataFrame({
    'id': df['id'],
    'drug_category': ['Depressants' if pred == 1 else 'Others' for pred in df['is_depressant_adjusted']]
})

# Save adjusted submission
output_file = 'submission_binary_ADJUSTED_threshold.csv'
submission_adjusted.to_csv(output_file, index=False)

print(f"\n✅ Adjusted submission saved: {output_file}")
print(f"\nPrediction distribution:")
print(submission_adjusted['drug_category'].value_counts())

# Check specific ID 521
id_521_idx = 521 - 501
print(f"\n{'='*80}")
print(f"ID 521 CHECK:")
print(f"{'='*80}")
print(f"Probability: {df.iloc[id_521_idx]['depressant_probability']:.4f}")
print(f"Original prediction: {'Depressants' if df.iloc[id_521_idx]['is_depressant'] == 1 else 'Others'}")
print(f"Adjusted prediction: {'Depressants' if df.iloc[id_521_idx]['is_depressant_adjusted'] == 1 else 'Others'}")
print(f"Rank by probability: {df_sorted.index.tolist().index(id_521_idx) + 1} out of 377")

# Save detailed results
df_detailed = df[['id', 'depressant_probability', 'is_depressant', 'is_depressant_adjusted']].copy()
df_detailed['prediction_original'] = df_detailed['is_depressant'].map({0: 'Others', 1: 'Depressants'})
df_detailed['prediction_adjusted'] = df_detailed['is_depressant_adjusted'].map({0: 'Others', 1: 'Depressants'})
df_detailed = df_detailed.sort_values('depressant_probability', ascending=False)
df_detailed.to_csv('binary_predictions/depressants_predictions_DETAILED.csv', index=False)

print(f"\n✅ Detailed predictions saved: binary_predictions/depressants_predictions_DETAILED.csv")

# Show top 65 (around expected + margin)
print(f"\n{'='*80}")
print(f"TOP 65 MOST LIKELY DEPRESSANTS:")
print(f"{'='*80}")
print(df_detailed.head(65)[['id', 'depressant_probability', 'prediction_adjusted']].to_string(index=False))
