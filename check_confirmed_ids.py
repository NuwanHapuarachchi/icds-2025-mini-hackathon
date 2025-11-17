import pandas as pd

df = pd.read_csv('binary_predictions/depressants_predictions_DETAILED.csv')
ids = [513, 521, 570, 642, 770]

print('='*80)
print('YOUR CONFIRMED DEPRESSANTS IDs - MODEL PERFORMANCE:')
print('='*80)

for id_val in ids:
    row = df[df['id'] == id_val].iloc[0]
    df_sorted = df.sort_values('depressant_probability', ascending=False)
    rank = list(df_sorted['id']).index(id_val) + 1
    print(f"\nID {id_val}:")
    print(f"  Probability: {row['depressant_probability']:.4f}")
    print(f"  Rank: {rank}/377")
    print(f"  Predicted: {row['prediction_adjusted']}")
    print(f"  ❌ MISSED!" if row['prediction_adjusted'] == 'Others' else "  ✓ CAUGHT")

# Calculate how many we're missing
df['is_confirmed'] = df['id'].isin(ids).astype(int)
caught = df[(df['is_confirmed'] == 1) & (df['prediction_adjusted'] == 'Depressants')].shape[0]
missed = df[(df['is_confirmed'] == 1) & (df['prediction_adjusted'] == 'Others')].shape[0]

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Confirmed Depressants: {len(ids)}")
print(f"  Caught by model: {caught}")
print(f"  Missed by model: {missed}")
print(f"  Success rate: {caught/len(ids)*100:.1f}%")
print('='*80)
