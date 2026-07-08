```python
FINAL_TEST_X_PATH       = "../test_final.csv"

final_test_df = pd.read_csv(FINAL_TEST_X_PATH)


preds = predict(ckpt, final_test_df)
assert len(preds) == len(final_test_df), "You must return one prediction for each row."

f1  = f1_score(final_test_df[TARGET], preds)
acc = accuracy_score(final_test_df[TARGET], preds)
print(f"Final F1 (leaderboard metric): {f1:.4f}")
print(f"Final accuracy (reference)   : {acc:.4f}")

```