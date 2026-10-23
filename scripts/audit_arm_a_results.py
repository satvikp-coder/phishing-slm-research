import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

results_csv = r"c:\Users\satvi\Desktop\Amit Sir\arm_a_results_package\preds_qwen2_5_1_5b_naive_val.csv"
df = pd.read_csv(results_csv)

print(f"Total validation samples: {len(df)}")
print("Columns:", df.columns.tolist())
print("\n--- Ground Truth (y_true) ---")
print(df["y_true"].value_counts())
print("\n--- Predicted (y_pred) ---")
print(df["y_pred"].value_counts())
print(f"\nParse Success Rate: {df['parsed_successfully'].mean() * 100:.2f}%")

cm = confusion_matrix(df["y_true"], df["y_pred"])
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  TN (Legitimate -> Legitimate): {tn}")
print(f"  FP (Legitimate -> Phishing):   {fp}")
print(f"  FN (Phishing -> Legitimate):   {fn}")
print(f"  TP (Phishing -> Phishing):     {tp}")

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
print(f"\nFalse Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%)")

acc = accuracy_score(df["y_true"], df["y_pred"])
prec = precision_score(df["y_true"], df["y_pred"])
rec = recall_score(df["y_true"], df["y_pred"])
f1 = f1_score(df["y_true"], df["y_pred"])
auroc = roc_auc_score(df["y_true"], df["y_pred"])

print(f"\nSummary Metrics:")
print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  AUROC:     {auroc:.4f}")

print("\n--- Detailed Classification Report ---")
print(classification_report(df["y_true"], df["y_pred"], target_names=["Legitimate (0)", "Phishing (1)"], digits=4))

# Let's inspect raw_output distribution
print("\n--- Raw Output Unique Counts (Top 10) ---")
print(df["raw_output"].value_counts().head(10))
