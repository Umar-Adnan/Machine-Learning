import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# 1 = Passed / Positive, 0 = Failed / Negative
y_actual    = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

# -----------------------------------------------------------------------------
# CALCULATE INDIVIDUAL METRICS
# -----------------------------------------------------------------------------
accuracy  = accuracy_score(y_actual, y_predicted)
precision = precision_score(y_actual, y_predicted)
recall    = recall_score(y_actual, y_predicted)
f1        = f1_score(y_actual, y_predicted)

# -----------------------------------------------------------------------------
# DISPLAY SCORES
# -----------------------------------------------------------------------------
print("=== CLASSIFICATION EVALUATION METRICS ===")
print(f"Accuracy : {accuracy * 100:.2f}%  (Overall correct predictions)")
print(f"Precision: {precision * 100:.2f}%  (Out of all positive predictions, how many were right)")
print(f"Recall   : {recall * 100:.2f}%  (Out of all actual positives, how many were caught)")
print(f"F1-Score : {f1 * 100:.2f}%  (Harmonic mean of Precision and Recall)\n")

# -----------------------------------------------------------------------------
# COMPREHENSIVE SUMMARY
# -----------------------------------------------------------------------------
print("=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_actual, y_predicted, target_names=['Failed (0)', 'Passed (1)']))