import pandas as pd
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------
# 1. DUMMY GROUND TRUTH (Actual) & MODEL PREDICTIONS
# -----------------------------------------------------------------------------
# Passed / Positive = 1, Failed / Negative = 0
y_actual    = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

# -----------------------------------------------------------------------------
# 2. SCIKIT-LEARN IMPLEMENTATION
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_actual, y_predicted)

# Format into a readable Pandas DataFrame
cm_df = pd.DataFrame(
    cm,
    index=['Actual: 0 (Fail)', 'Actual: 1 (Pass)'],
    columns=['Pred: 0 (Fail)', 'Pred: 1 (Pass)']
)

print("\n=== SCIKIT-LEARN CONFUSION MATRIX ===")
print(cm_df)