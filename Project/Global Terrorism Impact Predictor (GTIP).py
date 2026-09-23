import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# -----------------------------------------------------------------------------
df = pd.read_csv("global_terrorism_2000_2017_subset.csv")

# Keep a subset of features to avoid massive memory usage
features_to_use = ['success', 'suicide', 'nkill', 'attacktype1_txt', 'region_txt']
df = df[features_to_use].dropna() # Drop rows with missing values (NaNs in nkill)

# Filter out extreme outliers in 'nkill' to make regression visualization readable
df = df[df['nkill'] < 50]

# One-Hot Encode categorical text columns into 1s and 0s
df_encoded = pd.get_dummies(df, columns=['attacktype1_txt', 'region_txt'], drop_first=True)

# -----------------------------------------------------------------------------
# 2. CLASSIFICATION PIPELINE: Predict 'success' (Decision Tree)
# -----------------------------------------------------------------------------
# Target is 'success' (0 = Fail, 1 = Success). Features are everything else.
X_clf = df_encoded.drop(columns=['success'])
y_clf = df_encoded['success']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

clf_model = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)

print("=== CLASSIFICATION REPORT (Predicting Attack Success) ===")
print(classification_report(y_test_c, y_pred_c))

# -----------------------------------------------------------------------------
# 3. REGRESSION PIPELINE: Predict 'nkill' (Linear Regression)
# -----------------------------------------------------------------------------
# Target is 'nkill' (Continuous fatalities). Features are everything else.
X_reg = df_encoded.drop(columns=['nkill'])
y_reg = df_encoded['nkill']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

print("=== REGRESSION METRICS (Predicting Fatalities) ===")
print(f"MAE : {mean_absolute_error(y_test_r, y_pred_r):.2f} average deaths off")
print(f"R²  : {r2_score(y_test_r, y_pred_r):.4f}\n")

# -----------------------------------------------------------------------------
# 4. DATA VISUALIZATION (Matplotlib & Seaborn)
# -----------------------------------------------------------------------------
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Classification Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Pred: Fail', 'Pred: Success'],
            yticklabels=['Actual: Fail', 'Actual: Success'])
axes[0].set_title('Classification: Predict Attack Success\n(Confusion Matrix)', fontsize=14, pad=15)

# Plot 2: Regression Actual vs Predicted Scatter
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.3, color='crimson', ax=axes[1])
# Draw a perfect prediction line (y = x)
max_val = max(y_test_r.max(), y_pred_r.max())
axes[1].plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Perfect Prediction')
axes[1].set_xlabel('Actual Fatalities (nkill)')
axes[1].set_ylabel('Predicted Fatalities (nkill)')
axes[1].set_title('Regression: Predict Fatalities\n(Actual vs. Predicted)', fontsize=14, pad=15)
axes[1].legend()

plt.tight_layout()
plt.show()