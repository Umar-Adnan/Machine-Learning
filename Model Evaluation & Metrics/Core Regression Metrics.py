import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. CREATE CONTINUOUS DATASET (Study & Sleep Hours vs Exam Score)
# -----------------------------------------------------------------------------
data = {
    'Study_Hours': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'Sleep_Hours': [5.0, 6.0, 5.5, 7.0, 6.5, 8.0, 7.5, 8.5, 8.0, 9.0],
    'Exam_Score':  [45.0, 52.0, 60.0, 68.0, 73.0, 81.0, 85.0, 92.0, 95.0, 98.0]  # Continuous Target
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Sleep_Hours']]
y = df['Exam_Score']

# -----------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------------------------------------------
# 3. TRAIN LINEAR REGRESSION MODEL
# -----------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions on unseen test data
y_pred = model.predict(X_test)

# -----------------------------------------------------------------------------
# 4. CALCULATE REGRESSION METRICS
# -----------------------------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Square root of MSE converts error back to score units
r2   = r2_score(y_test, y_pred)

# -----------------------------------------------------------------------------
# 5. DISPLAY RESULTS & METRIC BREAKDOWN
# -----------------------------------------------------------------------------
print("=== REGRESSION EVALUATION METRICS ===")
print(f"Mean Absolute Error (MAE)  : {mae:.2f} score points")
print(f"Mean Squared Error (MSE)   : {mse:.2f}")
print(f"Root Mean Sq. Error (RMSE) : {rmse:.2f} score points")
print(f"R2 Score (Goodness of Fit) : {r2:.4f} ({r2 * 100:.2f}% variance explained)\n")

# -----------------------------------------------------------------------------
# 6. COMPARISON TABLE: ACTUAL VS PREDICTED
# -----------------------------------------------------------------------------
results_df = pd.DataFrame({
    'Actual Score': y_test.values,
    'Predicted Score': np.round(y_pred, 2),
    'Absolute Error': np.round(np.abs(y_test.values - y_pred), 2)
})

print("=== ACTUAL VS PREDICTED COMPARISON ===")
print(results_df.to_string(index=False))