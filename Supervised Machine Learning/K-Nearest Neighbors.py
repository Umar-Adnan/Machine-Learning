import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. DATASET (Historical Training Data)
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 8, 3, 7],
    'Sleep_Hours': [5, 4, 6, 5, 7, 8, 6, 7, 8, 9, 8, 7, 5, 6, 4, 8],
    'Passed':      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Sleep_Hours']]
y = df['Passed']

# 2. FEATURE SCALING (Fit on 100% of the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TRAIN KNN MODEL ON ENTIRE DATASET
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)  # Model uses all 16 rows to learn

print("--- Model successfully trained on full dataset ---")

# 4. TAKE INTERACTIVE USER INPUT
try:
    user_study = float(input("Enter hours studied: "))
    user_sleep = float(input("Enter hours slept: "))

    # 5. FORMAT AND SCALE USER INPUT
    # Double brackets [[ ]] create a 2D array: shape (1, 2)
    user_data = pd.DataFrame([[user_study, user_sleep]], columns=['Study_Hours', 'Sleep_Hours'])
    user_data_scaled = scaler.transform(user_data)

    # 6. PREDICT & EXTRACT VALUE
    # Adding [0] extracts the single scalar output from the array
    prediction = knn.predict(user_data_scaled)[0]
    probability = knn.predict_proba(user_data_scaled)[0][1] * 100

    # 7. OUTPUT RESULT
    status = "PASS" if prediction == 1 else "FAIL"
    print(f"\nResult: {status}")
    print(f"Pass Probability: {probability:.1f}%")

except ValueError:
    print("Please enter valid numerical values for hours.")