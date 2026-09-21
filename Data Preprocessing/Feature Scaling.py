import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Create a sample dataset with features on vastly different scales
data = {
    'Age': [20, 30, 40, 50, 60],
    'Salary': [30000, 50000, 70000, 90000, 110000],
    'Score': [10, 85, 45, 95, 60]
}
df = pd.DataFrame(data)

# 2. Apply StandardScaler (Mean = 0, Std Dev = 1)
std_scaler = StandardScaler()
df_standardized = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

# 3. Apply MinMaxScaler (Scales values strictly between 0 and 1)
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)

# 4. Display Results
print("--- Original Data ---")
print(df)

print("\n--- StandardScaler Output (Z-score Scaling) ---")
print(df_standardized.round(3))

print("\n--- MinMaxScaler Output (0 to 1 Range) ---")
print(df_minmax.round(3))