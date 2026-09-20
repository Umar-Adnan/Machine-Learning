import seaborn as sns
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. LOAD DATASET & SELECT FEATURES
# -----------------------------------------------------------------------------
df = sns.load_dataset('titanic')
print("A sample to look at to understand the data set: \n")
print(df.head(), "\n", df.tail())
# Select relevant columns
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features].copy()
y = df['survived'].copy()

# -----------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (Pure Pandas/NumPy approach)
# -----------------------------------------------------------------------------
# Shuffle dataset indices deterministically
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X))

test_size = int(len(X) * 0.2)
test_idx = shuffled_indices[:test_size]
train_idx = shuffled_indices[test_size:]

# Split into train and test sets
X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

# -----------------------------------------------------------------------------
# 3. IMPUTATION (Handling Missing Values with Pandas)
# -----------------------------------------------------------------------------
# Calculate medians and modes STRICTLY from X_train to avoid data leakage
age_median = X_train['age'].median()
fare_median = X_train['fare'].median()
embarked_mode = X_train['embarked'].mode()[0]

# Fill missing values in X_train
X_train['age'] = X_train['age'].fillna(age_median)
X_train['fare'] = X_train['fare'].fillna(fare_median)
X_train['embarked'] = X_train['embarked'].fillna(embarked_mode)

# Fill missing values in X_test USING THE TRAIN MEDIANS/MODES
X_test['age'] = X_test['age'].fillna(age_median)
X_test['fare'] = X_test['fare'].fillna(fare_median)
X_test['embarked'] = X_test['embarked'].fillna(embarked_mode)

# -----------------------------------------------------------------------------
# 4. CATEGORICAL ENCODING (One-Hot Encoding with pd.get_dummies)
# -----------------------------------------------------------------------------
cat_cols = ['sex', 'embarked', 'pclass']

# One-Hot Encode categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, dtype=float)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, dtype=float)

# Align columns to ensure train and test have identical features
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# -----------------------------------------------------------------------------
# 5. FEATURE SCALING (Standardization with Pure Math)
# -----------------------------------------------------------------------------
num_cols = ['age', 'fare', 'sibsp', 'parch']

# Calculate mean and std dev strictly from training data: z = (x - mean) / std
train_mean = X_train_encoded[num_cols].mean()
train_std = X_train_encoded[num_cols].std()

# Scale X_train
X_train_encoded[num_cols] = (X_train_encoded[num_cols] - train_mean) / train_std

# Scale X_test using TRAIN statistics
X_test_encoded[num_cols] = (X_test_encoded[num_cols] - train_mean) / train_std

# -----------------------------------------------------------------------------
# 6. VERIFY OUTPUT
# -----------------------------------------------------------------------------

print("\n--- PURE PANDAS PREPROCESSING RESULTS ---")
print(f"Any remaining NaNs in X_train? {X_train_encoded.isna().sum().sum()}")
print(f"X_train Shape: {X_train_encoded.shape}")
print(f"X_test Shape: {X_test_encoded.shape}\n")

print("First 5 rows of fully preprocessed numerical matrix:")
print(X_train_encoded.head())