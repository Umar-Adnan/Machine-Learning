import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    "StudyHours": [1,2,3,4,5,6,7],
    "TestScore":  [40,50,60,80,90,100,130]
        }
df = pd.DataFrame(data)

std_Scalar = StandardScaler()
std_scalar = std_Scalar.fit_transform(df)
print("Standard Scalar Output: \n")
print(pd.DataFrame(std_scalar, columns=["StudyHours,", "TestScore"]))

minmax_scalar=MinMaxScaler()
minmax_scaled=minmax_scalar.fit_transform(df)
print("\nMinmaxScaled out: \n")
print(pd.DataFrame(minmax_scaled, columns=["StudyHours", "TestScore"]))

X = df[["StudyHours"]]
y = df[["TestScore"]]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data:\n ", X_train)
print("Testing Data:\n ", X_test)


print("Training Data:\n ", y_train)
print("Testing Data:\n ", y_test)