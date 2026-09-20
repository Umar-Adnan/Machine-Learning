from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("sample_data.csv")
df_label = df.copy() #Making a copy so that the original data does not change.
le = LabelEncoder()

df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"]) #fit means learn from the values and then transform them with number on categories like 1 or 0 on values like male or female
df_label["Passed_Encoded"] = le.fit_transform(df_label["Passed"]) #labels the column named Passed in sample_data.csv containing vlaues yes or no.

print("\n Label Encoded Data:\n")
print(df_label[["Name", "Gender", "Gender_Encoded", "Passed", "Passed_Encoded"]].head())

print("\n One-Hot Encoded Data (City):  ")
df_encoded = pd.get_dummies(df_label, columns=["City"], dtype = int)
# get_dummies breaks down a column and make subcolumns so that binary values can be assigned to it like 1/0 or True/False
# By default, it assigns boolean values but we can change it using "dtype = data_type"
# Like City_Chennai, City_Delhi and City_Mumbai, each having binary values easily understandable by the model.
print("\n", df_encoded)
