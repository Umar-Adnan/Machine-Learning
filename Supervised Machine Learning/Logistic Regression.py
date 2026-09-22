from sklearn.linear_model import LogisticRegression
X = [[1], [2], [3], [4], [5], [6]] #inputs studied hours
y=[0,0,0,1,1,1] #result 0 for fail, 1 for pass

model = LogisticRegression()
model.fit(X, y)
hours = float(input("Enter how many hours you studied: "))
prediction = model.predict([[hours]])[0]

if prediction == 1:
    print("The model predicts that you are likely to Pass")
else:
    print("The model predicts that you are likely to Fail.")