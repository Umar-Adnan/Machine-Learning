from sklearn.linear_model import LinearRegression
X = [[1], [2], [3], [4], [5], [6]]
y = [40, 50, 60, 68, 76, 80]

model = LinearRegression()

model.fit(X, y)
hours = float(input("Enter how many hours you studied: "))
prediction = model.predict([[hours]]).round(2)
print(f"Based on the studied hours: {hours}, you may score around: {prediction}")
