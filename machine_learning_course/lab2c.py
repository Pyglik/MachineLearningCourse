import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# TODO 5
data = []
with open("trainingdata.txt", "r") as csv_f:
    csv_reader = csv.reader(csv_f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)

data = np.array(data)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.scatter(X_train, y_train)
plt.show()

clf = Pipeline([
    ('poly', PolynomialFeatures(degree=12)),
    ('line', LinearRegression())
])
clf.fit(X_train, y_train)
plt.scatter(X_train, clf.predict(X_train))
plt.show()
