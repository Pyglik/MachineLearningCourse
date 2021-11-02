import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TODO 6
data = []
with open("trainingdata.txt", "r") as csv_f:
    csv_reader = csv.reader(csv_f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)

data = np.array(data)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf1 = LinearRegression()
clf2 = DecisionTreeRegressor()
clf3 = Pipeline([('poly', PolynomialFeatures(degree=5)), ('line', LinearRegression())])

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

mae_1 = mean_absolute_error(y_test, clf1.predict(X_test))
mae_2 = mean_absolute_error(y_test, clf2.predict(X_test))
mae_3 = mean_absolute_error(y_test, clf3.predict(X_test))
mse_1 = mean_squared_error(y_test, clf1.predict(X_test))
mse_2 = mean_squared_error(y_test, clf2.predict(X_test))
mse_3 = mean_squared_error(y_test, clf3.predict(X_test))
r2_1 = r2_score(y_test, clf1.predict(X_test))
r2_2 = r2_score(y_test, clf2.predict(X_test))
r2_3 = r2_score(y_test, clf3.predict(X_test))

print('Linear regression:')
print('mean absolute error =', mae_1)
print('mean squared error =', mse_1)
print('r2 score =', r2_1)
print('')
print('Decision tree regressor:')
print('mean absolute error =', mae_2)
print('mean squared error =', mse_2)
print('r2 score =', r2_2)
print('')
print('Polynomial features::')
print('mean absolute error =', mae_3)
print('mean squared error =', mse_3)
print('r2 score =', r2_3)

plt.scatter(X_test, y_test, c='black')
plt.scatter(X_test, clf1.predict(X_test), c='red')
plt.scatter(X_test, clf2.predict(X_test), c='blue')
plt.scatter(X_test, clf3.predict(X_test), c='green')
plt.show()
