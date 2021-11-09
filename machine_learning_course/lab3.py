import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd

# wczytanie datasetu i podział na zbiory z równymi ilościami klas (stratify)
X, y = datasets.load_iris(return_X_y=True, as_frame=True)
print(X.describe())
X = np.array([x[0:2] for x in np.array(X)])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(X_train)

# sprawdzenie liczebności klas
print('Zbiór treningowy:')
print(y_train.value_counts())
print('Zbiór testowy')
print(y_test.value_counts())

# wizualizacja danych
plt.scatter(np.array(X_train)[:, 0], np.array(X_train)[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.show()

# normalizacja (0-1)
skaler = MinMaxScaler()
skaler.fit(X_train)
X_train_n = skaler.transform(X_train)

plt.scatter(np.array(X_train_n)[:, 0], np.array(X_train_n)[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.show()

# standaryzacja (mu=0, sigma=1)
skaler = StandardScaler()
skaler.fit(X_train)
X_train_s = skaler.transform(X_train)

plt.scatter(np.array(X_train_s)[:, 0], np.array(X_train_s)[:, 1])
plt.axvline(x=0)
plt.axhline(y=0)
plt.show()

# porównaie klasyfikatorów z pipelinem
clf1 = Pipeline([
    ('skaler', MinMaxScaler()),
    ('svc', SVC())
])

clf1.fit(X_train, y_train)
print('MinMaxSkaler:')
print(clf1.predict(X_test[0:10]))
print(clf1.score(X_test, y_test))

clf2 = Pipeline([
    ('skaler', StandardScaler()),
    ('svc', SVC())
])

print('StandardSkaler:')
clf2.fit(X_train, y_train)
print(clf2.predict(X_test[0:10]))
print(clf2.score(X_test, y_test))

# wizualizacja przestrzeni decyzyjnej
# clf = SVC()
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, np.array(y_test)))
plot_decision_regions(X_test, np.array(y_test), clf=clf, legend=1)
plt.show()
