import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import impute
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from mlxtend.plotting import plot_decision_regions

diabetes = pd.read_csv('dataset_37_diabetes.csv')
diabetes['class'].loc[diabetes['class'] == 'tested_positive'] = 1
diabetes['class'].loc[diabetes['class'] == 'tested_negative'] = 0

diabetes['class'] = diabetes['class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(['class'], axis=1), diabetes['class'],
                                                    test_size=0.25, stratify=diabetes['class'], random_state=42)

clf1 = SVC()
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier()

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

print(clf1.score(X_test, y_test))
print(clf2.score(X_test, y_test))
print(clf3.score(X_test, y_test))

print(clf1.predict(X_test[0:10]))
print(clf2.predict(X_test[0:10]))
print(clf3.predict(X_test[0:10]))

print(np.array(y_test[0:10]))

for a in ['plas', 'pres', 'skin', 'insu', 'mass']:
    X_train[a].loc[X_train[a] == 0] = np.NaN
    print(a+':', X_train[a].isna().sum()/len(X_train)*100)

clf1.fit(X_train.dropna(), y_train[X_train.notna().all(axis=1)])
print(clf1.score(X_test.dropna(), y_test[X_test.notna().all(axis=1)]))

clf1 = Pipeline([
    ('imputer', impute.KNNImputer()),
    ('svc', SVC())
])
clf1.fit(X_train, y_train)
print(clf1.score(X_test, y_test))

zscore = abs((diabetes - diabetes.mean())/diabetes.std())
diabetes = diabetes.loc[(zscore >= 3).any(axis=1)]
print(diabetes)

clf = IsolationForest()
clf.fit(diabetes[['mass', 'plas']])
# print(clf.predict(diabetes[['mass', 'plas']]).head)

plot_decision_regions(np.array(diabetes[['mass', 'plas']]), np.array(clf.predict(diabetes[['mass', 'plas']])), clf)
plt.show()
