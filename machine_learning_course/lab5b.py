import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import missingno as msno


def predict(x):
    return [1 if random.random() < 0.5 else 0 for i in np.array(x)]


titanic = pd.read_csv('RMS_Titanic.csv')
titanic.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)

titanic.replace('?', np.NaN, inplace=True)
msno.bar(titanic)
plt.show()

print(titanic.loc[titanic['embarked'] == np.NaN]['name'])
titanic['embarked'].iloc[168] = 'S'
titanic['embarked'].iloc[284] = 'S'

print(titanic.loc[titanic['fare'] == np.NaN]['name'])
titanic.drop(1225, inplace=True)

titanic['age'] = pd.to_numeric(titanic['age'], downcast='float')
titanic.loc[titanic['survived'] == 0, 'age'].hist()
plt.show()
titanic.drop(['cabin'], axis=1, inplace=True)

ages_titanic = titanic.groupby(['sex', 'pclass'])['age'].median().round(1)
print(ages_titanic)

for row, passenger in titanic.loc[np.isnan(titanic['age'])].iterrows():
    titanic['age'].iloc[row] = ages_titanic[passenger.sex][passenger.pclass]
titanic.dropna(inplace=True)
msno.bar(titanic)
plt.show()

titanic['sex'].loc[titanic['sex'] == 'female'] = 1
titanic['sex'].loc[titanic['sex'] == 'male'] = 0

titanic['pclass'].loc[titanic['pclass'] == 3] = 0
titanic['pclass'].loc[titanic['pclass'] == 2] = 0.5

titanic['sex'] = titanic['sex'].astype(float)
titanic['fare'] = titanic['fare'].astype(float)
titanic['pclass'] = titanic['pclass'].astype(float)

titanic['family_size'] = titanic['sibsp'] + titanic['parch']
titanic['family_size'] = titanic['family_size'].astype(float)
titanic.drop(['sibsp', 'parch'], axis=1, inplace=True)

titanic.drop(['name', 'ticket', 'embarked'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(titanic.drop(['survived'], axis=1), titanic['survived'],
                                                    test_size=0.1, stratify=titanic['survived'], random_state=42)

param_grid = [
    {"n_estimators": [1, 10, 100, 1000],
     "max_depth": [None, 1, 10, 10],
     "class_weight": ["balanced", None]}
]

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
clf = GridSearchCV(rf, param_grid)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.score(X_test, y_test))