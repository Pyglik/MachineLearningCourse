from sklearn import svm, model_selection
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
                                                    test_size=0.25, stratify=iris['target'])

parameters = {
    'kernel': ('linear', 'rbf', 'sigmoid'),
    'C': [1, 5, 10, 15, 20, 25, 30]
}
clf = GridSearchCV(svm.SVC(), parameters, cv=10)
clf.fit(X_train, y_train)

pvt = pd.pivot_table(
    pd.DataFrame(clf.cv_results_),
    values='mean_test_score',
    index='param_kernel',
    columns='param_C'
)

ax = sns.heatmap(pvt)
plt.show()

with open('model.pickle', 'wb') as file:
    pickle.dump(clf.best_estimator_, file)
    # clf = pickle.load(file)

print(clf.best_estimator_.score(X_test, y_test))
