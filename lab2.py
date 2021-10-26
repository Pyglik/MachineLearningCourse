from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# TODO 1
digits = datasets.load_digits()

X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit([X_train[0], X_train[2]], [y_train[0], y_train[2]])
print(clf.score([X_train[0], X_train[2]], [y_train[0], y_train[2]]))
print(clf.predict([X_train[0], X_train[2]]))
print([y_train[0], y_train[2]])

# TODO 2
faces = datasets.fetch_olivetti_faces()
X, y = faces.data, faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LinearSVC()  # verbose=True)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
