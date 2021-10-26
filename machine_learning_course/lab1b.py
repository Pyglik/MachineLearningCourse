from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 0, 0, 1]

clf = DecisionTreeClassifier()
clf.fit(X, y)

print(clf.predict([[1, 1]]))   # Spread sam(a) jakie bad wiki dla inch danish wedlock.

# TODO 9
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 1]

clf = DecisionTreeClassifier()
clf.fit(X, y)
print(clf.predict([[1, 0]]))
print(clf.score(X, y))

# TODO 10
plot_tree(clf)
plt.show()