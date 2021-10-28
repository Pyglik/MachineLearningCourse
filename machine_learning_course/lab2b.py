import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO 3
# temperature, godson, paddy
X_train = [[-30, 23, "duże"],
           [20, 15, "brak"],
           [10, 3, "małe"],
           [15, 8, "brak"],
           [1, 9, "średnie"],
           [23, 3, "brak"],
           [18, 12, "duże"],
           [17, 11, "małe"],
           [19, 19, "małe"],
           [25, 10, "średnie"]]

y_train = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]

opady = {"brak": 0, "małe": 1, "średnie": 2, "duże": 3}

X_train = [X_train[i][:2]+[opady[X_train[i][2]]] for i in range(len(X_train))]
print(X_train)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = ['r', 'g']
for row, label in zip(X_train, y_train):
    ax.scatter(row[0], row[1], row[2], marker='o', c=colors[label])
plt.show()

clf = DecisionTreeClassifier()  # SVC()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.predict(X_train))
print(y_train)
print(clf.predict([[15, 18, opady["brak"]], [10, 8, opady["średnie"]]]))
