from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# TODO 2
digits = datasets.load_digits()
print(digits.target.shape)
print(digits.data.shape)
print(digits.images.shape)

plt.imshow(digits.images[0], cmap='gray_r')
plt.show()

elements = 5
fig, axs = plt.subplots(len(digits.target_names), elements)
for nr in range(len(digits.target_names)):
    for i in range(elements):
        axs[nr][i].imshow(digits.images[digits.target == nr][i], cmap='gray_r')
        axs[nr][i].axis('off')
plt.show()

# TODO 3
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# TODO 4
faces = datasets.fetch_olivetti_faces()
print(faces.target.shape)
print(faces.data.shape)
print(faces.images.shape)
plt.imshow(faces.images[0], cmap='gray')
plt.show()

X, y = faces.data, faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
