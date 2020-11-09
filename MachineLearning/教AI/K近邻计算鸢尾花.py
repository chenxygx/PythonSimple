from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=2003)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

correct = np.count_nonzero((clf.predict(X_test) == y_test))
print("精准度为 : %.3f" % (correct / len(X_test)))
