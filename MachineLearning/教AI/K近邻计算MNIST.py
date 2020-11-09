import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('../Data/train.csv')

x_train, x_test, y_train, y_test = train_test_split(train.iloc[0:, 1:], train.iloc[0:, 0], random_state=3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

correct = np.count_nonzero(knn.predict(x_test) == y_test)
print('精准度为：%.3f' % (correct / len(y_test)))
