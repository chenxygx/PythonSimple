from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# 导入iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 设置需要搜索的K
parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
knn = KNeighborsClassifier()

# 通过GridSearchCV搜索最好的K值
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(X, y)
print('最好的分数是：%.2f' % clf.best_score_, " 最好的K值是：", clf.best_params_)
