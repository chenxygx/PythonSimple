from sklearn.linear_model import SGDClassifier

X = [[0, 0], [2, 1], [5, 4]]
y = [0, 2, 2]

# penalty: 损失函数惩罚项，取值none、l1、l2、elasticnet
# l2惩罚项：对应岭回归。l1惩罚性：对应Lasso回归
reg = SGDClassifier(penalty="l2", max_iter=100)
reg.fit(X, y)

print(reg.predict([[4, 3]]))
print(reg.intercept_)
print(reg.coef_)
