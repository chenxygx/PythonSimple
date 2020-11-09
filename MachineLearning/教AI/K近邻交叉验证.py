import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# 导入数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# 搜索K值候选集
ks = [1, 3, 5, 7, 9, 11, 13, 15]
# K折交叉验证，KFold返回的是每一折中训练数据和验证数据的index
kf = KFold(n_splits=5, random_state=2001, shuffle=True)
# 保存当前的k值和对应的准确率值
best_k = ks[0]
best_score = 0
# 循环每一个k值
for k in ks:
    curr_score = 0
    for train_index, valid_index in kf.split(X):
        # 计算每一折的训练以及计算准确率
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_index], y[train_index])
        curr_score = curr_score + clf.score(X[valid_index], y[valid_index])
    # 求5折的平均准确率
    avg_score = curr_score / 5
    if avg_score > best_score:
        best_k = k
        best_score = avg_score
    print('目前的最好成绩是:%.2f' % best_score, '最好的 k:%d' % best_k)

print('经过交叉验证，最终的最佳k是:%d' % best_k)
