from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
gnb = GaussianNB()
gnb.fit(iris.data, iris.target)
print(gnb.class_prior_)  # 模型先验概率
print(gnb.class_count_)  # 训练集标签数量
print(gnb.theta_)  # 高斯模型期望值
print(gnb.sigma_)  # 高斯模型方差
