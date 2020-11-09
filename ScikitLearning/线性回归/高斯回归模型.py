import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C, Product


def f(X):
    return X * np.sin(X) - X


X = np.linspace(0, 10, 20)
# 给训练样本添加噪声
y = f(X) + np.random.normal(0, 0.5, X.shape[0])
# 定义测试样本特征值
x = np.linspace(0, 10, 200)
# 定义两个核函数，并取他们的积
kernel = Product(C(0.1), RBF(10, (1e-2, 1e2)))
# 初始化模型：传入核函数对象、优化次数、噪声超参数
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.3)
gp.fit(X.reshape(-1, 1), y)  # 训练
y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)  # 预测
fig = plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x)=x\,\sin(x)-x$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 2 * sigma,
                         (y_pred + 2 * sigma)[::-1]]),
         alpha=.3, fc='b', label='95% confidence')
plt.legend(loc='lower left')
plt.show()
