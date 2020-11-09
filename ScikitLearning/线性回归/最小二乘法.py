import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def make_data(nDim):
    """
    维度增加
    :param nDim: 维度
    :return:  增加后数据
    """
    # 一个维度特征
    x0 = np.linspace(1, np.pi, 50)
    # nDim个维度的特征
    x = np.vstack([[x0, ], [i ** x0 for i in range(2, nDim + 1)]])
    # 目标值
    y = np.sin(x0) + np.random.normal(0, 0.15, len(x0))
    return x.transpose(), y


# 创建50个12维样本
x, y = make_data(500)


def linear_regreesion():
    # 训练维度
    dims = [1, 3, 6, 500]
    for idx, i in enumerate(dims):
        plt.subplot(2, len(dims) / 2, idx + 1)
        reg = linear_model.LinearRegression()
        # 取x前i个维度特征
        sub_x = x[:, 0:i]
        reg.fit(sub_x, y)
        # 绘制模型
        plt.plot(x[:, 0], reg.predict(sub_x))
        plt.plot(x[:, 0], y, ".")
        plt.title("dim=%s" % i)
        print("dim %d" % i)
        print("截距参数: %s" % reg.intercept_)
        print("回归参数: %s" % reg.coef_)
    plt.show()


linear_regreesion()
