import numpy as np
import matplotlib.pyplot as plt

# 1 3 10阶多项式来模拟过拟合和欠拟合

n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1


def plot_polynomial_fit(x, y, order):
    p = np.poly1d(np.polyfit(x, y, order))
    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'ro', t, p(t), '-', t, np.sqrt(t), 'r--')
    return p


plt.figure(figsize=(8, 10), dpi=200)
titles = ['Under Fitting', 'Fitting', 'Over Fitting']
models = [None, None, None]
for index, order in enumerate([1, 3, 10]):
    plt.subplot(3, 1, index + 1)
    models[index] = plot_polynomial_fit(x, y, order)
    plt.title(titles[index], fontsize=20)

plt.show()
