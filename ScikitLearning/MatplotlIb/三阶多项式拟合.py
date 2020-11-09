import matplotlib.pyplot as plt
import numpy as np

n_dots = 20
n_order = 3

x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots)
p = np.poly1d(np.polyfit(x, y, n_order))
print(p.coeffs)
# 画出拟合曲线
t = np.linspace(0, 1, 200)
plt.plot(x, y, 'ro', t, p(t), '-')
plt.show()
