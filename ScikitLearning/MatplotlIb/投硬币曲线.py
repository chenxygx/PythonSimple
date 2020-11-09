import matplotlib.pyplot as plt
import numpy as np

n_person = 2000
n_times = 500
t = np.arange(n_times)
# 创建-1 和 1两种类型元素表示输赢序列
steps = 2 * np.random.random_integers(0, 1, (n_person, n_times)) - 1
amount = np.cumsum(steps, axis=1)
sd_amount = amount ** 2
mean_sd_amount = sd_amount.mean(axis=0)

plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")
plt.plot(t, np.sqrt(mean_sd_amount), 'g.', t, np.sqrt(t), 'r-')
plt.show()