import numpy as np

a = np.arange(1, 101)
n_max = int(np.sqrt(len(a)))
# 创建100个元素的数组，用来标记是否素数
is_prime = np.ones(len(a), dtype=bool)
is_prime[0] = False
for i in range(2, n_max):
    if i in a[is_prime]:
        is_prime[(i ** 2 - 1)::i] = False

print(a[is_prime])
