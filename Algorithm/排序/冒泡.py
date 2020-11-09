import numpy as np

oriList = np.random.randint(0, 1000, 50)
n = len(oriList) - 1
for i in range(n):
    for j in range(n, i, -1):
        if oriList[j - 1] > oriList[j]:
            S = oriList[j]
            oriList[j] = oriList[j - 1]
            oriList[j - 1] = S
print(oriList)
