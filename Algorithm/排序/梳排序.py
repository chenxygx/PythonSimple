import numpy as np

oriList = np.random.randint(0, 1000, 50)
n = len(oriList)
swaps = 1
while n != 1 or swaps == 0:
    n = int(n / 1.3)
    if n < 1:
        n = 1
    i = 0
    swaps = 0
    while i + n < len(oriList):
        if oriList[i] > oriList[i + n]:
            S = oriList[i]
            oriList[i] = oriList[i + n]
            oriList[i + n] = S
            swaps = 1
        i += 1

print(oriList)
