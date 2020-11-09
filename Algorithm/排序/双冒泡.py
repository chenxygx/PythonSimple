import numpy as np

oriList = np.random.randint(0, 1000, 50)
n = len(oriList)
st = -1
swapped = True
while st < n and swapped:
    n -= 1
    st += 1
    swapped = False
    for j in range(st, n):
        if oriList[j] > oriList[j + 1]:
            S = oriList[j]
            oriList[j] = oriList[j + 1]
            oriList[j + 1] = S
            swapped = True
    for j in range(n - 1, st - 1, -1):
        if oriList[j] > oriList[j + 1]:
            S = oriList[j]
            oriList[j] = oriList[j + 1]
            oriList[j + 1] = S
            swapped = True

print(oriList)
