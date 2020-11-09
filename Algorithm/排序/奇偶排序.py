import numpy as np

oriList = np.random.randint(0, 1000, 50)
sorted = False
while not sorted:
    sorted = True
    for i in range(1, len(oriList) - 1, 2):
        if oriList[i] > oriList[i + 1]:
            S = oriList[i + 1]
            oriList[i + 1] = oriList[i]
            oriList[i] = S
            sorted = False
    for i in range(0, len(oriList) - 1, 2):
        if oriList[i] > oriList[i + 1]:
            S = oriList[i + 1]
            oriList[i + 1] = oriList[i]
            oriList[i] = S
            sorted = False

print(oriList)
