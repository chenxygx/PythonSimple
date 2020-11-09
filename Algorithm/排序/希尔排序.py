import numpy as np

oriList = np.random.randint(0, 1000, 50)
h = int(len(oriList) / 2)
while h > 0:
    for i in range(h, len(oriList)):
        temp = oriList[i]
        if temp < oriList[i - h]:
            for j in range(0, i, h):
                if temp < oriList[j]:
                    temp = oriList[j]
                    oriList[j] = oriList[i]
                    oriList[i] = temp
    h = int(h / 2)

print(oriList)
