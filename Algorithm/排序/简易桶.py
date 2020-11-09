import numpy as np

oriList = np.random.randint(0, 1000, 50)
resultList = []
T = [0] * (max(oriList) + 1)
for i in range(len(oriList)):
    T[oriList[i]] += 1
for i in range(0, len(T) - 1):
    for j in range(0, T[i]):
        resultList.append(i)

print(resultList)
