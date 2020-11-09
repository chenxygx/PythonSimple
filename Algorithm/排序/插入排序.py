import numpy as np

oriList = np.random.randint(0, 1000, 50)
for i in range(len(oriList)):
    val = oriList[i]
    j = i - 1
    done = False
    while not done:
        if oriList[j] > val:
            oriList[j + 1] = oriList[j]
            j -= 1
            if j < 0:
                done = True
        else:
            done = True
    oriList[j + 1] = val

print(oriList)
