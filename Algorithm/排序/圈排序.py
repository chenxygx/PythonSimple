import numpy as np

oriList = np.random.randint(0, 1000, 50)

for i in range(len(oriList)):
    item = oriList[i]
    pos = i
    swapped = True
    while i != pos or swapped:
        to = 0
        swapped = False
        for j in range(len(oriList)):
            if j != i and oriList[j] < item:
                to += 1
        if pos != to:
            while pos != to and item == oriList[to]:
                to += 1
            temp = oriList[to]
            oriList[to] = item
            item = temp
            pos = to

print(oriList)
