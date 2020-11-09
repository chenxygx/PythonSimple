import numpy as np

def heapSort(list):
    for i in range(int((len(list) - 1) / 2), -1, -1):
        adjust(list, i, len(list) - 1)
    for i in range(len(list) - 1, 0, -1):
        S = list[i]
        list[i] = list[0]
        list[0] = S
        adjust(list, 0, i - 1)
    return list

def adjust(list, i, m):
    temp = list[i]
    j = i * 2 + 1
    while j <= m:
        if j < m and list[j] < list[j + 1]:
            j += 1
        if temp < list[j]:
            list[i] = list[j]
            i = j
            j = 2 * i + 1
        else:
            j = m + 1
    list[i] = temp

oriList = np.random.randint(0, 1000, 50)
print(heapSort(oriList))
