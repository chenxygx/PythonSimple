import numpy as np


def quickSort(list, low, high):
    i, j = low, high
    if low > high:
        return list
    temp = list[low]
    while i != j:
        while list[j] >= temp:
            j -= 1
        while list[i] <= temp:
            i += 1
        if i < j:
            S = list[j]
            list[j] = list[i]
            list[i] = S
    list[low] = list[i]
    list[i] = temp
    quickSort(list, low, i - 1)
    quickSort(list, i + 1, high)
    return list


oriList = np.random.randint(1, 1000, 50)
print(quickSort(oriList, 0, len(oriList) - 1))
