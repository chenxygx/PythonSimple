import numpy as np

def QuickSort(list, left, right):
    right = right == 0 and len(list) - 1 or right
    i, j = left, right
    x = list[int((left + right) / 2)]
    while i <= j:
        while list[i] < x:
            i += 1
        while list[j] > x:
            j -= 1
        if i <= j:
            S = list[j]
            list[j] = list[i]
            list[i] = S
            i += 1
            j -= 1
    if left < j:
        QuickSort(list, left, j)
    if right > i:
        QuickSort(list, i, right)
    return list

oriList = np.random.randint(0, 10, 8)
print(QuickSort(oriList, 0, 0))
